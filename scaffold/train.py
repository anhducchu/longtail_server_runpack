"""Stage 2: optimize scaffold phi while score theta remains frozen."""

from __future__ import annotations

import csv
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scaffold.common import (
    atomic_torch_save,
    choose_batch,
    create_scaffold,
    load_training_data,
)
from scaffold.core import (
    ScalarScaffold,
    alpha_bar_schedule,
    budget_loss,
    effective_class_weights,
    forward_path_pair,
    sliced_wasserstein,
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def checkpoint_payload(
    scaffold: ScalarScaffold,
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    source_checkpoint: Path,
    source_sha256: str,
    config: dict[str, Any],
    data_rng: np.random.Generator,
    path_generator: torch.Generator,
    prior_generator: torch.Generator,
) -> dict[str, Any]:
    payload = {
        "schema": "frozen_score_scaffold_v1",
        "update": update,
        "state_dict": scaffold.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "source_checkpoint": str(source_checkpoint),
        "source_sha256": source_sha256,
        "training_signature": training_signature(config),
        "config": config,
        "data_rng_state": data_rng.bit_generator.state,
        "path_generator_state": path_generator.get_state(),
        "prior_generator_state": prior_generator.get_state(),
        "torch_cpu_rng_state": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda_rng_state"] = torch.cuda.get_rng_state()
    return payload


def training_signature(config: dict[str, Any]) -> dict[str, Any]:
    """Fields that must stay fixed when phi training is resumed."""
    training = dict(config["training"])
    for extendable in ("updates", "save_every", "log_every"):
        training.pop(extendable, None)
    return {
        "dataset": config["dataset"],
        "diffusion": config["diffusion"],
        "model": config["model"],
        "training": training,
        "score_guidance_scale": float(config["sampling"]["guidance_scale"]),
    }


def restore_checkpoint(
    path: Path,
    scaffold: ScalarScaffold,
    optimizer: torch.optim.Optimizer,
    source_sha256: str,
    config: dict[str, Any],
    data_rng: np.random.Generator,
    path_generator: torch.Generator,
    prior_generator: torch.Generator,
    device: torch.device,
) -> int:
    if not path.is_file():
        return 1
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("source_sha256") != source_sha256:
        raise ValueError("existing phi.pt was trained from a different score checkpoint")
    if saved.get("training_signature") != training_signature(config):
        raise ValueError("training config changed; use a new --name instead of mixing runs")
    scaffold.load_state_dict(saved["state_dict"], strict=True)
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    data_rng.bit_generator.state = saved["data_rng_state"]
    path_generator.set_state(saved["path_generator_state"])
    prior_generator.set_state(saved["prior_generator_state"])
    torch.random.set_rng_state(saved["torch_cpu_rng_state"])
    if device.type == "cuda" and "torch_cuda_rng_state" in saved:
        torch.cuda.set_rng_state(saved["torch_cuda_rng_state"], device=device)
    return int(saved["update"]) + 1


def train_scaffold(
    model: torch.nn.Module,
    source_checkpoint: Path,
    source_sha256: str,
    config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> Path:
    """Optimize phi only; theta stays frozen but differentiable in its input."""
    training = config["training"]
    diffusion = config["diffusion"]
    images, class_indices, counts_array = load_training_data(config)
    counts = torch.as_tensor(counts_array, device=device)
    class_weights = effective_class_weights(
        counts, float(training["rarity_weight_exponent"])
    )
    alpha_bar = alpha_bar_schedule(
        int(diffusion["steps"]),
        float(diffusion["beta_1"]),
        float(diffusion["beta_T"]),
        device=device,
    )
    scaffold = create_scaffold(config, device)
    optimizer = torch.optim.Adam(
        scaffold.parameters(), lr=float(training["learning_rate"])
    )
    phi_path = output / "phi.pt"
    log_path = output / "training.csv"
    seed = int(config["experiment"]["seed"])
    data_rng = np.random.default_rng(seed)
    path_generator = torch.Generator(device=device).manual_seed(seed + 1)
    prior_generator = torch.Generator(device=device).manual_seed(seed + 2)
    rows = read_rows(log_path)
    start_update = restore_checkpoint(
        phi_path,
        scaffold,
        optimizer,
        source_sha256,
        config,
        data_rng,
        path_generator,
        prior_generator,
        device,
    )

    updates = int(training["updates"])
    if start_update > updates:
        print(f"[train] keeping completed {phi_path}")
        return phi_path

    started = time.monotonic()
    for update in range(start_update, updates + 1):
        clean_images, labels = choose_batch(
            images,
            class_indices,
            int(training["batch_size"]),
            data_rng,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        paths = forward_path_pair(
            model,
            scaffold,
            clean_images,
            labels,
            counts,
            alpha_bar,
            steps=int(diffusion["steps"]),
            stride=int(training["stride"]),
            microbatch=int(config["model"]["microbatch"]),
            generator=path_generator,
            guidance_scale=float(config["sampling"]["guidance_scale"]),
        )
        difference = (paths[0] - paths[1]).flatten(1)
        per_source = difference.square().mean(dim=1) / 2.0
        coupling = (per_source * class_weights.index_select(0, labels)).mean()
        budget, mean_rho = budget_loss(
            scaffold,
            counts,
            time_samples=int(training["budget_time_samples"]),
        )
        flat_paths = paths.flatten(0, 1)
        prior = sliced_wasserstein(
            flat_paths,
            torch.randn(
                flat_paths.shape,
                device=device,
                generator=prior_generator,
            ),
            projections=int(training["prior_projections"]),
            generator=prior_generator,
        )
        loss = (
            coupling
            + float(training["budget_weight"]) * budget
            + float(training["prior_weight"]) * prior
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite scaffold loss at update {update}")
        loss.backward()

        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("frozen score parameter received a gradient")
        gradients = [
            parameter.grad for parameter in scaffold.parameters() if parameter.grad is not None
        ]
        if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError(f"invalid scaffold gradient at update {update}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            scaffold.parameters(), float(training["gradient_clip"])
        )
        optimizer.step()

        row = {
            "update": update,
            "loss": float(loss.detach()),
            "coupling": float(coupling.detach()),
            "budget": float(budget.detach()),
            "prior_swd": float(prior.detach()),
            "mean_rho": float(mean_rho.detach()),
            "gradient_norm": float(gradient_norm),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        if update == 1 or update % max(1, int(training["log_every"])) == 0 or update == updates:
            rows.append(row)
            print(f"[train] {row}", flush=True)
        if update % max(1, int(training["save_every"])) == 0 or update == updates:
            write_rows(log_path, rows)
            atomic_torch_save(
                phi_path,
                checkpoint_payload(
                    scaffold,
                    optimizer,
                    update=update,
                    source_checkpoint=source_checkpoint,
                    source_sha256=source_sha256,
                    config=config,
                    data_rng=data_rng,
                    path_generator=path_generator,
                    prior_generator=prior_generator,
                ),
            )
    return phi_path
