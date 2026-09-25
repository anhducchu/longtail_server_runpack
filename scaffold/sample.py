"""Paired rho=1/scaffold sampling and optional canonical evaluation."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scaffold.common import (
    ROOT,
    VENDOR_ROOT,
    atomic_json,
    atomic_numpy,
    create_scaffold,
    load_training_data,
    rooted_path,
    sha256_file,
)
from scaffold.core import ScalarScaffold, alpha_bar_schedule, reverse_sample


def load_trained_scaffold(
    path: Path,
    source_sha256: str,
    config: dict[str, Any],
    device: torch.device,
) -> ScalarScaffold:
    if not path.is_file():
        raise FileNotFoundError(f"train the scaffold first; missing {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("source_sha256") != source_sha256:
        raise ValueError("phi.pt belongs to a different score checkpoint")
    scaffold = create_scaffold(config, device)
    scaffold.load_state_dict(payload["state_dict"], strict=True)
    return scaffold.requires_grad_(False).eval()


def sample_path(output: Path, condition: str) -> Path:
    return output / "samples" / f"{condition}.npy"


def generate_condition(
    *,
    condition: str,
    model: torch.nn.Module,
    scaffold: ScalarScaffold | None,
    rho_override: float | None,
    counts: torch.Tensor,
    labels: np.ndarray,
    alpha_bar: torch.Tensor,
    source_checkpoint: Path,
    source_sha256: str,
    phi_sha256: str | None,
    config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> Path:
    sampling = config["sampling"]
    destination = sample_path(output, condition)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".npy", dir=destination.parent
    )
    os.close(descriptor)
    generated = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(
            len(labels),
            3,
            int(config["dataset"]["image_size"]),
            int(config["dataset"]["image_size"]),
        ),
    )
    generator = torch.Generator(device=device).manual_seed(int(sampling["seed"]))
    batch_size = int(sampling["batch_size"])
    try:
        with torch.inference_mode():
            for start in range(0, len(labels), batch_size):
                stop = min(start + batch_size, len(labels))
                batch_labels = torch.as_tensor(
                    labels[start:stop], dtype=torch.long, device=device
                )
                images = reverse_sample(
                    model,
                    scaffold,
                    batch_labels,
                    counts,
                    alpha_bar,
                    image_size=int(config["dataset"]["image_size"]),
                    steps=int(config["diffusion"]["steps"]),
                    stride=int(sampling["stride"]),
                    microbatch=int(config["model"]["microbatch"]),
                    generator=generator,
                    rho_override=rho_override,
                )
                generated[start:stop] = images.clamp(-1.0, 1.0).cpu().numpy()
                if start == 0 or stop == len(labels) or (start // batch_size) % 50 == 0:
                    print(f"[sample:{condition}] {stop}/{len(labels)}", flush=True)
        generated.flush()
        del generated
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass

    atomic_json(
        destination.with_suffix(".provenance.json"),
        {
            "condition": condition,
            "source_checkpoint": str(source_checkpoint),
            "source_sha256": source_sha256,
            "phi_sha256": phi_sha256,
            "num_images": len(labels),
            "seed": int(sampling["seed"]),
            "stride": int(sampling["stride"]),
            "paired_noise": True,
        },
    )
    return destination


def sample_conditions(
    model: torch.nn.Module,
    source_checkpoint: Path,
    source_sha256: str,
    config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> None:
    _, _, counts_array = load_training_data(config)
    counts = torch.as_tensor(counts_array, device=device)
    phi_path = output / "phi.pt"
    scaffold = load_trained_scaffold(phi_path, source_sha256, config, device)
    phi_sha256 = sha256_file(phi_path)
    number_of_classes = int(config["dataset"]["num_classes"])
    number_of_images = int(config["sampling"]["num_images"])
    labels = np.arange(number_of_images, dtype=np.int64) % number_of_classes
    atomic_numpy(output / "samples" / "labels.npy", labels)
    diffusion = config["diffusion"]
    alpha_bar = alpha_bar_schedule(
        int(diffusion["steps"]),
        float(diffusion["beta_1"]),
        float(diffusion["beta_T"]),
        device=device,
    )
    shared = {
        "model": model,
        "counts": counts,
        "labels": labels,
        "alpha_bar": alpha_bar,
        "source_checkpoint": source_checkpoint,
        "source_sha256": source_sha256,
        "config": config,
        "output": output,
        "device": device,
    }
    generate_condition(
        condition="rho_1",
        scaffold=None,
        rho_override=1.0,
        phi_sha256=None,
        **shared,
    )
    generate_condition(
        condition="learned",
        scaffold=scaffold,
        rho_override=None,
        phi_sha256=phi_sha256,
        **shared,
    )


def evaluate_conditions(config: dict[str, Any], output: Path) -> None:
    if int(config["sampling"]["num_images"]) != 50_000:
        raise ValueError("canonical evaluator requires sampling.num_images=50000")
    labels = output / "samples" / "labels.npy"
    metrics_root = rooted_path(config["evaluation"]["metrics_root"])
    metrics_dir = output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for condition in ("rho_1", "learned"):
        samples = sample_path(output, condition)
        if not samples.is_file() or not labels.is_file():
            raise FileNotFoundError(f"sample both conditions first; missing {samples}")
        command = [
            sys.executable,
            "-m",
            "tools.evaluate_ccua",
            "--repo",
            str(VENDOR_ROOT),
            "--samples",
            str(samples),
            "--labels",
            str(labels),
            "--metrics-root",
            str(metrics_root),
            "--mode",
            "detailed",
            "--device",
            str(config["evaluation"].get("device", "cuda")),
            "--inception-batch-size",
            str(config["evaluation"]["inception_batch_size"]),
            "--generated-features-cache",
            str(metrics_dir / f"{condition}.features.npz"),
            "--output",
            str(metrics_dir / f"{condition}.json"),
            "--per-class-output",
            str(metrics_dir / f"{condition}.per_class.json"),
        ]
        print("[evaluate] " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
