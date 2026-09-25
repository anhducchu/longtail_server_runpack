"""Configuration, frozen model loading, data access, and safe local writes."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from scaffold.core import ScalarScaffold


ROOT = Path(__file__).resolve().parents[1]
SCAFFOLD_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = SCAFFOLD_ROOT / "outputs"
VENDOR_ROOT = ROOT / "third_party" / "CCUA-DDPM"
ENV_VALUE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))


def load_local_env() -> None:
    path = ROOT / ".env.local"
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return ENV_VALUE.sub(
            lambda match: os.environ.get(match.group(1)) or match.group(2) or "",
            value,
        )
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    return value


def load_config(path: Path) -> dict[str, Any]:
    load_local_env()
    config = expand_env(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    required = {"experiment", "dataset", "diffusion", "model", "training", "sampling", "evaluation"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"config is missing sections: {missing}")
    if int(config["sampling"]["num_images"]) % int(config["dataset"]["num_classes"]):
        raise ValueError("sampling.num_images must be divisible by dataset.num_classes")
    return config


def rooted_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def output_directory(config: dict[str, Any], requested_name: str | None) -> Path:
    name = requested_name or str(config["experiment"]["name"])
    if not SAFE_NAME.fullmatch(name) or name in {".", ".."}:
        raise ValueError("experiment name may contain only letters, numbers, '.', '_' and '-'")
    output = (OUTPUTS_ROOT / name).resolve()
    if not output.is_relative_to(OUTPUTS_ROOT.resolve()):
        raise ValueError("scaffold output must remain below scaffold/outputs")
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".pt", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npy", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def model_weights(payload: dict[str, Any], key: str) -> dict[str, torch.Tensor]:
    weights = payload.get(key)
    if not isinstance(weights, dict):
        raise KeyError(f"checkpoint has no {key!r} state dict; keys={list(payload)}")
    if any(str(name).startswith("module.") for name in weights):
        weights = {
            str(name).removeprefix("module."): value for name, value in weights.items()
        }
    return weights


def load_frozen_score(
    checkpoint_path: Path,
    config: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """Load EMA theta and prove that no score parameter remains trainable."""
    from model.model import UNet

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = config["model"]
    diffusion = config["diffusion"]
    model = UNet(
        T=int(diffusion["steps"]),
        ch=int(model_config["channels"]),
        ch_mult=list(model_config["channel_multipliers"]),
        attn=list(model_config["attention_levels"]),
        num_res_blocks=int(model_config["residual_blocks"]),
        dropout=float(model_config["dropout"]),
        cond=True,
        num_class=int(config["dataset"]["num_classes"]),
    )
    model.load_state_dict(
        model_weights(checkpoint, str(model_config.get("checkpoint_key", "ema_model"))),
        strict=True,
    )
    model.requires_grad_(False).to(device).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("score freeze invariant failed")
    return model


def load_training_data(config: dict[str, Any]) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    from dataset import ImbalanceCIFAR100

    dataset_config = config["dataset"]
    dataset = ImbalanceCIFAR100(
        root=str(rooted_path(dataset_config["root"])),
        imb_type="exp",
        imb_factor=float(dataset_config["imbalance_factor"]),
        rand_number=int(dataset_config["split_seed"]),
        train=True,
        download=False,
    )
    images = np.asarray(dataset.data, dtype=np.uint8)
    labels = np.asarray(dataset.targets, dtype=np.int64)
    number_of_classes = int(dataset_config["num_classes"])
    indices = [np.flatnonzero(labels == class_id) for class_id in range(number_of_classes)]
    counts = np.bincount(labels, minlength=number_of_classes).astype(np.float32)
    if any(len(class_indices) == 0 for class_indices in indices):
        raise ValueError("the long-tail training split contains an empty class")
    return images, indices, counts


def choose_batch(
    images: np.ndarray,
    class_indices: list[np.ndarray],
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = rng.integers(0, len(class_indices), size=batch_size, dtype=np.int64)
    image_indices = np.asarray(
        [rng.choice(class_indices[int(label)]) for label in labels], dtype=np.int64
    )
    tensor = (
        torch.from_numpy(np.ascontiguousarray(images[image_indices]))
        .to(device=device, dtype=torch.float32)
        .permute(0, 3, 1, 2)
        .div_(127.5)
        .sub_(1.0)
    )
    return tensor, torch.as_tensor(labels, dtype=torch.long, device=device)


def create_scaffold(config: dict[str, Any], device: torch.device) -> ScalarScaffold:
    training = config["training"]
    return ScalarScaffold(
        hidden_width=int(training["hidden_width"]),
        rho_min=float(training["rho_min"]),
        rho_max=float(training["rho_max"]),
    ).to(device)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
