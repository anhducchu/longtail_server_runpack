"""Paired rho=1/scaffold sampling and optional canonical evaluation."""

from __future__ import annotations

import csv
import json
import math
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
    SAFE_NAME,
    VENDOR_ROOT,
    atomic_json,
    atomic_numpy,
    create_scaffold,
    load_training_data,
    rooted_path,
    sha256_file,
)
from scaffold.core import ScalarScaffold, alpha_bar_schedule, reverse_sample


FIXED_SWEEP = (
    ("pf_ode", "PF-ODE", 0.0),
    ("reduced_025", "Reduced", 0.25),
    ("reduced_05", "Reduced", 0.5),
    ("standard_sde", "Standard SDE", 1.0),
    ("increased_2", "Increased", 2.0),
)


def fixed_sweep_conditions(
    condition: str | None = None,
) -> tuple[tuple[str, str, float], ...]:
    """Return the full sweep or one named condition for parallel workers."""
    if condition is None:
        return FIXED_SWEEP
    selected = tuple(row for row in FIXED_SWEEP if row[0] == condition)
    if not selected:
        valid = ", ".join(row[0] for row in FIXED_SWEEP)
        raise ValueError(f"unknown fixed-sweep condition {condition!r}; choose from {valid}")
    return selected


def sweep_conditions(
    config: dict[str, Any], condition: str | None = None
) -> tuple[dict[str, Any], ...]:
    """Return validated sweep rows, with the legacy five-row sweep as default."""
    configured = config["experiment"].get("conditions")
    if configured is None:
        configured = [
            {"name": name, "label": label, "rho": rho}
            for name, label, rho in FIXED_SWEEP
        ]
    if not isinstance(configured, list) or not configured:
        raise ValueError("experiment.conditions must be a non-empty list")

    steps = int(config["diffusion"]["steps"])
    default_stride = int(config["sampling"]["stride"])
    rows = []
    for raw in configured:
        if not isinstance(raw, dict):
            raise ValueError("each experiment condition must be a mapping")
        name = str(raw.get("name", ""))
        if not SAFE_NAME.fullmatch(name):
            raise ValueError(f"invalid condition name: {name!r}")
        label = str(raw.get("label", name))
        rho = float(raw["rho"])
        if not math.isfinite(rho) or rho < 0.0:
            raise ValueError(f"condition {name}: rho must be finite and non-negative")
        stride = int(raw.get("stride", default_stride))
        if stride < 1 or steps % stride:
            raise ValueError(f"condition {name}: stride must divide diffusion.steps")
        window_value = raw.get("time_window")
        window = None
        if window_value is not None:
            if not isinstance(window_value, list) or len(window_value) != 2:
                raise ValueError(f"condition {name}: time_window must be [lower, upper]")
            window = (float(window_value[0]), float(window_value[1]))
            if not 0.0 <= window[0] < window[1] <= 1.0:
                raise ValueError(
                    f"condition {name}: time_window must satisfy 0 <= lower < upper <= 1"
                )
        rows.append({
            "name": name,
            "label": label,
            "rho": rho,
            "stride": stride,
            "grid_points": steps // stride,
            "time_window": window,
        })

    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("condition names must be unique")
    if condition is None:
        return tuple(rows)
    selected = tuple(row for row in rows if row["name"] == condition)
    if not selected:
        raise ValueError(f"unknown sweep condition {condition!r}; choose from {', '.join(names)}")
    return selected


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
    stride: int | None = None,
    rho_window: tuple[float, float] | None = None,
) -> Path:
    sampling = config["sampling"]
    condition_stride = int(stride or sampling["stride"])
    destination = sample_path(output, condition)
    destination.parent.mkdir(parents=True, exist_ok=True)
    provenance_path = destination.with_suffix(".provenance.json")
    provenance = {
        "schema": "fixed_scaffold_samples_v3",
        "condition": condition,
        "rho_over_a0": rho_override,
        "source_checkpoint": str(source_checkpoint),
        "source_sha256": source_sha256,
        "phi_sha256": phi_sha256,
        "num_images": len(labels),
        "seed": int(sampling["seed"]),
        "stride": condition_stride,
        "guidance_scale": float(sampling["guidance_scale"]),
        "guidance_formula": "eps_cond + omega * (eps_cond - eps_uncond)",
        "sampler": "generalized_ddim_variance",
        "variance_projection": "clip only when rho exceeds finite-step marginal budget",
        "sample_value_range": "zero_one",
        "paired_noise": True,
    }
    if rho_window is not None:
        provenance["time_window"] = list(rho_window)
    if reusable_sample(destination, provenance_path, provenance, config):
        print(f"[sample:{condition}] keeping {destination}")
        return destination

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
                    stride=condition_stride,
                    microbatch=int(config["model"]["microbatch"]),
                    generator=generator,
                    rho_override=rho_override,
                    rho_window=rho_window,
                    guidance_scale=float(sampling["guidance_scale"]),
                )
                generated[start:stop] = (
                    (images.clamp(-1.0, 1.0) + 1.0) / 2.0
                ).cpu().numpy()
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

    atomic_json(provenance_path, provenance)
    return destination


def reusable_sample(
    samples: Path,
    provenance_path: Path,
    expected: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        array = np.load(samples, mmap_mode="r")
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    expected_shape = (
        int(config["sampling"]["num_images"]),
        3,
        int(config["dataset"]["image_size"]),
        int(config["dataset"]["image_size"]),
    )
    return provenance == expected and array.shape == expected_shape


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


def sample_fixed_sweep(
    model: torch.nn.Module,
    source_checkpoint: Path,
    source_sha256: str,
    config: dict[str, Any],
    output: Path,
    device: torch.device,
    condition: str | None = None,
) -> None:
    """Generate all fixed A/A0 conditions, or one parallel-worker condition."""
    _, _, counts_array = load_training_data(config)
    counts = torch.as_tensor(counts_array, device=device)
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
    for row in sweep_conditions(config, condition):
        generate_condition(
            condition=row["name"],
            model=model,
            scaffold=None,
            rho_override=row["rho"],
            counts=counts,
            labels=labels,
            alpha_bar=alpha_bar,
            source_checkpoint=source_checkpoint,
            source_sha256=source_sha256,
            phi_sha256=None,
            config=config,
            output=output,
            device=device,
            stride=row["stride"],
            rho_window=row["time_window"],
        )


def evaluate_conditions(
    config: dict[str, Any],
    output: Path,
    conditions: tuple[str, ...] = ("rho_1", "learned"),
    *,
    mode: str = "detailed",
    cache_features: bool = True,
) -> None:
    if int(config["sampling"]["num_images"]) != 50_000:
        raise ValueError("canonical evaluator requires sampling.num_images=50000")
    labels = output / "samples" / "labels.npy"
    metrics_root = rooted_path(config["evaluation"]["metrics_root"])
    metrics_dir = output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for condition in conditions:
        samples = sample_path(output, condition)
        if not samples.is_file() or not labels.is_file():
            raise FileNotFoundError(f"sample both conditions first; missing {samples}")
        metric_output = metrics_dir / f"{condition}.json"
        per_class_output = metrics_dir / f"{condition}.per_class.json"
        if (
            metric_output.is_file()
            and per_class_output.is_file()
            and min(metric_output.stat().st_mtime, per_class_output.stat().st_mtime)
            >= samples.stat().st_mtime
        ):
            print(f"[evaluate:{condition}] keeping completed metrics")
            continue
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
            mode,
            "--device",
            str(config["evaluation"].get("device", "cuda")),
            "--inception-batch-size",
            str(config["evaluation"]["inception_batch_size"]),
            "--output",
            str(metric_output),
            "--per-class-output",
            str(per_class_output),
        ]
        feature_cache = metrics_dir / f"{condition}.features.npz"
        if cache_features or feature_cache.is_file():
            command.extend([
                "--generated-features-cache",
                str(feature_cache),
            ])
        print("[evaluate] " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


def write_fixed_sweep_table(output: Path, config: dict[str, Any]) -> Path:
    """Collect canonical evaluator artifacts into the requested comparison table."""
    rows = []
    conditions = sweep_conditions(config)
    custom = config["experiment"].get("conditions") is not None
    for condition in conditions:
        name = condition["name"]
        metric_path = output / "metrics" / f"{name}.json"
        group_path = output / "metrics" / f"{name}.per_class.json"
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))["metrics"]
        groups = json.loads(group_path.read_text(encoding="utf-8"))["groups"]
        row = {
            "Condition": condition["label"],
            "Grid": condition["grid_points"],
            "Window": (
                "global" if condition["time_window"] is None
                else f"[{condition['time_window'][0]:g},{condition['time_window'][1]:g})"
            ),
            "A/A0": condition["rho"],
            "Train thêm?": "No",
            "Overall FID": float(metrics["FID"]),
            "Many": float(groups["Many"]["FID"]),
            "Medium": float(groups["Medium"]["FID"]),
            "Few": float(groups["Few"]["FID"]),
            "Recall": float(metrics["Recall"]),
            "Class-valid": None,
        }
        if not custom:
            row.pop("Grid")
            row.pop("Window")
            row["Scaffold"] = row.pop("Condition")
        rows.append(row)

    json_path = output / "fixed_sweep_results.json"
    atomic_json(json_path, {
        "rows": rows,
        "class_valid_status": (
            "not measured: no independent frozen CIFAR-100 classifier checkpoint is available"
        ),
    })
    csv_path = output / "fixed_sweep_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if custom:
        lines = [
            "| Condition | Grid | Window | A/A0 | Overall FID ↓ | Many ↓ | Medium ↓ | Few ↓ | Recall ↑ |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    else:
        lines = [
            "| Scaffold | A/A0 | Train thêm? | Overall FID ↓ | Many ↓ | Medium ↓ | Few ↓ | Recall ↑ | Class-valid ↑ |",
            "|---|---:|:---:|---:|---:|---:|---:|---:|---:|",
        ]
    for row in rows:
        if custom:
            lines.append(
                f"| {row['Condition']} | {row['Grid']} | {row['Window']} | "
                f"{row['A/A0']:g} | {row['Overall FID']:.4f} | {row['Many']:.4f} | "
                f"{row['Medium']:.4f} | {row['Few']:.4f} | {row['Recall']:.4f} |"
            )
        else:
            lines.append(
                f"| {row['Scaffold']} | {row['A/A0']:g} | No | "
                f"{row['Overall FID']:.4f} | {row['Many']:.4f} | "
                f"{row['Medium']:.4f} | {row['Few']:.4f} | "
                f"{row['Recall']:.4f} | N/A |"
            )
    markdown_path = output / "fixed_sweep_results.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path
