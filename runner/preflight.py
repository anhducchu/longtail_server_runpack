"""Fail-closed checks for the machine, vendor source, and paper protocol."""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import LoadedCampaign
from .gpu import query_gpus
from .utils import sha256_file


SUPPORTED_OBJECTIVES = frozenset({"ddpm", "cbdm", "t2h", "ccua"})
CCUA_COMMIT = "baff9def6cfd553d4452d10eab0bac66ec2727aa"


@dataclass
class Check:
    level: str
    name: str
    message: str


def _check(name: str, errors: list[str], success: str) -> Check:
    return Check("ERROR" if errors else "PASS", name, "; ".join(errors) or success)


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _finite_float(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _close(value: object, expected: float) -> bool:
    return math.isclose(_finite_float(value, math.inf), expected)


def _load_vendor_manifest(repo_root: Path) -> tuple[dict, Check]:
    path = repo_root / "THIRD_PARTY_MANIFEST.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        component = manifest["components"]["ccua"]
        if set(manifest["components"]) != {"ccua"}:
            raise ValueError("only the CCUA-DDPM component may be active")
        if set(manifest.get("baseline_objectives", ())) != SUPPORTED_OBJECTIVES:
            raise ValueError("baseline_objectives must be ddpm/cbdm/t2h/ccua")
        if component.get("commit") != CCUA_COMMIT:
            raise ValueError(f"CCUA commit must be {CCUA_COMMIT}")
        if component.get("directory") != "CCUA-DDPM":
            raise ValueError("CCUA directory must be CCUA-DDPM")
    except Exception as exc:
        return {}, Check("ERROR", "vendor-manifest", f"{path}: {exc}")
    return manifest, Check(
        "PASS",
        "vendor-manifest",
        f"CCUA-DDPM pinned at {CCUA_COMMIT[:12]}; four baseline objectives declared",
    )


def _check_metric_assets(campaign: LoadedCampaign, repo_root: Path) -> list[Check]:
    if not any(task.eval.get("paper_metrics", False) for task in campaign.tasks):
        return []

    configured = os.environ.get("LTX_METRICS_ROOT", "").strip()
    metrics_root = (
        Path(configured).expanduser()
        if configured
        else repo_root / "CCUA-DDPM" / "stats"
    )
    if not metrics_root.is_absolute():
        metrics_root = (campaign.root / metrics_root).resolve()

    dataset = "cifar100"
    manifest_path = metrics_root / f"{dataset}.metric_manifest.json"
    verify_hashes = _as_bool(os.environ.get("LTX_PREFLIGHT_VERIFY_ASSET_HASHES", "0"))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset") != dataset:
            raise ValueError(f"dataset must be {dataset}")
        if int(manifest.get("num_images", -1)) != 50_000:
            raise ValueError("balanced reference must contain 50,000 images")
        if manifest.get("repository_commit") != CCUA_COMMIT:
            raise ValueError("assets were not built from the pinned CCUA source")

        assets = (
            ("feature_file", "feature_sha256"),
            ("labels_file", "labels_sha256"),
            ("fid_file", "fid_sha256"),
            ("vgg_feature_file", "vgg_feature_sha256"),
            ("vgg_radii_file", "vgg_radii_sha256"),
        )
        for file_key, hash_key in assets:
            filename = str(manifest.get(file_key, "")).strip()
            expected_hash = str(manifest.get(hash_key, "")).strip()
            if not filename or not expected_hash:
                raise ValueError(f"missing {file_key}/{hash_key}")
            path = metrics_root / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            if verify_hashes and sha256_file(path) != expected_hash:
                raise ValueError(f"sha256 mismatch: {path}")
    except Exception as exc:
        return [Check("ERROR", f"metric-assets:{dataset}", f"{manifest_path}: {exc}")]

    suffix = " with SHA-256 verification" if verify_hashes else ""
    return [
        Check(
            "PASS",
            f"metric-assets:{dataset}",
            f"balanced FID/PRD manifest and arrays present{suffix}",
        )
    ]


def _check_matrix(campaign: LoadedCampaign, contract: dict) -> Check:
    methods = set(contract.get("methods", ("ddpm", "cbdm", "ccua")))
    seeds = sorted(
        int(seed)
        for seed in contract.get(
            "seeds", campaign.raw.get("campaign", {}).get("paired_seeds", [0, 1, 2])
        )
    )
    expected = {(method, seed) for method in methods for seed in seeds}
    actual = {(task.method, int(task.seed)) for task in campaign.tasks}
    errors = []
    if len(campaign.tasks) != len(expected) or actual != expected:
        errors.append(f"expected {sorted(expected)}, found {sorted(actual)}")
    return _check(
        "native-cifar-matrix",
        errors,
        f"{len(methods)} methods × {len(seeds)} seeds = {len(campaign.tasks)} tasks",
    )


def _check_adapters(campaign: LoadedCampaign, contract: dict) -> Check:
    errors = []
    if str(contract.get("adapter", "ccua")).lower() != "ccua":
        errors.append("native_contract.adapter must be 'ccua'")
    if str(contract.get("repository", "ccua")).lower() != "ccua":
        errors.append("native_contract.repository must be 'ccua'")
    for task in campaign.tasks:
        prefix = f"{task.method}/seed{task.seed}"
        if task.adapter != "ccua":
            errors.append(f"{prefix}: adapter must be 'ccua'")
        if task.repository.get("directory") != "CCUA-DDPM":
            errors.append(f"{prefix}: repository must be 'CCUA-DDPM'")
    return _check(
        "native-cifar-adapters",
        errors,
        "all objectives use the CCUA adapter and CCUA-DDPM repository",
    )


def _check_objectives(campaign: LoadedCampaign, contract: dict) -> Check:
    errors = []
    unsupported = set(contract.get("methods", ())) - SUPPORTED_OBJECTIVES
    if unsupported:
        errors.append(f"unsupported contract objectives: {sorted(unsupported)}")

    for task in campaign.tasks:
        prefix = f"{task.method}/seed{task.seed}"
        objective = str(task.method_config.get("objective", task.method)).lower()
        if objective not in SUPPORTED_OBJECTIVES:
            errors.append(f"{prefix}: unsupported objective={objective!r}")
            continue
        if objective != task.method:
            errors.append(f"{prefix}: objective={objective!r}, expected {task.method!r}")
            continue

        if objective == "cbdm":
            tau = _finite_float(task.method_config.get("cb_tau", 1.0), -1.0)
            if tau <= 0:
                errors.append(f"{prefix}: cb_tau must be finite and > 0")

        if objective in {"ddpm", "cbdm", "t2h"}:
            for key in ("ccua_al", "ccua_ucl"):
                if _finite_float(task.method_config.get(key, 0.0), 1.0) != 0.0:
                    errors.append(f"{prefix}: {key} must be zero for {objective}")

        if objective == "ccua":
            for key in ("ccua_al", "ccua_ucl"):
                if _finite_float(task.method_config.get(key, 1.0), 0.0) <= 0:
                    errors.append(f"{prefix}: {key} must be finite and > 0")

    return _check(
        "native-cifar-objectives",
        errors,
        "DDPM/CBDM/T2H/CCUA objective dispatch is explicit",
    )


def _check_repository(campaign: LoadedCampaign, repo_root: Path | None) -> Check:
    repositories = campaign.raw.get("repositories", {})
    config = repositories.get("ccua", {})
    directory = str(config.get("directory", "CCUA-DDPM"))
    errors = []

    if set(repositories) != {"ccua"}:
        errors.append(f"repositories must contain only 'ccua', found {sorted(repositories)}")
    if directory != "CCUA-DDPM":
        errors.append("repositories.ccua.directory must be 'CCUA-DDPM'")
    if config.get("commit") != CCUA_COMMIT:
        errors.append(f"repositories.ccua.commit must be {CCUA_COMMIT}")

    if repo_root is not None:
        source = repo_root / directory / "main.py"
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"cannot read {source}: {exc}")
        else:
            required = (
                "GaussianDiffusionSamplerDDIM",
                'flags.DEFINE_bool("sample"',
                'flags.DEFINE_string("sample_output"',
                'flags.DEFINE_bool("uniform_labels"',
                "if FLAGS.sample:",
            )
            missing = [token for token in required if token not in text]
            if missing:
                errors.append(f"official sample entrypoint missing {missing}")

    return _check(
        "native-cifar-repository",
        errors,
        "pinned CCUA-DDPM source exposes the official DDIM sampler and array export",
    )


def _check_controls(campaign: LoadedCampaign, contract: dict) -> Check:
    updates = int(contract.get("train_updates", 300_000))
    batch_size = int(contract.get("train_batch_size", 64))
    learning_rate = float(contract.get("learning_rate", 2e-4))
    diffusion_steps = int(contract.get("diffusion_steps", 1000))
    generated_images = int(contract.get("generated_images", 50_000))
    guidance_scale = float(contract.get("guidance_scale", 1.5))
    sampler_family = str(contract.get("sampler_family", "ddim_100"))
    ddim_steps = (
        int(sampler_family.rsplit("_", 1)[-1])
        if sampler_family.startswith("ddim_")
        else -1
    )
    errors = []

    for task in campaign.tasks:
        prefix = f"{task.method}/seed{task.seed}"
        expected = {
            "dataset.name": (
                task.dataset.get("name"),
                contract.get("dataset", "cifar100lt_if100"),
            ),
            "dataset.data_type": (task.dataset.get("data_type"), "cifar100lt"),
            "dataset.num_class": (int(task.dataset.get("num_class", -1)), 100),
            "dataset.img_size": (int(task.dataset.get("img_size", -1)), 32),
            "dataset.split_seed": (int(task.dataset.get("split_seed", -1)), 0),
            "train.total_steps": (int(task.train.get("total_steps", -1)), updates),
            "train.batch_size": (int(task.train.get("batch_size", -1)), batch_size),
            "train.T": (int(task.train.get("T", -1)), diffusion_steps),
            "eval.checkpoint_step": (int(task.eval.get("checkpoint_step", -1)), updates),
            "eval.num_images": (int(task.eval.get("num_images", -1)), generated_images),
            "eval.uniform_labels": (bool(task.eval.get("uniform_labels", False)), True),
            "eval.sampler_family": (task.eval.get("sampler_family"), sampler_family),
            "eval.metric_protocol": (task.eval.get("metric_protocol"), "native_cifar_v1"),
        }
        for name, (actual, wanted) in expected.items():
            if actual != wanted:
                errors.append(f"{prefix}: {name}")

        if not _close(task.dataset.get("imbalance_factor"), 0.01):
            errors.append(f"{prefix}: dataset.imbalance_factor")
        if not _close(task.train.get("lr"), learning_rate):
            errors.append(f"{prefix}: train.lr")
        if not _close(task.eval.get("guidance_scale"), guidance_scale):
            errors.append(f"{prefix}: eval.guidance_scale")

        stride = int(task.eval.get("ddim_skip_step", -1))
        valid_sampler = (
            task.eval.get("sample_method") == "ddim"
            and stride > 0
            and diffusion_steps // stride == ddim_steps
        )
        if not valid_sampler:
            errors.append(f"{prefix}: CCUA official DDIM stride")

    return _check(
        "native-cifar-controls",
        errors,
        (
            f"{updates} updates, batch={batch_size}, T={diffusion_steps}, "
            f"DDIM-{ddim_steps}, N={generated_images}"
        ),
    )


def _check_native_cifar_contract(
    campaign: LoadedCampaign,
    repo_root: Path | None = None,
) -> list[Check]:
    """Validate the one dataset, one host, and four supported objectives."""
    if campaign.raw.get("campaign", {}).get("protocol") != "native_cifar_v1":
        return [
            Check(
                "ERROR",
                "protocol",
                "main supports only campaign.protocol=native_cifar_v1",
            )
        ]

    if repo_root is None:
        configured_root = campaign.server.get("runtime", {}).get("repos_root")
        repo_root = Path(configured_root) if configured_root else None
    contract = campaign.raw.get("native_contract", {})
    return [
        _check_matrix(campaign, contract),
        _check_adapters(campaign, contract),
        _check_objectives(campaign, contract),
        _check_repository(campaign, repo_root),
        _check_controls(campaign, contract),
    ]


def _check_gpu(campaign: LoadedCampaign) -> Check:
    gpus = query_gpus()
    required = float(campaign.server.get("machine", {}).get("min_free_gpu_memory_gb", 0))
    eligible = [gpu for gpu in gpus if gpu.memory_free_mb / 1024 >= required]
    if not gpus:
        message = f"nvidia-smi returned no GPUs; required={required:.1f}GB"
    else:
        available = ", ".join(
            f"{gpu.index}:{gpu.name} {gpu.memory_free_mb / 1024:.1f}GB free"
            for gpu in gpus
        )
        message = f"{available}; required={required:.1f}GB"
    return Check("PASS" if eligible else "ERROR", "gpu", message)


def _check_disk(campaign: LoadedCampaign) -> Check:
    runs_root = Path(campaign.server["runtime"]["runs_root"])
    target = runs_root if runs_root.exists() else campaign.root
    free_gb = shutil.disk_usage(target).free / 1024**3
    guard = float(campaign.server.get("machine", {}).get("disk_stop_free_gb", 0))
    return Check(
        "PASS" if free_gb >= guard else "ERROR",
        "disk",
        f"{free_gb:.1f} GB free; guard={guard:.1f} GB",
    )


def _check_wandb(campaign: LoadedCampaign) -> Check:
    runtime = campaign.server["runtime"]
    mode = runtime.get("wandb_mode", os.getenv("WANDB_MODE", "online"))
    ready = mode != "online" or bool(os.getenv("WANDB_API_KEY"))
    message = (
        f"mode={mode}, project={runtime.get('wandb_project', 'longtail')}"
        if ready
        else "online mode but WANDB_API_KEY is empty"
    )
    return Check("PASS" if ready else "ERROR", "wandb", message)


def run_preflight(campaign: LoadedCampaign) -> list[Check]:
    repo_root = Path(campaign.server["runtime"]["repos_root"])
    _, vendor = _load_vendor_manifest(repo_root)
    return [
        _check_gpu(campaign),
        _check_disk(campaign),
        _check_wandb(campaign),
        vendor,
        *_check_native_cifar_contract(campaign, repo_root),
        *_check_metric_assets(campaign, repo_root),
    ]
