from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

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


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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
            raise ValueError(
                f"CCUA commit={component.get('commit')!r}, expected {CCUA_COMMIT}"
            )
        if component.get("directory") != "CCUA-DDPM":
            raise ValueError("CCUA directory must be CCUA-DDPM")
    except Exception as exc:
        return {}, Check("ERROR", "vendor-manifest", f"{path}: {exc}")
    return manifest, Check(
        "PASS",
        "vendor-manifest",
        f"CCUA-DDPM pinned at {component['commit'][:12]}; four baseline objectives declared",
    )


def _check_metric_assets(campaign: LoadedCampaign, repo_root: Path) -> List[Check]:
    datasets = {
        str(task.dataset.get("data_type", ""))
        for task in campaign.tasks
        if task.eval.get("paper_metrics", False)
    }
    wanted: set[str] = set()
    if datasets & {"cifar10", "cifar10lt"}:
        wanted.add("cifar10")
    if datasets & {"cifar100", "cifar100lt"}:
        wanted.add("cifar100")
    if not wanted:
        return []

    configured = os.environ.get("LTX_METRICS_ROOT", "").strip()
    metrics_root = Path(configured).expanduser() if configured else repo_root / "CCUA-DDPM" / "stats"
    if not metrics_root.is_absolute():
        metrics_root = (campaign.root / metrics_root).resolve()

    checks: List[Check] = []
    verify_hashes = _as_bool(os.environ.get("LTX_PREFLIGHT_VERIFY_ASSET_HASHES", "0"))
    for dataset in sorted(wanted):
        manifest_path = metrics_root / f"{dataset}.metric_manifest.json"
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if payload.get("dataset") != dataset:
                raise ValueError(f"dataset={payload.get('dataset')!r}, expected {dataset!r}")
            if int(payload.get("num_images", -1)) != 50_000:
                raise ValueError("balanced reference must contain 50,000 images")
            if payload.get("repository_commit") != CCUA_COMMIT:
                raise ValueError("metric assets were not built from the pinned CCUA source")
            assets = (
                ("feature_file", "feature_sha256"),
                ("labels_file", "labels_sha256"),
                ("fid_file", "fid_sha256"),
                ("vgg_feature_file", "vgg_feature_sha256"),
                ("vgg_radii_file", "vgg_radii_sha256"),
            )
            for file_key, hash_key in assets:
                filename = str(payload.get(file_key, "")).strip()
                expected_hash = str(payload.get(hash_key, "")).strip()
                if not filename or not expected_hash:
                    raise ValueError(f"missing {file_key}/{hash_key}")
                path = metrics_root / filename
                if not path.is_file():
                    raise FileNotFoundError(path)
                if verify_hashes and sha256_file(path) != expected_hash:
                    raise ValueError(f"sha256 mismatch: {path}")
        except Exception as exc:
            checks.append(Check("ERROR", f"metric-assets:{dataset}", f"{manifest_path}: {exc}"))
        else:
            suffix = " with SHA-256 verification" if verify_hashes else ""
            checks.append(Check(
                "PASS",
                f"metric-assets:{dataset}",
                f"balanced FID/PRD manifest and arrays present{suffix}",
            ))
    return checks


def _check_native_cifar_contract(
    campaign: LoadedCampaign, repo_root: Path | None = None
) -> List[Check]:
    """Validate the single-host CIFAR baseline contract before GPU work."""
    if campaign.raw.get("campaign", {}).get("protocol") != "native_cifar_v1":
        return [Check(
            "ERROR",
            "protocol",
            "main supports only campaign.protocol=native_cifar_v1",
        )]

    checks: List[Check] = []
    contract = campaign.raw.get("native_contract", {})
    expected_methods = set(contract.get("methods", ("ddpm", "cbdm", "ccua")))
    unsupported = expected_methods - SUPPORTED_OBJECTIVES
    if unsupported:
        checks.append(Check(
            "ERROR",
            "native-cifar-objectives",
            f"unsupported objective(s) {sorted(unsupported)}; allowed={sorted(SUPPORTED_OBJECTIVES)}",
        ))

    expected_seeds = sorted(map(
        int,
        contract.get(
            "seeds",
            campaign.raw.get("campaign", {}).get("paired_seeds", [0, 1, 2]),
        ),
    ))
    tasks = campaign.tasks
    actual_methods = {task.method for task in tasks}
    actual_seeds = sorted({int(task.seed) for task in tasks})
    expected_pairs = {
        (method, seed) for method in expected_methods for seed in expected_seeds
    }
    actual_pairs = {(task.method, int(task.seed)) for task in tasks}
    if len(tasks) != len(expected_pairs) or actual_pairs != expected_pairs:
        checks.append(Check(
            "ERROR",
            "native-cifar-matrix",
            f"expected {sorted(expected_pairs)}, found {sorted(actual_pairs)}",
        ))
    else:
        checks.append(Check(
            "PASS",
            "native-cifar-matrix",
            f"{len(expected_methods)} methods × {len(expected_seeds)} seeds = {len(tasks)} tasks",
        ))
    if actual_methods != expected_methods or actual_seeds != expected_seeds:
        checks.append(Check(
            "ERROR",
            "native-cifar-methods",
            f"methods={sorted(actual_methods)} seeds={actual_seeds}",
        ))

    expected_adapter = str(contract.get("adapter", "ccua")).strip().lower()
    expected_repository = str(contract.get("repository", "ccua")).strip().lower()
    adapter_errors: list[str] = []
    if expected_adapter != "ccua":
        adapter_errors.append(f"native_contract.adapter={expected_adapter!r}, expected='ccua'")
    if expected_repository != "ccua":
        adapter_errors.append(
            f"native_contract.repository={expected_repository!r}, expected='ccua'"
        )
    for task in tasks:
        if task.adapter != "ccua":
            adapter_errors.append(
                f"{task.method}/seed{task.seed}: adapter={task.adapter!r}, expected='ccua'"
            )
        if task.repository.get("directory") != "CCUA-DDPM":
            adapter_errors.append(
                f"{task.method}/seed{task.seed}: repository="
                f"{task.repository.get('directory')!r}, expected='CCUA-DDPM'"
            )
    checks.append(Check(
        "ERROR" if adapter_errors else "PASS",
        "native-cifar-adapters",
        "; ".join(adapter_errors)
        if adapter_errors
        else "all objectives use the CCUA adapter and CCUA-DDPM repository",
    ))

    objective_errors: list[str] = []
    for task in tasks:
        prefix = f"{task.method}/seed{task.seed}"
        objective = str(task.method_config.get("objective", task.method)).strip().lower()
        if objective not in SUPPORTED_OBJECTIVES:
            objective_errors.append(
                f"{prefix}: unsupported objective={objective!r}; "
                f"allowed={sorted(SUPPORTED_OBJECTIVES)}"
            )
            continue
        if objective != task.method:
            objective_errors.append(
                f"{prefix}: objective={objective!r}, expected {task.method!r}"
            )
            continue
        if objective == "cbdm":
            try:
                cb_tau = float(task.method_config.get("cb_tau", 1.0))
            except (TypeError, ValueError):
                cb_tau = -1.0
            if not np.isfinite(cb_tau) or cb_tau <= 0:
                objective_errors.append(f"{prefix}: cb_tau must be finite and > 0")
        if objective in {"ddpm", "cbdm", "t2h"}:
            for key in ("ccua_al", "ccua_ucl"):
                try:
                    value = float(task.method_config.get(key, 0.0))
                except (TypeError, ValueError):
                    value = 1.0
                if value != 0.0:
                    objective_errors.append(f"{prefix}: {key} must be zero for {objective}")
        if objective == "ccua":
            for key in ("ccua_al", "ccua_ucl"):
                try:
                    value = float(task.method_config.get(key, 1.0))
                except (TypeError, ValueError):
                    value = 0.0
                if not np.isfinite(value) or value <= 0:
                    objective_errors.append(
                        f"{prefix}: {key} must be finite and > 0 for ccua"
                    )
    if not unsupported:
        checks.append(Check(
            "ERROR" if objective_errors else "PASS",
            "native-cifar-objectives",
            "; ".join(objective_errors)
            if objective_errors
            else "DDPM/CBDM/T2H/CCUA objective dispatch is explicit",
        ))

    if repo_root is None:
        runtime_root = campaign.server.get("runtime", {}).get("repos_root")
        repo_root = Path(runtime_root) if runtime_root else None
    repositories = campaign.raw.get("repositories", {})
    ccua_cfg = repositories.get("ccua", {})
    directory = str(ccua_cfg.get("directory", "CCUA-DDPM"))
    repository_errors: list[str] = []
    if set(repositories) != {"ccua"}:
        repository_errors.append(
            f"repositories must contain only 'ccua', found {sorted(repositories)}"
        )
    if directory != "CCUA-DDPM":
        repository_errors.append(
            f"repositories.ccua.directory={directory!r}, expected 'CCUA-DDPM'"
        )
    if ccua_cfg.get("commit") != CCUA_COMMIT:
        repository_errors.append(
            f"repositories.ccua.commit={ccua_cfg.get('commit')!r}, expected {CCUA_COMMIT}"
        )
    if repo_root is not None:
        source = repo_root / directory / "main.py"
        marker = repo_root / directory / ".ltx_ccua_sample_export_v1"
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as exc:
            repository_errors.append(f"cannot read {source}: {exc}")
        else:
            required = (
                "GaussianDiffusionSamplerDDIM",
                "flags.DEFINE_bool('sample'",
                "if FLAGS.sample:",
            )
            missing = [item for item in required if item not in text]
            if missing:
                repository_errors.append(f"official sample entrypoint missing {missing}")
        if not marker.is_file():
            repository_errors.append(
                "sample-output/uniform-label patch marker missing; run bootstrap"
            )
    checks.append(Check(
        "ERROR" if repository_errors else "PASS",
        "native-cifar-repository",
        "; ".join(repository_errors)
        if repository_errors
        else "pinned CCUA-DDPM source exposes the official DDIM sampler and array export",
    ))

    def close(value: object, expected: float) -> bool:
        try:
            return bool(np.isclose(float(value), expected))
        except (TypeError, ValueError):
            return False

    setting_errors: list[str] = []
    updates = int(contract.get("train_updates", 300_000))
    batch = int(contract.get("train_batch_size", 64))
    lr = float(contract.get("learning_rate", 2e-4))
    diffusion_steps = int(contract.get("diffusion_steps", 1000))
    generated = int(contract.get("generated_images", 50_000))
    guidance = float(contract.get("guidance_scale", 1.5))
    wanted_family = str(contract.get("sampler_family", "ddim_100"))
    wanted_steps = (
        int(wanted_family.rsplit("_", 1)[-1])
        if wanted_family.startswith("ddim_")
        else -1
    )
    for task in tasks:
        prefix = f"{task.method}/seed{task.seed}"
        if task.dataset.get("name") != contract.get("dataset", "cifar100lt_if100"):
            setting_errors.append(f"{prefix}: dataset.name")
        if task.dataset.get("data_type") != "cifar100lt":
            setting_errors.append(f"{prefix}: dataset.data_type")
        if int(task.dataset.get("num_class", -1)) != 100:
            setting_errors.append(f"{prefix}: dataset.num_class")
        if int(task.dataset.get("img_size", -1)) != 32:
            setting_errors.append(f"{prefix}: dataset.img_size")
        if not close(task.dataset.get("imbalance_factor"), 0.01):
            setting_errors.append(f"{prefix}: dataset.imbalance_factor")
        if int(task.dataset.get("split_seed", -1)) != 0:
            setting_errors.append(f"{prefix}: dataset.split_seed")
        if int(task.train.get("total_steps", -1)) != updates:
            setting_errors.append(f"{prefix}: train.total_steps")
        if int(task.train.get("batch_size", -1)) != batch:
            setting_errors.append(f"{prefix}: train.batch_size")
        if not close(task.train.get("lr"), lr):
            setting_errors.append(f"{prefix}: train.lr")
        if int(task.train.get("T", -1)) != diffusion_steps:
            setting_errors.append(f"{prefix}: train.T")
        if int(task.eval.get("checkpoint_step", -1)) != updates:
            setting_errors.append(f"{prefix}: eval.checkpoint_step")
        if int(task.eval.get("num_images", -1)) != generated:
            setting_errors.append(f"{prefix}: eval.num_images")
        if not close(task.eval.get("guidance_scale"), guidance):
            setting_errors.append(f"{prefix}: eval.guidance_scale")
        if not task.eval.get("uniform_labels", False):
            setting_errors.append(f"{prefix}: eval.uniform_labels")
        if task.eval.get("sampler_family") != wanted_family:
            setting_errors.append(f"{prefix}: eval.sampler_family")
        if task.eval.get("metric_protocol") != "native_cifar_v1":
            setting_errors.append(f"{prefix}: eval.metric_protocol")
        skip = int(task.eval.get("ddim_skip_step", -1))
        if (
            task.eval.get("sample_method") != "ddim"
            or skip <= 0
            or diffusion_steps // skip != wanted_steps
        ):
            setting_errors.append(f"{prefix}: CCUA official DDIM stride")
    checks.append(Check(
        "ERROR" if setting_errors else "PASS",
        "native-cifar-controls",
        "; ".join(setting_errors)
        if setting_errors
        else (
            f"{updates} updates, batch={batch}, T={diffusion_steps}, "
            f"DDIM-{wanted_steps}, N={generated}"
        ),
    ))
    return checks


def run_preflight(campaign: LoadedCampaign) -> List[Check]:
    checks: List[Check] = []
    gpus = query_gpus()
    checks.append(Check(
        "PASS" if gpus else "ERROR",
        "gpu",
        ", ".join(
            f"{gpu.index}:{gpu.name} {gpu.memory_free_mb / 1024:.1f}GB free"
            for gpu in gpus
        )
        if gpus
        else "nvidia-smi returned no GPUs",
    ))

    runs_root = Path(campaign.server["runtime"]["runs_root"])
    disk_target = runs_root if runs_root.exists() else campaign.root
    free_gb = shutil.disk_usage(disk_target).free / 1024**3
    guard = float(campaign.server.get("machine", {}).get("disk_stop_free_gb", 0))
    checks.append(Check(
        "PASS" if free_gb >= guard else "ERROR",
        "disk",
        f"{free_gb:.1f} GB free; guard={guard:.1f} GB",
    ))

    runtime = campaign.server["runtime"]
    wandb_mode = runtime.get("wandb_mode", os.getenv("WANDB_MODE", "online"))
    wandb_ready = wandb_mode != "online" or bool(os.getenv("WANDB_API_KEY"))
    checks.append(Check(
        "PASS" if wandb_ready else "ERROR",
        "wandb",
        f"mode={wandb_mode}, project={runtime.get('wandb_project', 'longtail')}"
        if wandb_ready
        else "online mode but WANDB_API_KEY is empty",
    ))

    repo_root = Path(runtime["repos_root"])
    _, manifest_check = _load_vendor_manifest(repo_root)
    checks.append(manifest_check)
    checks.extend(_check_native_cifar_contract(campaign, repo_root))
    checks.extend(_check_metric_assets(campaign, repo_root))
    return checks
