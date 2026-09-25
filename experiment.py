"""Configuration and direct train/sample/evaluate operations for one run.

Read the public functions in this order: ``load_experiment``, ``train``,
``sample``, and ``evaluate``. Everything prefixed with ``_`` only supports
those four operations.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_OBJECTIVES = {"ddpm", "cbdm", "t2h", "ccua"}
CCUA_COMMIT = "baff9def6cfd553d4452d10eab0bac66ec2727aa"
CHECKPOINT_NAME = re.compile(r"^ckpt_(\d+)\.pt$")
ENV_VALUE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


@dataclass(frozen=True)
class Run:
    root: Path
    experiment: str
    protocol: str
    method: str
    seed: int
    method_config: dict[str, Any]
    repository: dict[str, Any]
    dataset: dict[str, Any]
    train: dict[str, Any]
    evaluation: dict[str, Any]
    repo_dir: Path
    run_dir: Path
    metrics_root: Path

    def resolved_config(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Experiment:
    root: Path
    config_path: Path
    raw: dict[str, Any]
    name: str
    protocol: str
    output_dir: Path
    runs: list[Run]


@dataclass(frozen=True)
class Samples:
    suffix: str
    images: Path
    labels: Path


@dataclass(frozen=True)
class Check:
    level: str
    name: str
    message: str


# Configuration -------------------------------------------------------------


def load_local_env(root: Path) -> None:
    """Load the repo's ignored ``.env.local`` without overwriting the shell."""
    configured = os.environ.get("LTX_ENV_FILE", "").strip()
    path = Path(configured).expanduser() if configured else root / ".env.local"
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


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return ENV_VALUE.sub(
            lambda match: os.environ.get(match.group(1)) or match.group(2) or "",
            value,
        )
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _absolute(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _env_path(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def load_experiment(config_path: str | Path) -> Experiment:
    """Read one flat YAML file and create one run per method/seed pair."""
    config_path = Path(config_path).expanduser().resolve()
    root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    load_local_env(root)
    raw = _expand_env(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})

    experiment = raw.get("experiment", {})
    name = str(experiment.get("name", config_path.stem))
    protocol = str(experiment.get("protocol", "native_cifar_v1"))
    seeds = [int(seed) for seed in experiment.get("seeds", [0])]
    methods = raw.get("methods", [])
    if not methods:
        raise ValueError(f"{config_path}: methods must not be empty")
    if not seeds:
        raise ValueError(f"{config_path}: experiment.seeds must not be empty")

    repository = dict(raw.get("repository", {}))
    repo_root = _absolute(root, _env_path("LTX_REPOS_ROOT", "third_party"))
    repo_dir = repo_root / str(repository.get("directory", "CCUA-DDPM"))
    runs_root = _absolute(root, _env_path("LTX_RUNS_ROOT", "runs"))
    output_dir = runs_root / name
    output_group = str(experiment.get("output_group", "")).strip()
    grouped_output = output_dir / output_group if output_group else output_dir
    metrics_root = _absolute(
        root,
        _env_path("LTX_METRICS_ROOT", str(repo_dir / "stats")),
    )

    dataset = dict(raw.get("dataset", {}))
    dataset["root"] = str(
        _absolute(root, dataset.get("root") or _env_path("LTX_DATA_ROOT", "data"))
    )
    train = dict(raw.get("train", {}))
    evaluation = dict(raw.get("evaluation", {}))

    runs = []
    for method_config in methods:
        method_config = dict(method_config)
        method = str(method_config["name"]).lower()
        for seed in seeds:
            runs.append(Run(
                root=root,
                experiment=name,
                protocol=protocol,
                method=method,
                seed=seed,
                method_config=method_config,
                repository=repository,
                dataset=dataset,
                train=train,
                evaluation=evaluation,
                repo_dir=repo_dir,
                run_dir=grouped_output / method / f"seed_{seed}",
                metrics_root=metrics_root,
            ))
    return Experiment(root, config_path, raw, name, protocol, output_dir, runs)


def select_runs(
    experiment: Experiment,
    *,
    method: str | None = None,
    seed: int | None = None,
    index: int | None = None,
) -> list[Run]:
    """Select a method/seed or one stable Slurm-array index."""
    if index is not None:
        if method is not None or seed is not None:
            raise ValueError("--index cannot be combined with --method or --seed")
        if index < 0 or index >= len(experiment.runs):
            raise ValueError(f"--index must be between 0 and {len(experiment.runs) - 1}")
        return [experiment.runs[index]]

    selected = [
        run
        for run in experiment.runs
        if (method is None or run.method == method)
        and (seed is None or run.seed == seed)
    ]
    if not selected:
        raise ValueError(f"no run matches method={method!r}, seed={seed!r}")
    return selected


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


# Vendor command construction ----------------------------------------------


def _objective_flags(run: Run) -> list[str]:
    objective = str(run.method_config.get("objective", run.method)).lower()
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(
            "supported objectives are ddpm, cbdm, t2h, and ccua; "
            f"got {objective!r}"
        )
    if objective != run.method:
        raise ValueError(f"method {run.method!r} cannot use objective {objective!r}")

    disabled = ["--nocbdm", "--ccua_al=0.0", "--ccua_ucl=0.0"]
    if objective == "ddpm":
        return ["--notransfer_x0", *disabled]
    if objective == "t2h":
        return ["--transfer_x0", *disabled]
    if objective == "cbdm":
        tau = float(run.method_config.get("cb_tau", 1.0))
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError(f"CBDM requires cb_tau > 0, got {tau}")
        return [
            "--notransfer_x0",
            "--cbdm",
            f"--cb_tau={tau}",
            "--ccua_al=0.0",
            "--ccua_ucl=0.0",
        ]
    alignment = float(run.method_config.get("ccua_al", 1.0))
    contrastive = float(run.method_config.get("ccua_ucl", 1.0))
    if not all(math.isfinite(value) and value > 0 for value in (alignment, contrastive)):
        raise ValueError("CCUA requires positive finite ccua_al and ccua_ucl")
    return [
        "--notransfer_x0",
        "--nocbdm",
        f"--ccua_al={alignment}",
        f"--ccua_ucl={contrastive}",
    ]


def _backbone_flags(train: dict[str, Any]) -> list[str]:
    flags = []
    if "ch" in train:
        flags.append(f"--ch={train['ch']}")
    flags.extend(f"--ch_mult={value}" for value in train.get("ch_mult", []))
    flags.extend(f"--attn={value}" for value in train.get("attn", []))
    for key in ("num_res_blocks", "ema_decay"):
        if key in train:
            flags.append(f"--{key}={train[key]}")
    return flags


def _checkpoint_step(path: Path, explicit_step: int | None = None) -> int:
    if explicit_step is not None:
        if explicit_step < 0:
            raise ValueError("--resume-step must be non-negative")
        return explicit_step
    match = CHECKPOINT_NAME.fullmatch(path.name)
    if not match:
        raise ValueError(
            f"cannot infer a step from {path.name!r}; pass --resume-step"
        )
    return int(match.group(1))


def _latest_checkpoint(run_dir: Path, target: int) -> Path | None:
    checkpoints = []
    for path in run_dir.glob("ckpt_*.pt"):
        match = CHECKPOINT_NAME.fullmatch(path.name)
        if match and int(match.group(1)) < target:
            checkpoints.append((int(match.group(1)), path))
    return max(checkpoints, default=(None, None))[1]


def train_command(
    run: Run,
    *,
    resume_checkpoint: str | None = None,
    resume_step: int | None = None,
) -> list[str]:
    """Build the direct command for ``third_party/CCUA-DDPM/main.py``."""
    train = run.train
    dataset = run.dataset
    target = int(run.evaluation.get("checkpoint_step", train.get("total_steps", 300_000)))

    checkpoint = Path(resume_checkpoint).expanduser().resolve() if resume_checkpoint else None
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint}")
    checkpoint = checkpoint or _latest_checkpoint(run.run_dir, target)
    completed_step = _checkpoint_step(checkpoint, resume_step) if checkpoint else None
    if completed_step is not None and completed_step >= target:
        raise ValueError(f"resume step {completed_step} must be below target {target}")
    remaining_steps = (target - completed_step if completed_step is not None else target) + 1

    command = [
        sys.executable,
        "main.py",
        "--train",
        f"--data_type={dataset.get('data_type', 'cifar100lt')}",
        f"--data_path={dataset['root']}",
        f"--imb_factor={dataset.get('imbalance_factor', 0.01)}",
        f"--num_class={dataset.get('num_class', 100)}",
        f"--img_size={dataset.get('img_size', 32)}",
        f"--logdir={run.run_dir}",
        f"--seed={run.seed}",
        f"--batch_size={train.get('batch_size', 64)}",
        f"--lr={train.get('lr', 2e-4)}",
        f"--T={train.get('T', 1000)}",
        f"--beta_1={train.get('beta_1', 1e-4)}",
        f"--beta_T={train.get('beta_T', 0.02)}",
        f"--var_type={train.get('var_type', 'fixedlarge')}",
        f"--dropout={train.get('dropout', 0.1)}",
        f"--grad_clip={train.get('grad_clip', 1.0)}",
        f"--warmup={train.get('warmup', 5000)}",
        f"--save_step={train.get('save_step', 25000)}",
        f"--sample_step={train.get('sample_step', 100000)}",
        f"--num_workers={train.get('num_workers', 4)}",
        f"--total_steps={remaining_steps}",
        *_backbone_flags(train),
        *_objective_flags(run),
    ]
    if checkpoint is not None:
        command.extend([
            "--resume",
            f"--resume_dir={checkpoint.parent}",
            f"--ckpt_step={completed_step}",
        ])
    return command


def _run_command(command: list[str], *, cwd: Path, gpu: int | None) -> None:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[run] cwd={cwd}")
    print(f"[run] command={shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


# Direct experiment stages --------------------------------------------------


def train(
    run: Run,
    *,
    gpu: int | None = None,
    resume_checkpoint: str | None = None,
    resume_step: int | None = None,
) -> Path:
    """Train one method/seed, resuming the newest complete checkpoint."""
    target = int(run.evaluation.get("checkpoint_step", run.train.get("total_steps", 300_000)))
    final_checkpoint = run.run_dir / f"ckpt_{target}.pt"
    if final_checkpoint.is_file():
        print(f"[train] keeping existing {final_checkpoint}")
        return final_checkpoint
    run.run_dir.mkdir(parents=True, exist_ok=True)
    _run_command(
        train_command(
            run,
            resume_checkpoint=resume_checkpoint,
            resume_step=resume_step,
        ),
        cwd=run.repo_dir,
        gpu=gpu,
    )
    if not final_checkpoint.is_file():
        raise RuntimeError(f"training finished without {final_checkpoint}")
    return final_checkpoint


def expected_samples(run: Run) -> list[Samples]:
    scales = run.method_config.get(
        "guidance_scales",
        [run.method_config.get("guidance_scale", run.evaluation.get("guidance_scale", 1.0))],
    )
    if not isinstance(scales, list):
        scales = [scales]
    single_scale = len(scales) == 1
    return [
        Samples(
            suffix="" if single_scale else f"_w{scale}",
            images=run.run_dir / f"ccua_samples{'' if single_scale else f'_w{scale}'}.npy",
            labels=run.run_dir / f"ccua_samples{'' if single_scale else f'_w{scale}'}.npy.labels.npy",
        )
        for scale in scales
    ]


def sample(run: Run, *, gpu: int | None = None) -> list[Samples]:
    """Generate samples from the final EMA checkpoint."""
    train_cfg = run.train
    evaluate = run.evaluation
    target = int(evaluate.get("checkpoint_step", train_cfg.get("total_steps", 300_000)))
    outputs = expected_samples(run)
    scales = run.method_config.get(
        "guidance_scales",
        [run.method_config.get("guidance_scale", evaluate.get("guidance_scale", 1.0))],
    )
    if not isinstance(scales, list):
        scales = [scales]

    for scale, output in zip(scales, outputs):
        if output.images.is_file() and output.labels.is_file():
            print(f"[sample] keeping existing {output.images}")
            continue
        command = [
            sys.executable,
            "main.py",
            "--sample",
            f"--logdir={run.run_dir}",
            f"--seed={run.seed}",
            f"--ckpt_step={target}",
            f"--num_class={run.dataset.get('num_class', 100)}",
            f"--img_size={run.dataset.get('img_size', 32)}",
            f"--num_images={evaluate.get('num_images', 50000)}",
            f"--batch_size={train_cfg.get('batch_size', 64)}",
            f"--sample_batch_size={evaluate.get('sample_batch_size', train_cfg.get('batch_size', 64))}",
            f"--sample_method={evaluate.get('sample_method', 'ddpm')}",
            f"--ddim_skip_step={evaluate.get('ddim_skip_step', 1)}",
            f"--omega={scale}",
            f"--T={train_cfg.get('T', 1000)}",
            f"--beta_1={train_cfg.get('beta_1', 1e-4)}",
            f"--beta_T={train_cfg.get('beta_T', 0.02)}",
            f"--var_type={train_cfg.get('var_type', 'fixedlarge')}",
            f"--dropout={train_cfg.get('dropout', 0.1)}",
            f"--sample_output={output.images}",
            *_backbone_flags(train_cfg),
        ]
        if evaluate.get("uniform_labels", False):
            command.append("--uniform_labels")
        _run_command(command, cwd=run.repo_dir, gpu=gpu)
    return outputs


def evaluate(run: Run, samples: list[Samples], *, gpu: int | None = None) -> None:
    """Run the shared evaluator for every generated guidance scale."""
    if not run.evaluation.get("paper_metrics", False):
        return
    single_scale = len(samples) == 1
    for sample_set in samples:
        base = [
            sys.executable,
            "-m",
            "tools.evaluate_ccua",
            "--repo",
            str(run.repo_dir),
            "--samples",
            str(sample_set.images),
            "--labels",
            str(sample_set.labels),
            "--metrics-root",
            str(run.metrics_root),
            "--inception-batch-size",
            str(run.evaluation.get("inception_batch_size", 16)),
            "--generated-features-cache",
            str(run.run_dir / f"inception_features{sample_set.suffix}.npz"),
        ]

        if run.evaluation.get("fast_metrics", False):
            preview = run.run_dir / f"fast_metrics_preview{sample_set.suffix}.json"
            preview_outputs = [preview]
            if single_scale:
                preview_outputs.append(run.run_dir / "fast_per_class_fids.json")
            if not all(path.is_file() for path in preview_outputs):
                command = [*base, "--mode", "gpu_fast", "--output", str(preview)]
                if single_scale:
                    command.extend([
                        "--per-class-output",
                        str(run.run_dir / "fast_per_class_fids.json"),
                    ])
                _run_command(command, cwd=run.root, gpu=gpu)

        metrics = run.run_dir / f"metrics.native{sample_set.suffix}.json"
        metric_outputs = [metrics]
        if single_scale:
            metric_outputs.append(run.run_dir / "metrics.per_class.native.json")
        if all(path.is_file() for path in metric_outputs):
            print(f"[evaluate] keeping existing {metrics}")
            continue
        command = [*base, "--mode", "detailed", "--output", str(metrics)]
        if single_scale:
            command.extend([
                "--per-class-output",
                str(run.run_dir / "metrics.per_class.native.json"),
            ])
        _run_command(command, cwd=run.root, gpu=gpu)


# Preflight -----------------------------------------------------------------


def _gpu_check(selected_gpu: int | None = None) -> Check:
    required_gb = float(os.environ.get("LTX_MIN_FREE_GPU_GB", "12"))
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        return_code, output = result.returncode, result.stdout
    except FileNotFoundError:
        return_code, output = 1, ""
    gpus = []
    if return_code == 0:
        for row in csv.reader(io.StringIO(output)):
            try:
                gpus.append((int(row[0].strip()), row[1].strip(), float(row[2])))
            except (IndexError, ValueError):
                continue
    candidates = [gpu for gpu in gpus if selected_gpu is None or gpu[0] == selected_gpu]
    eligible = [gpu for gpu in candidates if gpu[2] / 1024 >= required_gb]
    available = ", ".join(
        f"{index}:{name} {memory / 1024:.1f}GB free"
        for index, name, memory in gpus
    ) or "nvidia-smi returned no GPUs"
    return Check(
        "PASS" if eligible else "ERROR",
        "gpu",
        f"{available}; selected={selected_gpu if selected_gpu is not None else 'any'}; "
        f"required={required_gb:.1f}GB",
    )


def _protocol_check(experiment: Experiment) -> Check:
    if experiment.protocol == "smoke":
        return Check("PASS", "protocol", "smoke configuration")
    if experiment.protocol != "native_cifar_v1":
        return Check("ERROR", "protocol", f"unsupported protocol {experiment.protocol!r}")

    run = experiment.runs[0]
    expected = {
        "dataset.name": (run.dataset.get("name"), "cifar100lt_if100"),
        "dataset.imbalance_factor": (float(run.dataset.get("imbalance_factor", -1)), 0.01),
        "dataset.split_seed": (int(run.dataset.get("split_seed", -1)), 0),
        "dataset.num_class": (int(run.dataset.get("num_class", -1)), 100),
        "dataset.img_size": (int(run.dataset.get("img_size", -1)), 32),
        "train.total_steps": (int(run.train.get("total_steps", -1)), 300_000),
        "train.batch_size": (int(run.train.get("batch_size", -1)), 64),
        "train.lr": (float(run.train.get("lr", -1)), 2e-4),
        "train.T": (int(run.train.get("T", -1)), 1000),
        "train.save_step": (int(run.train.get("save_step", -1)), 25_000),
        "evaluation.checkpoint_step": (
            int(run.evaluation.get("checkpoint_step", -1)),
            300_000,
        ),
        "evaluation.num_images": (int(run.evaluation.get("num_images", -1)), 50_000),
        "evaluation.guidance_scale": (
            float(run.evaluation.get("guidance_scale", -1)),
            1.5,
        ),
        "evaluation.sample_method": (run.evaluation.get("sample_method"), "ddim"),
        "evaluation.ddim_skip_step": (
            int(run.evaluation.get("ddim_skip_step", -1)),
            10,
        ),
        "evaluation.uniform_labels": (
            bool(run.evaluation.get("uniform_labels", False)),
            True,
        ),
    }
    errors = [name for name, (actual, wanted) in expected.items() if actual != wanted]
    return Check(
        "ERROR" if errors else "PASS",
        "protocol",
        f"drift: {', '.join(errors)}" if errors else "native CIFAR-100-LT contract",
    )


def preflight(experiment: Experiment, *, gpu: int | None = None) -> list[Check]:
    """Validate only facts required before a direct experiment run."""
    first = experiment.runs[0]
    checks = []

    objective_errors = []
    for run in experiment.runs:
        try:
            _objective_flags(run)
        except ValueError as exc:
            objective_errors.append(f"{run.method}/seed{run.seed}: {exc}")
    checks.append(Check(
        "ERROR" if objective_errors else "PASS",
        "objectives",
        "; ".join(objective_errors) or f"{len(experiment.runs)} method/seed runs",
    ))
    checks.append(_protocol_check(experiment))

    manifest_path = first.repo_dir.parent / "THIRD_PARTY_MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        component = manifest["components"]["ccua"]
        if component.get("commit") != CCUA_COMMIT:
            raise ValueError(f"expected commit {CCUA_COMMIT}")
        if first.repository.get("commit") != CCUA_COMMIT:
            raise ValueError("config repository commit does not match vendor manifest")
        for relative in (
            "main.py",
            "diffusion.py",
            "model/model.py",
            ".ltx_ccua_sample_export_v1",
        ):
            if not (first.repo_dir / relative).is_file():
                raise FileNotFoundError(first.repo_dir / relative)
    except Exception as exc:
        checks.append(Check("ERROR", "vendor", f"{manifest_path}: {exc}"))
    else:
        checks.append(Check("PASS", "vendor", f"CCUA-DDPM {CCUA_COMMIT[:12]}"))

    if any(run.evaluation.get("paper_metrics", False) for run in experiment.runs):
        metric_manifest = first.metrics_root / "cifar100.metric_manifest.json"
        required = (
            "cifar100.train.npz",
            "cifar100_feats.npy",
            "cifar100_labels.npy",
            "cifar100_vgg16_fc2.npy",
            "cifar100_vgg16_fc2_k3_radii.npy",
        )
        missing = [name for name in required if not (first.metrics_root / name).is_file()]
        try:
            metric_info = json.loads(metric_manifest.read_text(encoding="utf-8"))
            valid_manifest = (
                metric_info.get("dataset") == "cifar100"
                and metric_info.get("repository_commit") == CCUA_COMMIT
                and int(metric_info.get("num_images", -1)) == 50_000
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            valid_manifest = False
        level = "PASS" if valid_manifest and not missing else "ERROR"
        message = "balanced CIFAR metric assets present" if level == "PASS" else (
            f"missing manifest/assets: {metric_manifest}, {missing}"
        )
        checks.append(Check(level, "metric-assets", message))

    disk_guard = float(os.environ.get("LTX_MIN_FREE_DISK_GB", "20"))
    disk_target = experiment.output_dir.parent
    if not disk_target.exists():
        disk_target = first.root
    free_gb = shutil.disk_usage(disk_target).free / 1024**3
    checks.append(Check(
        "PASS" if free_gb >= disk_guard else "ERROR",
        "disk",
        f"{free_gb:.1f}GB free; required={disk_guard:.1f}GB",
    ))
    checks.append(_gpu_check(gpu))
    return checks
