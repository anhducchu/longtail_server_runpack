"""Translate one campaign task into train, sample, and metric commands."""

from __future__ import annotations

import math
import os
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Task


SUPPORTED_OBJECTIVES = {"ddpm", "cbdm", "t2h", "ccua"}
CHECKPOINT_NAME = re.compile(r"^ckpt_(\d+)\.pt$")


def resolve_num_workers(train: dict, default: int) -> int:
    """Use the scheduler's worker limit when it is present."""
    override = os.environ.get("LTX_NUM_WORKERS")
    if override:
        try:
            return int(override)
        except ValueError:
            pass
    return int(train.get("num_workers", default))


def resolve_inception_batch_size(evaluate: dict, default: int = 16) -> int:
    value = int(evaluate.get("inception_batch_size", default))
    if value <= 0:
        raise ValueError(f"inception_batch_size must be positive, got {value}")
    return value


@dataclass
class Phase:
    name: str
    command: list[str]
    cwd: Path
    env: dict[str, str] = field(default_factory=dict)
    skip_if_exists: list[Path] = field(default_factory=list)


def get_resume_checkpoint(
    train: Mapping[str, Any],
    method_config: Mapping[str, Any] | None = None,
) -> Path | None:
    """Return the configured checkpoint after confirming that it exists."""
    settings = dict(train)
    if method_config:
        settings.update(
            {key: method_config[key] for key in ("resume_checkpoint", "resume_step") if key in method_config}
        )

    raw_path = str(settings.get("resume_checkpoint", "") or "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"explicit resume checkpoint does not exist: {path}")
    return path


def get_resume_step(
    train: Mapping[str, Any],
    method_config: Mapping[str, Any] | None,
    checkpoint: Path,
) -> int:
    """Read the completed update from config or a ``ckpt_<step>.pt`` name."""
    configured = None
    if method_config and "resume_step" in method_config:
        configured = method_config["resume_step"]
    elif "resume_step" in train:
        configured = train["resume_step"]

    if configured not in (None, ""):
        try:
            step = int(configured)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"resume_step must be an integer, got {configured!r} for {checkpoint}"
            ) from exc
    else:
        match = CHECKPOINT_NAME.fullmatch(checkpoint.name)
        if not match:
            raise ValueError(
                f"cannot infer resume step from {checkpoint.name!r}; set resume_step explicitly"
            )
        step = int(match.group(1))

    if step < 0:
        raise ValueError(f"resume_step must be non-negative, got {step}")
    return step


class CCUAAdapter:
    """Build the three phases used by every baseline objective.

    All objectives share the vendored CCUA-DDPM model, dataset, sampler,
    checkpoint format, and evaluator. Only their loss flags differ.
    """

    DEFAULT_CHECKPOINT_INTERVAL_STEPS = 25_000

    def __init__(self, root: Path):
        self.root = root

    def repo_dir(self, task: Task) -> Path:
        directory = Path(task.repository["directory"])
        if directory.is_absolute():
            return directory
        return Path(task.runtime["repos_root"]) / directory

    @staticmethod
    def latest(run_dir: Path, target: int) -> int | None:
        """Return the newest checkpoint strictly before the target step."""
        steps = []
        for path in run_dir.glob("ckpt_*.pt"):
            match = CHECKPOINT_NAME.fullmatch(path.name)
            if match and int(match.group(1)) < target:
                steps.append(int(match.group(1)))
        return max(steps) if steps else None

    @staticmethod
    def _backbone_flags(train: dict) -> list[str]:
        flags = []
        if "ch" in train:
            flags.append(f"--ch={train['ch']}")
        flags.extend(f"--ch_mult={value}" for value in train.get("ch_mult", []))
        flags.extend(f"--attn={value}" for value in train.get("attn", []))
        for key in ("num_res_blocks", "ema_decay"):
            if key in train:
                flags.append(f"--{key}={train[key]}")
        return flags

    @staticmethod
    def _objective_flags(task: Task) -> tuple[str, list[str]]:
        config = task.method_config
        objective = str(config.get("objective", task.method)).lower()
        if objective not in SUPPORTED_OBJECTIVES:
            raise ValueError(
                "CCUA U-Net adapter supports only objective=ddpm, cbdm, t2h, or ccua; "
                f"got {objective!r}"
            )

        shared_off = ["--nocbdm", "--ccua_al=0.0", "--ccua_ucl=0.0"]
        if objective == "ddpm":
            return objective, shared_off
        if objective == "t2h":
            return objective, ["--transfer_x0", *shared_off]
        if objective == "cbdm":
            tau = float(config.get("cb_tau", config.get("tau", 1.0)))
            if not math.isfinite(tau) or tau <= 0:
                raise ValueError(f"CCUA CBDM objective requires cb_tau > 0, got {tau}")
            return objective, [
                "--cbdm",
                f"--cb_tau={tau}",
                "--ccua_al=0.0",
                "--ccua_ucl=0.0",
            ]
        return objective, [
            "--nocbdm",
            f"--ccua_al={config.get('ccua_al', 1.0)}",
            f"--ccua_ucl={config.get('ccua_ucl', 1.0)}",
        ]

    def _save_step(self, task: Task) -> int:
        configured = int(task.train.get("save_step", 50_000))
        retry_interval = int(
            task.retry.get(
                "checkpoint_interval_steps",
                self.DEFAULT_CHECKPOINT_INTERVAL_STEPS,
            )
            or 0
        )
        if configured > 0 and retry_interval > 0:
            return min(configured, retry_interval)
        return configured

    def _resume(self, task: Task, run_dir: Path, target: int) -> tuple[int | None, Path]:
        checkpoint = get_resume_checkpoint(task.train, task.method_config)
        if checkpoint is None:
            return self.latest(run_dir, target), run_dir

        step = get_resume_step(task.train, task.method_config, checkpoint)
        if step >= target:
            raise ValueError(
                f"resume checkpoint step {step} must be below target {target}: {checkpoint}"
            )
        return step, checkpoint.parent

    def _train_phase(
        self,
        task: Task,
        repo: Path,
        run_dir: Path,
        batch_size: int,
        target: int,
    ) -> Phase:
        train = task.train
        dataset = task.dataset
        python = task.runtime.get("python", "python")
        objective, objective_flags = self._objective_flags(task)
        latest, resume_dir = self._resume(task, run_dir, target)
        remaining_steps = (target - latest if latest is not None else target) + 1

        command = [
            python,
            "main.py",
            "--train",
            f"--data_type={dataset.get('data_type', 'cifar100lt')}",
            f"--data_path={dataset.get('root', task.runtime.get('data_root', './data'))}",
            f"--imb_factor={dataset.get('imbalance_factor', 0.01)}",
            f"--num_class={dataset.get('num_class', dataset.get('num_classes', 100))}",
            f"--img_size={dataset.get('img_size', 32)}",
            f"--logdir={run_dir}",
            f"--seed={task.seed}",
            f"--batch_size={batch_size}",
            f"--lr={train.get('lr', 2e-4)}",
            f"--T={train.get('T', 1000)}",
            f"--beta_1={train.get('beta_1', 1e-4)}",
            f"--beta_T={train.get('beta_T', 0.02)}",
            f"--var_type={train.get('var_type', 'fixedlarge')}",
            f"--dropout={train.get('dropout', 0.1)}",
            f"--grad_clip={train.get('grad_clip', 1.0)}",
            f"--warmup={train.get('warmup', 5000)}",
            f"--save_step={self._save_step(task)}",
            f"--sample_step={train.get('sample_step', 100000)}",
            f"--num_workers={resolve_num_workers(train, 4)}",
            *self._backbone_flags(train),
        ]
        if objective != "t2h":
            command.append("--notransfer_x0")
        command.extend(objective_flags)
        command.append(f"--total_steps={remaining_steps}")
        if latest is not None:
            command.extend(["--resume", f"--resume_dir={resume_dir}", f"--ckpt_step={latest}"])

        return Phase(
            "train",
            command,
            repo,
            skip_if_exists=[run_dir / f"ckpt_{target}.pt"],
        )

    def _sample_phase(
        self,
        task: Task,
        repo: Path,
        run_dir: Path,
        batch_size: int,
        target: int,
        omega,
        suffix: str,
    ) -> tuple[Phase, Path, Path]:
        train = task.train
        evaluate = task.eval
        dataset = task.dataset
        samples = run_dir / f"ccua_samples{suffix}.npy"
        labels = Path(f"{samples}.labels.npy")
        marker = run_dir / f"CCUA_EVAL_DONE{suffix}"

        command = [
            task.runtime.get("python", "python"),
            "main.py",
            "--sample",
            f"--logdir={run_dir}",
            f"--seed={task.seed}",
            f"--ckpt_step={target}",
            f"--num_class={dataset.get('num_class', dataset.get('num_classes', 100))}",
            f"--img_size={dataset.get('img_size', 32)}",
            f"--num_images={evaluate.get('num_images', 50000)}",
            f"--batch_size={batch_size}",
            f"--sample_batch_size={evaluate.get('sample_batch_size', batch_size)}",
            f"--sample_method={evaluate.get('sample_method', 'ddpm')}",
            f"--ddim_skip_step={evaluate.get('ddim_skip_step', 1)}",
            f"--omega={omega}",
            f"--T={train.get('T', 1000)}",
            f"--beta_1={train.get('beta_1', 1e-4)}",
            f"--beta_T={train.get('beta_T', 0.02)}",
            f"--var_type={train.get('var_type', 'fixedlarge')}",
            f"--dropout={train.get('dropout', 0.1)}",
            f"--sample_output={samples}",
            *self._backbone_flags(train),
        ]
        if evaluate.get("uniform_labels", False):
            command.append("--uniform_labels")

        shell_command = " ".join(shlex.quote(str(value)) for value in command)
        phase = Phase(
            f"eval{suffix}",
            ["bash", "-lc", f"{shell_command} && touch {shlex.quote(str(marker))}"],
            repo,
            skip_if_exists=[marker],
        )
        return phase, samples, labels

    def _metrics_root(self, task: Task, repo: Path) -> Path:
        path = Path(os.environ.get("LTX_METRICS_ROOT", str(repo / "stats"))).expanduser()
        if path.is_absolute():
            return path
        return (Path(task.runtime["repos_root"]).parent / path).resolve()

    def _metric_phases(
        self,
        task: Task,
        repo: Path,
        run_dir: Path,
        samples: Path,
        labels: Path,
        suffix: str,
        single_scale: bool,
    ) -> list[Phase]:
        evaluate = task.eval
        python = task.runtime.get("python", "python")
        base = [
            python,
            "-m",
            "tools.evaluate_ccua",
            "--repo",
            str(repo),
            "--samples",
            str(samples),
            "--labels",
            str(labels),
            "--metrics-root",
            str(self._metrics_root(task, repo)),
            "--inception-batch-size",
            str(resolve_inception_batch_size(evaluate)),
            "--generated-features-cache",
            str(run_dir / f"inception_features{suffix}.npz"),
        ]
        phases = []

        if evaluate.get("fast_metrics", False):
            output = run_dir / f"fast_metrics_preview{suffix}.json"
            command = [*base, "--mode", "gpu_fast", "--output", str(output)]
            outputs = [output]
            if single_scale:
                per_class = run_dir / "fast_per_class_fids.json"
                command.extend(["--per-class-output", str(per_class)])
                outputs.append(per_class)
            phases.append(
                Phase(f"fast_metrics{suffix}", command, self.root, skip_if_exists=outputs)
            )

        output = run_dir / f"metrics.native{suffix}.json"
        command = [*base, "--mode", "detailed", "--output", str(output)]
        outputs = [output]
        if single_scale:
            per_class = run_dir / "metrics.per_class.native.json"
            command.extend(["--per-class-output", str(per_class)])
            outputs.append(per_class)
        phases.append(
            Phase(f"paper_metrics{suffix}", command, self.root, skip_if_exists=outputs)
        )
        return phases

    @staticmethod
    def _guidance_scales(task: Task) -> list:
        evaluate = task.eval
        value = task.method_config.get(
            "guidance_scales",
            [task.method_config.get("guidance_scale", evaluate.get("guidance_scale", 1.0))],
        )
        return value if isinstance(value, list) else [value]

    def phases(self, task: Task, batch_size: int | None = None) -> list[Phase]:
        """Build the task in execution order: train, sample, preview, report."""
        repo = self.repo_dir(task)
        run_dir = Path(task.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        target = int(task.eval.get("checkpoint_step", task.train.get("total_steps", 300_000)))
        batch = int(batch_size or task.train.get("batch_size", 64))

        phases = [self._train_phase(task, repo, run_dir, batch, target)]
        scales = self._guidance_scales(task)
        single_scale = len(scales) == 1
        for omega in scales:
            suffix = "" if single_scale else f"_w{omega}"
            sample_phase, samples, labels = self._sample_phase(
                task, repo, run_dir, batch, target, omega, suffix
            )
            phases.append(sample_phase)
            if task.eval.get("paper_metrics", False):
                phases.extend(
                    self._metric_phases(
                        task,
                        repo,
                        run_dir,
                        samples,
                        labels,
                        suffix,
                        single_scale,
                    )
                )
        return phases
