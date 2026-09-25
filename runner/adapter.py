from __future__ import annotations

import math
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .config import Task


def resolve_num_workers(train: Dict, default: int) -> int:
    """Reduce dataloader workers when several tasks share one machine."""
    override = os.environ.get("LTX_NUM_WORKERS")
    if override:
        try:
            return int(override)
        except ValueError:
            pass
    return int(train.get("num_workers", default))


def resolve_inception_batch_size(evaluate: Dict, default: int = 16) -> int:
    """Validate the evaluator's GPU micro-batch."""
    value = int(evaluate.get("inception_batch_size", default))
    if value <= 0:
        raise ValueError(f"inception_batch_size must be positive, got {value}")
    return value


@dataclass
class Phase:
    name: str
    command: List[str]
    cwd: Path
    env: Dict[str, str] = field(default_factory=dict)
    skip_if_exists: List[Path] = field(default_factory=list)


_CHECKPOINT_NAME = re.compile(r"^ckpt_(\d+)\.pt$")


def get_resume_checkpoint(
    train: Mapping[str, Any], method_config: Mapping[str, Any] | None = None
) -> Path | None:
    """Resolve an explicitly configured full-state checkpoint."""
    values: dict[str, Any] = dict(train)
    if method_config:
        for key in ("resume_checkpoint", "resume_step"):
            if key in method_config:
                values[key] = method_config[key]
    raw_path = str(values.get("resume_checkpoint", "") or "").strip()
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
    values: dict[str, Any] = dict(train)
    if method_config and "resume_step" in method_config:
        values["resume_step"] = method_config["resume_step"]
    raw_step = str(values.get("resume_step", "") or "").strip()
    if raw_step:
        try:
            step = int(raw_step)
        except ValueError as exc:
            raise ValueError(
                f"resume_step must be an integer, got {raw_step!r} for {checkpoint}"
            ) from exc
    else:
        match = _CHECKPOINT_NAME.fullmatch(checkpoint.name)
        if not match:
            raise ValueError(
                f"cannot infer resume step from {checkpoint.name!r}; set resume_step explicitly"
            )
        step = int(match.group(1))
    if step < 0:
        raise ValueError(f"resume_step must be non-negative, got {step}")
    return step


class CCUAAdapter:
    """Build train/evaluation commands for the shared CCUA-DDPM host.

    Upstream ships two pipelines; this drives ``CCUA-DDPM``, the U-Net one, whose
    flag defaults already match the unified backbone (ch=128, [1,2,2,2], attn[1],
    2 blocks, EMA 0.9999, T=1000). Its long-tailed CIFAR split is the same
    exponential ``ImbalanceCIFAR`` construction with ``rand_number=0``, so every
    row of the table trains on an identical subset.

    The same adapter expresses the released DDPM, CBDM, T2H and CCUA baselines
    while retaining one U-Net, dataset, checkpoint, sampler and evaluator.

    Sampling goes through ``main.py --sample`` and metrics go through the
    runpack's shared evaluator, so every objective uses one metric contract.
    """

    DEFAULT_CHECKPOINT_INTERVAL_STEPS = 25_000

    def __init__(self, root: Path):
        self.root = root

    def repo_dir(self, task: Task) -> Path:
        path = Path(task.repository["directory"])
        return path if path.is_absolute() else Path(task.runtime["repos_root"]) / path

    @staticmethod
    def latest(run_dir: Path, target: int) -> int | None:
        values = []
        for p in run_dir.glob("ckpt_*.pt"):
            m = re.fullmatch(r"ckpt_(\d+)\.pt", p.name)
            if m and int(m.group(1)) < target:
                values.append(int(m.group(1)))
        return max(values) if values else None

    def _backbone_flags(self, train: dict) -> List[str]:
        """Pin the architecture explicitly instead of trusting flag defaults to
        keep agreeing with the other repos across vendored-source updates."""
        flags: List[str] = []
        if "ch" in train:
            flags.append(f"--ch={train['ch']}")
        flags += [f"--ch_mult={v}" for v in train.get("ch_mult", [])]
        flags += [f"--attn={v}" for v in train.get("attn", [])]
        for key in ("num_res_blocks", "ema_decay"):
            if key in train:
                flags.append(f"--{key}={train[key]}")
        return flags

    def phases(self, task: Task, batch_size: int | None = None) -> List[Phase]:
        repo = self.repo_dir(task)
        run_dir = Path(task.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        py = task.runtime.get("python", "python")
        train, evaluate = task.train, task.eval
        inception_batch = resolve_inception_batch_size(evaluate)
        target = int(evaluate.get("checkpoint_step", train.get("total_steps", 300000)))
        batch = int(batch_size or train.get("batch_size", 64))
        num_class = task.dataset.get("num_class", task.dataset.get("num_classes", 100))
        img_size = task.dataset.get("img_size", 32)
        data_root = task.dataset.get("root", task.runtime.get("data_root", "./data"))
        metrics_root = Path(os.environ.get(
            "LTX_METRICS_ROOT",
            str(repo / "stats"),
        )).expanduser()
        if not metrics_root.is_absolute():
            metrics_root = (Path(task.runtime["repos_root"]).parent / metrics_root).resolve()

        method_cfg = task.method_config
        # Upstream's only U-Net-pipeline script sets both weights to 1.0; the
        # paper's alpha=gamma=0.05 is tuned for the DiT/SiT ImageNet pipeline,
        # whose latent-space loss scales are not comparable. Overridable per
        # method in the campaign YAML, but forced to zero for sibling
        # objectives so their objective identity cannot drift with defaults.
        objective = str(method_cfg.get("objective", task.method)).lower()
        if objective not in {"ddpm", "cbdm", "t2h", "ccua"}:
            raise ValueError(
                "CCUA U-Net adapter supports only objective=ddpm, cbdm, t2h, or ccua; "
                f"got {objective!r}"
            )
        if objective == "cbdm":
            cb_tau = float(method_cfg.get("cb_tau", method_cfg.get("tau", 1.0)))
            if not math.isfinite(cb_tau) or cb_tau <= 0:
                raise ValueError(f"CCUA CBDM objective requires cb_tau > 0, got {cb_tau}")
            objective_flags = [
                "--cbdm", f"--cb_tau={cb_tau}",
                "--ccua_al=0.0", "--ccua_ucl=0.0",
            ]
        elif objective == "ddpm":
            objective_flags = [
                "--nocbdm", "--ccua_al=0.0", "--ccua_ucl=0.0",
            ]
        elif objective == "t2h":
            # Keep T2H as a separate objective while running it through this
            # campaign's single U-Net/data host.
            objective_flags = [
                "--transfer_x0",
                "--nocbdm", "--ccua_al=0.0", "--ccua_ucl=0.0",
            ]
        else:  # objective == "ccua"
            ccua_al = method_cfg.get("ccua_al", 1.0)
            ccua_ucl = method_cfg.get("ccua_ucl", 1.0)
            objective_flags = [
                "--nocbdm", f"--ccua_al={ccua_al}", f"--ccua_ucl={ccua_ucl}",
            ]

        # Bound work lost to a SIGKILL while preserving a campaign's smaller
        # explicit interval. A configured value of zero still disables saves.
        configured_save_step = int(train.get("save_step", 50000))
        checkpoint_interval = int(
            task.retry.get(
                "checkpoint_interval_steps",
                self.DEFAULT_CHECKPOINT_INTERVAL_STEPS,
            )
            or 0
        )
        if configured_save_step > 0 and checkpoint_interval > 0:
            save_step = min(configured_save_step, checkpoint_interval)
        else:
            save_step = configured_save_step

        train_cmd = [
            py, "main.py", "--train",
            f"--data_type={task.dataset.get('data_type', 'cifar100lt')}",
            f"--data_path={data_root}",
            f"--imb_factor={task.dataset.get('imbalance_factor', 0.01)}",
            f"--num_class={num_class}", f"--img_size={img_size}",
            f"--logdir={run_dir}", f"--seed={task.seed}",
            f"--batch_size={batch}", f"--lr={train.get('lr', 2e-4)}",
            f"--T={train.get('T', 1000)}",
            f"--beta_1={train.get('beta_1', 1e-4)}",
            f"--beta_T={train.get('beta_T', 0.02)}",
            f"--var_type={train.get('var_type', 'fixedlarge')}",
            f"--dropout={train.get('dropout', 0.1)}",
            f"--grad_clip={train.get('grad_clip', 1.0)}",
            f"--warmup={train.get('warmup', 5000)}",
            f"--save_step={save_step}",
            f"--sample_step={train.get('sample_step', 100000)}",
            f"--num_workers={resolve_num_workers(train, 4)}",
        ]
        train_cmd += self._backbone_flags(train)
        # Transfer is disabled for every native objective except the explicit
        # T2H row. Keep objective switches after optional flags so a campaign
        # cannot accidentally re-enable a sibling loss.
        if objective != "t2h":
            train_cmd.append("--notransfer_x0")
        train_cmd += objective_flags

        # Upstream always iterates range(0, total_steps) and names checkpoints
        # step + ckpt_step, so a resume must be given the *remaining* budget or
        # it would train the full budget a second time. Explicit external
        # checkpoints always restore the full CCUA-DDPM training state.
        explicit_resume = get_resume_checkpoint(train, method_cfg)
        if explicit_resume is not None:
            resume_step = get_resume_step(train, method_cfg, explicit_resume)
            if resume_step >= target:
                raise ValueError(
                    f"explicit resume checkpoint step {resume_step} must be below "
                    f"target checkpoint step {target}: {explicit_resume}"
                )
            latest = resume_step
            resume_dir = explicit_resume.parent
        else:
            latest = self.latest(run_dir, target)
            resume_dir = run_dir
        remaining = (target - latest if latest is not None else target) + 1
        train_cmd.append(f"--total_steps={remaining}")
        if latest is not None:
            train_cmd += ["--resume", f"--resume_dir={resume_dir}", f"--ckpt_step={latest}"]
        phases = [Phase("train", train_cmd, repo, skip_if_exists=[run_dir / f"ckpt_{target}.pt"])]

        # One trained checkpoint can be sampled at several guidance strengths:
        # omega only affects sampling. A single scale keeps the plain file names
        # so existing runs still resume; a sweep suffixes each artefact.
        scales = method_cfg.get(
            "guidance_scales",
            [method_cfg.get("guidance_scale", evaluate.get("guidance_scale", 1.0))],
        )
        if not isinstance(scales, list):
            scales = [scales]
        single = len(scales) == 1
        for omega in scales:
            suffix = "" if single else f"_w{omega}"
            samples = run_dir / f"ccua_samples{suffix}.npy"
            labels = Path(str(samples) + ".labels.npy")
            marker = run_dir / f"CCUA_EVAL_DONE{suffix}"
            eval_core = [
                py, "main.py", "--sample",
                f"--logdir={run_dir}", f"--seed={task.seed}",
                f"--ckpt_step={target}", f"--num_class={num_class}",
                f"--img_size={img_size}",
                f"--num_images={evaluate.get('num_images', 50000)}",
                f"--batch_size={batch}",
                f"--sample_batch_size={evaluate.get('sample_batch_size', batch)}",
                f"--sample_method={evaluate.get('sample_method', 'ddpm')}",
                f"--ddim_skip_step={evaluate.get('ddim_skip_step', 1)}",
                f"--omega={omega}", f"--T={train.get('T', 1000)}",
                f"--beta_1={train.get('beta_1', 1e-4)}",
                f"--beta_T={train.get('beta_T', 0.02)}",
                f"--var_type={train.get('var_type', 'fixedlarge')}",
                f"--dropout={train.get('dropout', 0.1)}",
                f"--sample_output={samples}",
            ]
            eval_core += self._backbone_flags(train)
            if evaluate.get("uniform_labels", False):
                eval_core.append("--uniform_labels")
            # The marker is written only after a full, successful sampling pass,
            # so a half-written array is never mistaken for a finished eval.
            quoted = " ".join(shlex.quote(str(x)) for x in eval_core)
            phases.append(Phase(
                f"eval{suffix}", ["bash", "-lc", f"{quoted} && touch {shlex.quote(str(marker))}"],
                repo, skip_if_exists=[marker],
            ))

            if not evaluate.get("paper_metrics", False):
                continue
            metrics_file = "metrics.native.json"
            if not single:
                metrics_file = metrics_file.replace(".json", f"{suffix}.json")
            metric_base = [
                py, "-m", "tools.evaluate_ccua",
                "--repo", str(repo),
                "--samples", str(samples), "--labels", str(labels),
                "--metrics-root", str(metrics_root),
                "--inception-batch-size", str(inception_batch),
                "--generated-features-cache", str(run_dir / f"inception_features{suffix}.npz"),
            ]
            per_class_file = "metrics.per_class.native.json"

            # A non-canonical preview gives fast feedback without being
            # mistaken for the complete paper metric artifact.
            if evaluate.get("fast_metrics", False):
                fast_file = "fast_metrics_preview.json"
                if not single:
                    fast_file = fast_file.replace(".json", f"{suffix}.json")
                fast_output = run_dir / fast_file
                fast_cmd = metric_base + [
                    "--mode", "gpu_fast",
                    "--output", str(fast_output),
                ]
                fast_outputs = [fast_output]
                if single:
                    fast_per_class_file = "fast_per_class_fids.json"
                    fast_per_class_output = run_dir / fast_per_class_file
                    fast_cmd += [
                        "--per-class-output", str(fast_per_class_output),
                    ]
                    fast_outputs.append(fast_per_class_output)
                phases.append(Phase(
                    f"fast_metrics{suffix}",
                    fast_cmd,
                    self.root,
                    skip_if_exists=fast_outputs,
                ))

            metric_cmd = metric_base + [
                "--mode", "detailed",
                "--output", str(run_dir / metrics_file),
            ]
            if single:
                metric_cmd += ["--per-class-output", str(run_dir / per_class_file)]
            outputs = [run_dir / metrics_file]
            if single:
                outputs.append(run_dir / per_class_file)
            phases.append(Phase(f"paper_metrics{suffix}", metric_cmd, self.root, skip_if_exists=outputs))
        return phases
