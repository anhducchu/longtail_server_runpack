"""Run the paper experiments directly.

This is the complete runtime flow:

    load YAML -> choose method/seed -> train -> sample -> evaluate

``experiment.py`` contains the corresponding functions and translates the
resolved settings into direct calls to ``third_party/CCUA-DDPM/main.py``.
There is no scheduler, worker process, state database, or adapter layer.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from experiment import (
    Experiment,
    Run,
    atomic_write_json,
    evaluate,
    expected_samples,
    load_experiment,
    preflight,
    sample,
    select_runs,
    train,
)


DEFAULT_CONFIG = "configs/native_cifar100_if100.yaml"


def main() -> None:
    args = parse_args()
    experiment = load_experiment(args.config)

    if args.command == "plan":
        exit_code = show_plan(experiment)
    elif args.command == "preflight":
        exit_code = show_preflight(experiment, gpu=args.gpu)
    else:
        exit_code = run_selected(experiment, args)
    raise SystemExit(exit_code)


def run_selected(experiment: Experiment, args: argparse.Namespace) -> int:
    """Run selected tasks sequentially; shell/Slurm owns parallelism."""
    try:
        runs = select_runs(
            experiment,
            method=args.method,
            seed=args.seed,
            index=args.index,
        )
        if args.resume_checkpoint and len(runs) != 1:
            raise ValueError(
                "--resume-checkpoint requires exactly one run; select --method and --seed"
            )
        if args.resume_step is not None and not args.resume_checkpoint:
            raise ValueError("--resume-step requires --resume-checkpoint")
        if args.resume_checkpoint and args.stage not in {"all", "train"}:
            raise ValueError("--resume-checkpoint is valid only for --stage all or train")
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    if not args.skip_preflight:
        errors = [
            check for check in preflight(experiment, gpu=args.gpu)
            if check.level == "ERROR"
        ]
        if errors:
            for check in errors:
                print(f"[error] {check.name}: {check.message}", file=sys.stderr)
            return 2

    try:
        for run in runs:
            run_one(
                run,
                stage=args.stage,
                gpu=args.gpu,
                resume_checkpoint=args.resume_checkpoint,
                resume_step=args.resume_step,
            )
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


def run_one(
    run: Run,
    *,
    stage: str = "all",
    gpu: int | None = None,
    resume_checkpoint: str | None = None,
    resume_step: int | None = None,
) -> None:
    """The entire lifecycle of one method/seed run, in execution order."""
    print(f"\n[experiment] method={run.method} seed={run.seed} output={run.run_dir}")
    run.run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run.run_dir / "config.resolved.json", run.resolved_config())

    if stage in {"all", "train"}:
        train(
            run,
            gpu=gpu,
            resume_checkpoint=resume_checkpoint,
            resume_step=resume_step,
        )

    samples = expected_samples(run)
    if stage in {"all", "sample"}:
        samples = sample(run, gpu=gpu)

    if stage in {"all", "evaluate"}:
        evaluate(run, samples, gpu=gpu)

    if stage == "all":
        (run.run_dir / "SUCCESS").write_text(
            time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n",
            encoding="utf-8",
        )


def show_plan(experiment: Experiment) -> int:
    print(f"Experiment: {experiment.name} ({len(experiment.runs)} runs)")
    for index, run in enumerate(experiment.runs):
        print(
            f"[{index:02d}] method={run.method:5} seed={run.seed} "
            f"output={run.run_dir}"
        )
    return 0


def show_preflight(experiment: Experiment, *, gpu: int | None = None) -> int:
    checks = preflight(experiment, gpu=gpu)
    for check in checks:
        print(f"[{check.level:5}] {check.name:16} {check.message}")
    errors = sum(check.level == "ERROR" for check in checks)
    print(f"\nPreflight: {len(checks) - errors} PASS, {errors} ERROR")
    return 1 if errors else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Direct baseline flow: train -> sample -> evaluate.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="show method/seed runs without executing")
    _add_config(plan)

    check = commands.add_parser(
        "preflight", help="check config, vendor source, assets, disk, and GPU"
    )
    _add_config(check)
    check.add_argument("--gpu", type=int, help="require this physical GPU")

    run = commands.add_parser("run", help="run train, sample, and evaluation")
    _add_config(run)
    run.add_argument("--method", choices=["ddpm", "cbdm", "t2h", "ccua"])
    run.add_argument("--seed", type=int)
    run.add_argument("--index", type=int, help="one index printed by the plan command")
    run.add_argument("--gpu", type=int, help="physical GPU index for this process")
    run.add_argument(
        "--stage",
        choices=["all", "train", "sample", "evaluate"],
        default="all",
        help="run the full flow or only one stage",
    )
    run.add_argument("--resume-checkpoint", help="explicit full-state ckpt_<step>.pt")
    run.add_argument("--resume-step", type=int, help="step if absent from checkpoint name")
    run.add_argument(
        "--skip-preflight",
        action="store_true",
        help="use only for the smoke config or an intentional debug run",
    )
    return parser.parse_args()


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"experiment YAML (default: {DEFAULT_CONFIG})",
    )


if __name__ == "__main__":
    main()
