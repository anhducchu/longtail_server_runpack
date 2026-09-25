"""Public entrypoint for the baseline experiments.

Start reading at ``main()``. A training run follows this path:

    main
      -> run_campaign                         load and validate the campaign
      -> runner.scheduler.Scheduler.run      choose a GPU for each task
      -> runner.worker.main                  execute one task
      -> runner.adapter.CCUAAdapter.phases   build train/sample/eval commands
      -> third_party/CCUA-DDPM/main.py       train the U-Net and sample images
      -> third_party/CCUA-DDPM/diffusion.py  compute the objective

Parameter sources are deliberately separate:

* ``configs/native_*.yaml`` contains scientific parameters and method choices.
* ``configs/server.yaml`` contains machine, storage, retry, and logging policy.
* CLI options below are temporary machine/resume overrides for one launch.
* ``.env.local`` contains local paths and secrets.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import Counter
from pathlib import Path

from runner.config import LoadedCampaign, apply_run_overrides, load_campaign
from runner.preflight import run_preflight
from runner.scheduler import Scheduler
from runner.state import StateDB
from runner.utils import load_runtime_env


DEFAULT_CONFIG = "configs/native_cifar100_if100.yaml"
TERMINAL_STATES = {"completed", "failed", "skipped"}


# Start here. Dispatch is explicit so every CLI command is visible in one place.
def main() -> None:
    load_runtime_env(Path.cwd())
    args = parse_cli()

    if args.command == "run":
        exit_code = run_campaign(
            config_path=args.config,
            skip_preflight=args.skip_preflight,
            gpus=args.gpus,
            tasks_per_gpu=args.per_gpu,
            max_jobs=args.jobs,
            resume_checkpoint=args.resume_checkpoint,
            resume_method=args.resume_method,
            resume_seed=args.resume_seed,
            resume_stage=args.resume_stage,
            resume_step=args.resume_step,
        )
    elif args.command == "plan":
        exit_code = show_plan(args.config)
    elif args.command == "preflight":
        exit_code = check_campaign(args.config)
    elif args.command == "status":
        exit_code = show_status(args.config, watch_seconds=args.watch)
    elif args.command == "retry-failed":
        exit_code = retry_failed(args.config, stage=args.stage)
    elif args.command == "stop":
        exit_code = stop_workers(args.config)
    else:  # argparse rejects unknown commands before this point.
        raise AssertionError(f"unhandled command: {args.command}")

    raise SystemExit(exit_code)


def parse_cli() -> argparse.Namespace:
    """Define the public commands and their temporary overrides."""
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Plan, validate, run, and inspect baseline experiments.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="show resolved tasks without running")
    _add_config_argument(plan)

    preflight = commands.add_parser(
        "preflight", help="validate protocol, files, disk, W&B, and GPU"
    )
    _add_config_argument(preflight)

    run = commands.add_parser("run", help="validate, then run or resume a campaign")
    _add_config_argument(run)
    run.add_argument(
        "--skip-preflight",
        action="store_true",
        help="bypass validation only for an explicit smoke/debug run",
    )
    run.add_argument("--gpus", help="GPU indices for this launch, for example 0,1")
    run.add_argument(
        "--per-gpu",
        type=_int_or_auto,
        help="concurrent tasks per GPU: integer or auto",
    )
    run.add_argument(
        "--jobs",
        type=_int_or_auto,
        help="total concurrent tasks: integer or auto",
    )
    run.add_argument(
        "--resume-checkpoint",
        help="checkpoint path; may contain {seed}, {method}, {stage}, or {dataset}",
    )
    run.add_argument("--resume-method", help="method receiving the checkpoint")
    run.add_argument("--resume-seed", type=int, help="one seed to resume")
    run.add_argument("--resume-stage", help="one stage to resume")
    run.add_argument(
        "--resume-step",
        type=int,
        help="completed update when it cannot be inferred from ckpt_<step>.pt",
    )

    status = commands.add_parser("status", help="show campaign progress")
    _add_config_argument(status)
    status.add_argument("--watch", type=int, default=0, help="refresh every N seconds")

    retry = commands.add_parser("retry-failed", help="requeue failed tasks")
    _add_config_argument(retry)
    retry.add_argument("--stage", help="limit the retry to one stage")

    stop = commands.add_parser("stop", help="stop workers without deleting checkpoints")
    _add_config_argument(stop)
    return parser.parse_args()


def run_campaign(
    config_path: str,
    *,
    skip_preflight: bool = False,
    gpus: str | None = None,
    tasks_per_gpu: int | str | None = None,
    max_jobs: int | str | None = None,
    resume_checkpoint: str | None = None,
    resume_method: str | None = None,
    resume_seed: int | None = None,
    resume_stage: str | None = None,
    resume_step: int | None = None,
) -> int:
    """Load one config, validate it, then hand its tasks to the scheduler."""
    campaign = load_campaign(config_path)
    try:
        apply_run_overrides(
            campaign,
            gpus=gpus,
            per_gpu=tasks_per_gpu,
            jobs=max_jobs,
            resume_checkpoint=resume_checkpoint,
            resume_method=resume_method,
            resume_seed=resume_seed,
            resume_stage=resume_stage,
            resume_step=resume_step,
        )
    except ValueError as exc:
        print(f"[ERROR] run override: {exc}", file=sys.stderr)
        return 2

    if not skip_preflight:
        errors = [check for check in run_preflight(campaign) if check.level == "ERROR"]
        if errors:
            for check in errors:
                print(f"[ERROR] {check.name}: {check.message}", file=sys.stderr)
            print(
                "Refusing to launch. Fix preflight, or use --skip-preflight "
                "only for an explicit smoke/debug run.",
                file=sys.stderr,
            )
            return 2

    # Scheduler.run launches `python -m runner.worker` once per task.
    return Scheduler(campaign).run()


def show_plan(config_path: str) -> int:
    campaign = load_campaign(config_path)
    print(f"Campaign: {campaign.raw['campaign']['name']} ({len(campaign.tasks)} tasks)")
    for task in campaign.tasks:
        print(
            f"p={task.priority:3d} {task.id} {task.stage:32} "
            f"{task.method:16} seed={task.seed} adapter={task.adapter}"
        )
    return 0


def check_campaign(config_path: str) -> int:
    checks = run_preflight(load_campaign(config_path))
    level_order = {"ERROR": 0, "WARN": 1, "PASS": 2}
    for check in sorted(checks, key=lambda item: (level_order[item.level], item.name)):
        print(f"[{check.level:5}] {check.name:24} {check.message}")

    errors = sum(check.level == "ERROR" for check in checks)
    warnings = sum(check.level == "WARN" for check in checks)
    passed = len(checks) - errors - warnings
    print(f"\nPreflight: {passed} PASS, {warnings} WARN, {errors} ERROR")
    return 1 if errors else 0


def campaign_state(campaign: LoadedCampaign) -> Path:
    runs_root = Path(campaign.server["runtime"]["runs_root"])
    campaign_name = campaign.raw["campaign"]["name"]
    return runs_root / campaign_name / "state.sqlite"


def _print_status(rows: list[dict]) -> None:
    counts = Counter(row["status"] for row in rows)
    print("  ".join(f"{status}={count}" for status, count in sorted(counts.items())))
    print(
        f"{'STATUS':10} {'GPU':4} {'ATT':3} {'STAGE':31} "
        f"{'METHOD':15} {'SEED':4} MESSAGE"
    )
    for row in rows:
        task = json.loads(row["payload"])
        gpu = row["gpu_id"] if row["gpu_id"] is not None else "-"
        message = row["message"] or ""
        print(
            f"{row['status'][:10]:10} {str(gpu):4} {row['attempt']:3d} "
            f"{task['stage'][:31]:31} {task['method'][:15]:15} "
            f"{task['seed']:4d} {message[:80]}"
        )


def show_status(config_path: str, watch_seconds: int = 0) -> int:
    while True:
        campaign = load_campaign(config_path)
        state_path = campaign_state(campaign)
        if not state_path.exists():
            print("No state database yet.")
            return 1

        state = StateDB(state_path)
        try:
            rows = state.rows()
        finally:
            state.close()

        if watch_seconds:
            print("\033[2J\033[H", end="")
        _print_status(rows)
        finished = rows and all(row["status"] in TERMINAL_STATES for row in rows)
        if not watch_seconds or finished:
            return 0
        time.sleep(watch_seconds)


def retry_failed(config_path: str, stage: str | None = None) -> int:
    state_path = campaign_state(load_campaign(config_path))
    if not state_path.exists():
        print("No state database yet.")
        return 1
    state = StateDB(state_path)
    try:
        retried = state.retry_failed(stage)
    finally:
        state.close()
    print(f"Requeued {retried} failed task(s).")
    return 0


def stop_workers(config_path: str) -> int:
    state_path = campaign_state(load_campaign(config_path))
    if not state_path.exists():
        print("No running campaign state.")
        return 1

    state = StateDB(state_path)
    stopped = 0
    try:
        for row in state.rows("running"):
            if not row["pid"]:
                continue
            try:
                os.killpg(int(row["pid"]), signal.SIGTERM)
                stopped += 1
                print(f"Sent SIGTERM to task={row['id']} pid={row['pid']}")
            except ProcessLookupError:
                pass
    finally:
        state.close()
    print(f"Stopped {stopped} workers. A later run resumes them from checkpoints.")
    return 0


def _int_or_auto(value: str) -> int | str:
    return value if value == "auto" else int(value)


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"campaign YAML (default: {DEFAULT_CONFIG})",
    )


if __name__ == "__main__":
    main()
