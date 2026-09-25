"""Command-line entrypoint for planning and running baseline experiments."""

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


def campaign_state(campaign: LoadedCampaign) -> Path:
    runs_root = Path(campaign.server["runtime"]["runs_root"])
    campaign_name = campaign.raw["campaign"]["name"]
    return runs_root / campaign_name / "state.sqlite"


def cmd_plan(args: argparse.Namespace) -> int:
    campaign = load_campaign(args.config)
    print(f"Campaign: {campaign.raw['campaign']['name']} ({len(campaign.tasks)} tasks)")
    for task in campaign.tasks:
        print(
            f"p={task.priority:3d} {task.id} {task.stage:32} "
            f"{task.method:16} seed={task.seed} adapter={task.adapter}"
        )
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    checks = run_preflight(load_campaign(args.config))
    level_order = {"ERROR": 0, "WARN": 1, "PASS": 2}
    for check in sorted(checks, key=lambda item: (level_order[item.level], item.name)):
        print(f"[{check.level:5}] {check.name:24} {check.message}")

    errors = sum(check.level == "ERROR" for check in checks)
    warnings = sum(check.level == "WARN" for check in checks)
    passed = len(checks) - errors - warnings
    print(f"\nPreflight: {passed} PASS, {warnings} WARN, {errors} ERROR")
    return 1 if errors else 0


def cmd_run(args: argparse.Namespace) -> int:
    campaign = load_campaign(args.config)
    try:
        apply_run_overrides(
            campaign,
            gpus=args.gpus,
            per_gpu=args.per_gpu,
            jobs=args.jobs,
            resume_checkpoint=args.resume_checkpoint,
            resume_method=args.resume_method,
            resume_seed=args.resume_seed,
            resume_stage=args.resume_stage,
            resume_step=args.resume_step,
        )
    except ValueError as exc:
        print(f"[ERROR] run override: {exc}", file=sys.stderr)
        return 2

    if not args.skip_preflight:
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
    return Scheduler(campaign).run()


def _print_status(rows: list[dict]) -> None:
    counts = Counter(row["status"] for row in rows)
    print("  ".join(f"{status}={count}" for status, count in sorted(counts.items())))
    print(f"{'STATUS':10} {'GPU':4} {'ATT':3} {'STAGE':31} {'METHOD':15} {'SEED':4} MESSAGE")
    for row in rows:
        task = json.loads(row["payload"])
        gpu = row["gpu_id"] if row["gpu_id"] is not None else "-"
        message = row["message"] or ""
        print(
            f"{row['status'][:10]:10} {str(gpu):4} {row['attempt']:3d} "
            f"{task['stage'][:31]:31} {task['method'][:15]:15} "
            f"{task['seed']:4d} {message[:80]}"
        )


def cmd_status(args: argparse.Namespace) -> int:
    while True:
        campaign = load_campaign(args.config)
        state_path = campaign_state(campaign)
        if not state_path.exists():
            print("No state database yet.")
            return 1

        state = StateDB(state_path)
        try:
            rows = state.rows()
        finally:
            state.close()

        if args.watch:
            print("\033[2J\033[H", end="")
        _print_status(rows)
        if not args.watch or (rows and all(row["status"] in TERMINAL_STATES for row in rows)):
            return 0
        time.sleep(args.watch)


def cmd_retry_failed(args: argparse.Namespace) -> int:
    state_path = campaign_state(load_campaign(args.config))
    if not state_path.exists():
        print("No state database yet.")
        return 1
    state = StateDB(state_path)
    try:
        retried = state.retry_failed(args.stage)
    finally:
        state.close()
    print(f"Requeued {retried} failed task(s).")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    state_path = campaign_state(load_campaign(args.config))
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


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python main.py")
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="show the tasks without running them")
    _add_config(plan)
    plan.set_defaults(handler=cmd_plan)

    preflight = commands.add_parser("preflight", help="validate the machine and protocol")
    _add_config(preflight)
    preflight.set_defaults(handler=cmd_preflight)

    run = commands.add_parser("run", help="run or resume a campaign")
    _add_config(run)
    run.add_argument("--skip-preflight", action="store_true")
    run.add_argument("--gpus", help="comma-separated GPU indices")
    run.add_argument("--per-gpu", type=_int_or_auto, help="tasks per GPU, or auto")
    run.add_argument("--jobs", type=_int_or_auto, help="total concurrent tasks, or auto")
    run.add_argument("--resume-checkpoint", help="checkpoint path; may contain {seed}/{method}")
    run.add_argument("--resume-method", help="method receiving the checkpoint")
    run.add_argument("--resume-seed", type=int, help="one seed to resume")
    run.add_argument("--resume-stage", help="one stage to resume")
    run.add_argument("--resume-step", type=int, help="step when absent from the filename")
    run.set_defaults(handler=cmd_run)

    status = commands.add_parser("status", help="show campaign progress")
    _add_config(status)
    status.add_argument("--watch", type=int, default=0, help="refresh every N seconds")
    status.set_defaults(handler=cmd_status)

    retry = commands.add_parser("retry-failed", help="requeue failed tasks")
    _add_config(retry)
    retry.add_argument("--stage")
    retry.set_defaults(handler=cmd_retry_failed)

    stop = commands.add_parser("stop", help="stop running workers")
    _add_config(stop)
    stop.set_defaults(handler=cmd_stop)
    return parser


def main() -> None:
    load_runtime_env(Path.cwd())
    args = build_parser().parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
