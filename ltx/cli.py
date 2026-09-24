from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import signal
import sys
import time
from collections import Counter
from pathlib import Path

from .config import load_campaign
from .preflight import run_preflight
from .scheduler import Scheduler
from .state import StateDB
from .utils import load_runtime_env


def campaign_state(campaign):
    return Path(campaign.server["runtime"]["runs_root"]) / campaign.raw["campaign"]["name"] / "state.sqlite"


def cmd_preflight(args) -> int:
    campaign = load_campaign(args.config)
    checks = run_preflight(campaign)
    order = {"ERROR": 0, "WARN": 1, "PASS": 2}
    for c in sorted(checks, key=lambda x: (order.get(x.level, 9), x.name)):
        print(f"[{c.level:5}] {c.name:24} {c.message}")
    errors = sum(c.level == "ERROR" for c in checks)
    warns = sum(c.level == "WARN" for c in checks)
    print(f"\nPreflight: {len(checks)-errors-warns} PASS, {warns} WARN, {errors} ERROR")
    return 1 if errors else 0


def cmd_plan(args) -> int:
    campaign = load_campaign(args.config)
    print(f"Campaign: {campaign.raw['campaign']['name']} ({len(campaign.tasks)} tasks)")
    for task in campaign.tasks:
        print(f"p={task.priority:3d} {task.id} {task.stage:32} {task.method:16} seed={task.seed} adapter={task.adapter}")
    return 0


def apply_machine_overrides(campaign, args) -> None:
    """Let --gpus/--per-gpu/--jobs override configs/server.yaml for one launch
    without editing the file, so the same one-command entrypoint adapts to
    whatever GPU box it happens to land on."""
    machine = campaign.server.setdefault("machine", {})
    if getattr(args, "gpus", None):
        machine["gpu_ids"] = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    if getattr(args, "per_gpu", None) is not None:
        machine["tasks_per_gpu"] = args.per_gpu
    if getattr(args, "jobs", None) is not None:
        machine["max_concurrent"] = args.jobs


def apply_resume_override(campaign, args) -> None:
    """Attach an explicitly requested external checkpoint to selected tasks.

    The override is part of each task payload, so the campaign fingerprint
    changes and an old state database cannot silently mix a different
    provenance.  ``{seed}``, ``{method}``, ``{stage}``, and ``{dataset}`` are
    supported in the path for a per-task checkpoint layout.
    """
    checkpoint_template = getattr(args, "resume_checkpoint", None)
    resume_step = getattr(args, "resume_step", None)
    method = getattr(args, "resume_method", None)
    seed = getattr(args, "resume_seed", None)
    stage = getattr(args, "resume_stage", None)
    if not checkpoint_template:
        if resume_step is not None or method or seed is not None or stage:
            raise ValueError("resume override options require --resume-checkpoint")
        return
    if not method:
        raise ValueError("--resume-checkpoint requires --resume-method so another method cannot be resumed accidentally")
    selected = [task for task in campaign.tasks
                if task.method == method
                and (seed is None or task.seed == seed)
                and (stage is None or task.stage == stage)]
    if not selected:
        raise ValueError(
            f"no campaign task matches --resume-method={method!r} "
            f"--resume-seed={seed!r} --resume-stage={stage!r}"
        )
    supported_adapters = {"ccua"}
    unsupported = sorted({task.adapter for task in selected if task.adapter not in supported_adapters})
    if unsupported:
        raise ValueError(
            "external resume overrides currently support CCUA-DDPM tasks only; "
            f"selected adapters={unsupported}"
        )
    if seed is None and len(selected) > 1 and "{seed}" not in str(checkpoint_template):
        raise ValueError(
            "multiple seeds match; use --resume-seed N for one checkpoint or "
            "include {seed} in --resume-checkpoint"
        )
    if len({task.stage for task in selected}) > 1 and "{stage}" not in str(checkpoint_template):
        raise ValueError("multiple stages match; use --resume-stage or include {stage} in --resume-checkpoint")
    if len({str(task.dataset.get("name", "")) for task in selected}) > 1 and "{dataset}" not in str(checkpoint_template):
        raise ValueError("multiple datasets match; use --resume-stage or include {dataset} in --resume-checkpoint")
    selected_ids = {task.id for task in selected}
    updated = []
    for task in campaign.tasks:
        if task.id not in selected_ids:
            updated.append(task)
            continue
        try:
            checkpoint = str(checkpoint_template).format(
                seed=task.seed, method=task.method, stage=task.stage,
                dataset=task.dataset.get("name", ""),
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "--resume-checkpoint may use only {seed}, {method}, {stage}, and {dataset} placeholders"
            ) from exc
        cfg = dict(task.method_config)
        cfg["resume_checkpoint"] = str(Path(checkpoint).expanduser().resolve())
        if resume_step is not None:
            cfg["resume_step"] = int(resume_step)
        updated.append(replace(task, method_config=cfg))
    campaign.tasks = updated


def cmd_run(args) -> int:
    campaign = load_campaign(args.config)
    try:
        apply_machine_overrides(campaign, args)
        apply_resume_override(campaign, args)
    except ValueError as exc:
        print(f"[ERROR] run override: {exc}", file=sys.stderr)
        return 2
    if not args.skip_preflight:
        checks = run_preflight(campaign)
        errors = [c for c in checks if c.level == "ERROR"]
        if errors:
            for c in errors:
                print(f"[ERROR] {c.name}: {c.message}", file=sys.stderr)
            print("Refusing to launch. Fix preflight or use --skip-preflight only for an explicit smoke/debug run.", file=sys.stderr)
            return 2
    return Scheduler(campaign).run()


def cmd_status(args) -> int:
    while True:
        campaign = load_campaign(args.config)
        path = campaign_state(campaign)
        if not path.exists():
            print("No state database yet.")
            return 1
        db = StateDB(path); rows = db.rows(); counts = Counter(r["status"] for r in rows)
        if getattr(args, "watch", 0):
            print("\033[2J\033[H", end="")
        print("  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        print(f"{'STATUS':10} {'GPU':4} {'ATT':3} {'STAGE':31} {'METHOD':15} {'SEED':4} MESSAGE")
        for row in rows:
            task = json.loads(row["payload"])
            print(f"{row['status'][:10]:10} {str(row.get('gpu_id') if row.get('gpu_id') is not None else '-'):4} "
                  f"{row['attempt']:3d} {task['stage'][:31]:31} {task['method'][:15]:15} {task['seed']:4d} {(row.get('message') or '')[:80]}")
        db.close()
        if not getattr(args, "watch", 0): return 0
        if rows and all(r["status"] in {"completed","failed","skipped"} for r in rows): return 0
        time.sleep(args.watch)


def cmd_retry_failed(args) -> int:
    campaign = load_campaign(args.config); path = campaign_state(campaign)
    if not path.exists(): print("No state database yet."); return 1
    db = StateDB(path); n = db.retry_failed(getattr(args, "stage", None)); db.close()
    print(f"Requeued {n} failed task(s).")
    return 0


def cmd_stop(args) -> int:
    campaign = load_campaign(args.config)
    path = campaign_state(campaign)
    if not path.exists():
        print("No running campaign state.")
        return 1
    db = StateDB(path)
    killed = 0
    for row in db.rows("running"):
        pid = row.get("pid")
        if pid:
            try:
                os.killpg(int(pid), signal.SIGTERM)
                killed += 1
                print(f"Sent SIGTERM to task={row['id']} pid={pid}")
            except ProcessLookupError:
                pass
    db.close()
    print(f"Stopped {killed} workers. A later run resumes them from checkpoints when supported.")
    return 0


def main() -> None:
    root = Path.cwd()
    load_runtime_env(root)
    parser = argparse.ArgumentParser(prog="ltx")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "plan", "stop"):
        p = sub.add_parser(name); p.add_argument("--config", default="configs/native_cifar100_if100.yaml")
    status = sub.add_parser("status"); status.add_argument("--config", default="configs/native_cifar100_if100.yaml")
    status.add_argument("--watch", type=int, default=0, help="refresh every N seconds")
    retry_failed = sub.add_parser("retry-failed"); retry_failed.add_argument("--config", default="configs/native_cifar100_if100.yaml")
    retry_failed.add_argument("--stage", default=None)
    def int_or_auto(value: str):
        return value if value == "auto" else int(value)

    run = sub.add_parser("run")
    run.add_argument("--config", default="configs/native_cifar100_if100.yaml")
    run.add_argument("--skip-preflight", action="store_true")
    run.add_argument("--gpus", default=None, help="comma-separated GPU indices, e.g. 0,1,2,3 (default: all detected)")
    run.add_argument("--per-gpu", dest="per_gpu", type=int_or_auto, default=None,
                      help="tasks per GPU: an integer, or 'auto' to pack by free VRAM (default: config's machine.tasks_per_gpu)")
    run.add_argument("--jobs", type=int_or_auto, default=None,
                      help="cap on total concurrent tasks across all GPUs, or 'auto' (default: config's machine.max_concurrent)")
    run.add_argument("--resume-checkpoint", default=None,
                     help="explicit checkpoint path; may contain {seed}/{method} (requires --resume-method)")
    run.add_argument("--resume-method", default=None,
                     help="method whose task(s) receive --resume-checkpoint, e.g. ddpm")
    run.add_argument("--resume-seed", type=int, default=None,
                     help="one seed to resume; omit only when the checkpoint path contains {seed}")
    run.add_argument("--resume-stage", default=None,
                     help="one campaign stage to resume when a method occurs in multiple stages")
    run.add_argument("--resume-step", type=int, default=None,
                     help="completed update number when the checkpoint filename is not ckpt_<step>.pt")
    args = parser.parse_args()
    handlers = {
        "preflight": cmd_preflight, "plan": cmd_plan, "run": cmd_run,
        "status": cmd_status, "stop": cmd_stop, "retry-failed": cmd_retry_failed,
    }
    raise SystemExit(handlers[args.command](args))


if __name__ == "__main__":
    main()
