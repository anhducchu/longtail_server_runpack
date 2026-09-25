"""Assign pending tasks to GPUs and supervise their worker processes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .config import LoadedCampaign
from .gpu import plan_slots, query_gpus
from .state import StateDB


TERMINAL = {"completed", "failed", "skipped"}


class Scheduler:
    def __init__(self, campaign: LoadedCampaign):
        self.campaign = campaign
        runs_root = Path(campaign.server["runtime"]["runs_root"])
        self.campaign_dir = runs_root / campaign.raw["campaign"]["name"]
        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.campaign_dir / "state.sqlite"
        self.state = StateDB(self.state_path)
        canonical_tasks = [task.to_dict() for task in campaign.tasks]
        self.campaign_fingerprint = hashlib.sha256(
            json.dumps(canonical_tasks, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.state.initialize(campaign.tasks, campaign_fingerprint=self.campaign_fingerprint)
        (self.campaign_dir / "campaign_fingerprint.txt").write_text(self.campaign_fingerprint + "\n", encoding="utf-8")
        self.processes: dict[str, subprocess.Popen] = {}
        self.stop_requested = False
        self.launcher_logs: dict[str, object] = {}
        self._current_slots: dict[int, int] = {}
        self._last_printed_slots: dict[int, int] = {}
        self._last_launch_on_gpu: dict[int, float] = {}

    def _signal(self, signum, frame):
        self.stop_requested = True
        print(f"[runner] scheduler received signal {signum}; no new jobs will be started", flush=True)

    def _allowed_gpus(self) -> list[int]:
        configured = self.campaign.server.get("machine", {}).get("gpu_ids", "auto")
        detected = [g.index for g in query_gpus()]
        if configured == "auto":
            return detected
        return [int(x) for x in configured if int(x) in detected]

    def _task_memory_estimate_gb(self) -> float:
        """Largest observed per-task GPU footprint so far, with a safety margin.

        Falls back to a conservative 12 GB seed until at least one task has
        finished and reported its own peak usage via ``gpu_footprint.json``
        (written by the worker's Monitor thread).
        """
        best = 0.0
        for path in self.campaign_dir.rglob("gpu_footprint.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                peak = float(payload.get("peak_memory_gb", 0))
            except (OSError, ValueError, TypeError):
                continue
            best = max(best, peak)
        return best * 1.15 if best > 0 else 12.0

    def _running_on(self) -> dict[int, int]:
        """How many of our own tasks currently occupy each GPU."""
        counts: dict[int, int] = {}
        for row in self.state.rows("running"):
            gpu_id = row.get("gpu_id")
            if gpu_id is not None:
                counts[int(gpu_id)] = counts.get(int(gpu_id), 0) + 1
        return counts

    def _compute_slots(self, running_on: dict[int, int] | None = None) -> dict[int, int]:
        machine = self.campaign.server.get("machine", {})
        allowed = set(self._allowed_gpus())
        gpus = [g for g in query_gpus() if g.index in allowed]

        tasks_per_gpu_cfg = machine.get("tasks_per_gpu", "auto")
        if tasks_per_gpu_cfg != "auto":
            tasks_per_gpu_cfg = int(tasks_per_gpu_cfg)

        task_memory_cfg = machine.get("task_gpu_memory_gb", "auto")
        task_memory_gb = (
            self._task_memory_estimate_gb()
            if task_memory_cfg == "auto"
            else float(task_memory_cfg)
        )

        slots = plan_slots(
            gpus,
            tasks_per_gpu=tasks_per_gpu_cfg,
            task_memory_gb=task_memory_gb,
            headroom_gb=float(machine.get("gpu_memory_headroom_gb", 4)),
            ceiling=int(machine.get("max_tasks_per_gpu", 4)),
            running_on=running_on if running_on is not None else self._running_on(),
        )
        if slots != self._last_printed_slots:
            print(f"[runner] gpu slots={slots} task_memory_estimate_gb={task_memory_gb:.1f}", flush=True)
            self._last_printed_slots = dict(slots)
        self._current_slots = slots
        return slots

    def _available_slots(self) -> list[int]:
        """GPU indices with open capacity right now (may repeat a GPU index)."""
        machine = self.campaign.server.get("machine", {})
        min_free_mb = float(machine.get("min_free_gpu_memory_gb", 0)) * 1024
        stagger = float(machine.get("same_gpu_launch_stagger_seconds", 0))
        allowed = set(self._allowed_gpus())
        gpus = [g for g in query_gpus() if g.index in allowed]
        running_on = self._running_on()
        slots = self._compute_slots(running_on)

        now = time.time()
        available: list[int] = []
        for gpu in gpus:
            if gpu.memory_free_mb < min_free_mb:
                continue
            busy_here = running_on.get(gpu.index, 0)
            last_launch = self._last_launch_on_gpu.get(gpu.index, 0.0)
            # A GPU that already hosts a task waits out the stagger window
            # before taking another: nvidia-smi's free-memory reading lags a
            # freshly started task, so reading it too soon over-packs into OOM.
            if busy_here > 0 and stagger > 0 and (now - last_launch) < stagger:
                continue
            open_slots = slots.get(gpu.index, 0) - busy_here
            available.extend([gpu.index] * max(0, open_slots))
        return available

    def _disk_ok(self) -> bool:
        limit = float(self.campaign.server.get("machine", {}).get("disk_stop_free_gb", 0))
        free = shutil.disk_usage(self.campaign_dir).free / (1024 ** 3)
        if free < limit:
            print(f"[runner] disk guard: only {free:.1f} GB free (< {limit:.1f} GB); pausing launch", flush=True)
            return False
        return True

    def _worker_env(self) -> dict[str, str]:
        env = os.environ.copy()
        total_slots = sum(self._current_slots.values()) or 1
        max_workers = int(self.campaign.server.get("machine", {}).get("max_dataloader_workers", 8))
        cpu_count = os.cpu_count() or 1
        env["LTX_NUM_WORKERS"] = str(max(2, min(max_workers, cpu_count // total_slots)))
        return env

    def _launch(self, task_id: str, gpu_id: int) -> None:
        command = [
            sys.executable, "-m", "runner.worker",
            "--state", str(self.state_path), "--task", task_id,
            "--gpu", str(gpu_id), "--root", str(self.campaign.root),
        ]
        log = (self.campaign_dir / f"worker_{task_id}.launcher.log").open("a", encoding="utf-8")
        if not self.state.claim(task_id, gpu_id, 0):
            log.close()
            return
        proc = subprocess.Popen(
            command, cwd=str(self.campaign.root), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=self._worker_env(),
        )
        self.state.mark_started(task_id, proc.pid)
        self.processes[task_id] = proc
        self.launcher_logs[task_id] = log
        self._last_launch_on_gpu[gpu_id] = time.time()
        print(f"[runner] launched task={task_id} gpu={gpu_id} pid={proc.pid}", flush=True)

    def _reap(self) -> None:
        finished = []
        for task_id, proc in self.processes.items():
            code = proc.poll()
            if code is not None:
                finished.append((task_id, code))
                print(f"[runner] worker exited task={task_id} code={code}", flush=True)
        for task_id, code in finished:
            # A SIGKILL can terminate the worker before its exception handler
            # writes a terminal state.  Preserve the observed exit code and
            # apply the task's retry policy here.  If the worker did write its
            # result (for example, a killed phase), leave that state intact.
            if self.state.get(task_id)["status"] == "running":
                # The phase subprocess inherits the worker's process group.
                # If the worker itself was SIGKILLed, its child cannot run the
                # worker's cleanup handler and would otherwise keep writing the
                # same run directory while the retry starts.  Kill only this
                # worker-owned group before requeueing; it is never a blanket
                # kill of another user's processes.
                worker_pid = getattr(proc, "pid", None)
                if worker_pid is not None:
                    try:
                        os.killpg(worker_pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                status = self.state.recover_worker_exit(
                    task_id, code, f"Worker exited before updating state (exit {code})",
                )
                print(f"[runner] worker state task={task_id} status={status} code={code}", flush=True)
            self.processes.pop(task_id, None)
            handle = self.launcher_logs.pop(task_id, None)
            if handle is not None:
                handle.close()

    def _ready_tasks(self, rows: list[dict]) -> list[dict]:
        retry_delay = float(
            self.campaign.server.get("retry", {}).get("retry_delay_seconds", 0)
        )
        now = time.time()
        return [
            row
            for row in rows
            if row["status"] == "pending"
            or (
                row["status"] == "retry"
                and now - float(row.get("finished_at") or 0) >= retry_delay
            )
        ]

    def _launch_ready_tasks(self, rows: list[dict], max_concurrent) -> None:
        available_gpus = self._available_slots()
        if max_concurrent != "auto":
            running = len(self.state.rows("running"))
            available_gpus = available_gpus[: max(0, int(max_concurrent) - running)]
        for row, gpu_id in zip(self._ready_tasks(rows), available_gpus):
            self._launch(row["id"], gpu_id)

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self._signal)
        signal.signal(signal.SIGINT, self._signal)
        self.state.reset_stale_running()
        machine = self.campaign.server.get("machine", {})
        poll = int(machine.get("poll_seconds", 20))
        allowed = self._allowed_gpus()
        if not allowed:
            print("[runner] no NVIDIA GPUs detected", flush=True)
            return 2
        max_concurrent = machine.get("max_concurrent", "auto")
        initial_slots = self._compute_slots()
        display_cap = (
            f"auto({sum(initial_slots.values())})"
            if max_concurrent == "auto"
            else max_concurrent
        )
        print(
            f"[runner] campaign={self.campaign.raw['campaign']['name']} "
            f"GPUs={allowed} max_concurrent={display_cap}",
            flush=True,
        )

        while True:
            self._reap()
            recovered = self.state.reset_stale_running()
            if recovered:
                print(f"[runner] recovered {recovered} stale worker(s)", flush=True)
            rows = self.state.rows()
            if rows and all(row["status"] in TERMINAL for row in rows):
                break
            if not self.stop_requested and self._disk_ok():
                self._launch_ready_tasks(rows, max_concurrent)
            if self.stop_requested and not self.processes:
                break
            time.sleep(poll)

        rows = self.state.rows()
        completed = sum(row["status"] == "completed" for row in rows)
        failed = sum(row["status"] == "failed" for row in rows)
        print(
            f"[runner] campaign finished completed={completed} "
            f"failed={failed} total={len(rows)}",
            flush=True,
        )
        self.state.close()
        return 1 if failed else 0
