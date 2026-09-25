from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

import psutil

from .adapter import CCUAAdapter
from .config import Task
from .gpu import query_compute_apps
from .metrics import collect_metrics, latest_tensorboard_scalars
from .state import StateDB
from .utils import atomic_write_json, run_capture, shell_join, stable_id


class Monitor(threading.Thread):
    def __init__(self, task_id: str, run_dir: Path, gpu_id: int, state: StateDB, wb_run, interval: int = 30):
        super().__init__(daemon=True)
        self.task_id = task_id
        self.run_dir = run_dir
        self.gpu_id = gpu_id
        self.state = state
        self.wb_run = wb_run
        self.interval = interval
        self.stop_event = threading.Event()
        self.last_images = set()
        self.last_tb_step: Dict[str, int] = {}
        self.peak_memory_gb = 0.0

    def stop(self):
        self.stop_event.set()

    def _write_footprint(self) -> None:
        if self.peak_memory_gb > 0:
            atomic_write_json(self.run_dir / "gpu_footprint.json", {
                "task_id": self.task_id, "gpu_id": self.gpu_id, "peak_memory_gb": self.peak_memory_gb,
            })

    def finalize(self) -> None:
        """Write the final footprint reading once the task has stopped."""
        self._write_footprint()

    def run(self):
        while not self.stop_event.wait(self.interval):
            try:
                # Device-level GPU metrics are shared between any tasks packed
                # onto the same card; also measure this task's own subprocess
                # footprint so packing decisions and per-run W&B usage stay
                # attributable to the right task.
                try:
                    descendant_pids = {p.pid for p in psutil.Process(os.getpid()).children(recursive=True)}
                except Exception:
                    descendant_pids = set()
                proc_memory_mb = sum(mb for pid, mb in query_compute_apps().items() if pid in descendant_pids)
                if proc_memory_mb > 0:
                    self.peak_memory_gb = max(self.peak_memory_gb, proc_memory_mb / 1024)
                # W&B already logs GPU/CPU/RAM/disk natively into its System tab,
                # so mirroring them here only duplicated ten series into the
                # charts. The one thing its built-in cannot express is this
                # task's own share of a GPU that several tasks are packed onto.
                metrics = {"system/proc_gpu_memory_gb": proc_memory_mb / 1024}
                tb = latest_tensorboard_scalars(self.run_dir)
                for tag, (step, value) in tb.items():
                    if step > self.last_tb_step.get(tag, -1):
                        metrics[f"train/{tag}"] = value
                        metrics["train/global_step"] = max(step, metrics.get("train/global_step", 0))
                        self.last_tb_step[tag] = step
                if self.wb_run is not None and metrics:
                    self.wb_run.log(metrics)
                self.state.heartbeat(self.task_id, f"monitor step={metrics.get('train/global_step', 'n/a')}")
                if self.wb_run is not None:
                    for image in sorted(self.run_dir.rglob("*.png")):
                        key = str(image)
                        if key not in self.last_images and image.stat().st_size > 0:
                            try:
                                import wandb
                                self.wb_run.log({"samples/latest": wandb.Image(str(image), caption=image.name)})
                                self.last_images.add(key)
                            except Exception:
                                pass
                self._write_footprint()
            except Exception:
                continue


# Direction of each reported generation metric, so W&B shows the right
# best-value and arrow in run tables instead of guessing.
_METRIC_GOALS = {
    "generation/FID": "minimize", "generation/KID": "minimize",
    "generation/IS": "maximize", "generation/F_8": "maximize", "generation/F_1_8": "maximize",
    "generation/ImprovedPrecision": "maximize", "generation/Recall": "maximize",
}


def wandb_config(task: Task) -> Dict[str, object]:
    """The knobs that actually identify and constrain this run.

    ``task.to_dict()`` is a large dump that buries the protocol under
    absolute paths, retry policy, and the runner's own W&B settings. Config is
    what a reader checks to confirm two rows are comparable, so keep exactly
    the identity and the fairness contract.
    """
    train, evaluate = task.train, task.eval
    config: Dict[str, object] = {
        "campaign": task.campaign,
        "dataset": task.dataset.get("name"),
        "method": task.method,
        "seed": task.seed,
        "adapter": task.adapter,
        "imbalance_factor": task.dataset.get("imbalance_factor"),
        "upstream_repo": task.repository.get("directory"),
        "upstream_commit": task.repository.get("commit"),
        "run_dir": task.run_dir,
    }
    for key in ("data_type", "img_size", "num_class", "num_classes"):
        if key in task.dataset:
            config[f"dataset/{key}"] = task.dataset[key]
    for key in ("total_steps", "batch_size", "lr", "warmup", "T", "dropout", "ema_decay"):
        if key in train:
            config[f"train/{key}"] = train[key]
    for key in ("num_images", "guidance_scale", "sample_method", "ddim_steps", "ddim_skip_step",
                "sampler_family", "metric_protocol", "uniform_labels", "inception_batch_size"):
        if key in evaluate:
            config[f"eval/{key}"] = evaluate[key]
    config.setdefault("eval/inception_batch_size", int(evaluate.get("inception_batch_size", 16)))
    for key in (
        "objective", "cb_tau", "ccua_al", "ccua_ucl",
        "guidance_scale", "guidance_scales",
    ):
        if key in task.method_config:
            config[f"objective/{key}"] = task.method_config[key]
    if task.method_config.get("resume_checkpoint"):
        config["resume/checkpoint"] = str(task.method_config["resume_checkpoint"])
        if task.method_config.get("resume_step"):
            config["resume/step"] = task.method_config["resume_step"]
    for key in ("checkpoint_interval_steps", "auto_resume_max_attempts", "auto_resume_exit_codes"):
        if key in task.retry:
            config[f"reliability/{key}"] = task.retry[key]
    return config


def define_wandb_metrics(wb_run) -> None:
    """Give training curves a real x-axis and the metrics their direction.

    Without this every wandb.log() lands on wandb's auto-incrementing internal
    _step, so a loss curve is plotted against "number of log calls" rather than
    the training step it was actually recorded at.
    """
    if wb_run is None:
        return
    try:
        wb_run.define_metric("train/global_step")
        wb_run.define_metric("train/*", step_metric="train/global_step")
        for name, goal in _METRIC_GOALS.items():
            wb_run.define_metric(name, summary="last", goal=goal)
        wb_run.define_metric("generation/tail/*", summary="last", goal="minimize")
    except Exception as exc:
        print(f"[runner] could not define W&B metric axes: {exc}", flush=True)


def _resolve_wandb_run_identity(task: Task, run_dir: Path) -> tuple[str, str]:
    """Resume the same W&B run only when *this* run_dir is resuming a real
    local checkpoint, not just whenever the same (campaign, task.id) recurs.

    A run id keyed only on task identity collided with a zombie/crashed run
    left over from an earlier, unrelated attempt (different box, wiped
    ``runs/`` dir) and spliced that old history onto a fresh run's step
    counter, producing a step axis that jumps backwards. The id (and a
    matching human-readable start-time label for the display name, so a
    stale attempt is recognizable in the run list without opening it) are
    now stamped with this run_dir's first-seen time and cached next to its
    checkpoint: a crash-resume in the same run_dir reuses the cached
    id/label (one continuous curve, stable name), while a fresh run_dir
    always mints a new pair.
    """
    marker = run_dir / ".wandb_run_id"
    if marker.exists():
        cached = marker.read_text(encoding="utf-8").strip().split("\n")
        if len(cached) == 2 and cached[0]:
            return cached[0], cached[1]
    now = time.time()
    run_id = stable_id(task.campaign, task.id, now, length=16)
    label = time.strftime("%Y%m%d-%H%M%S", time.gmtime(now))
    marker.write_text(f"{run_id}\n{label}", encoding="utf-8")
    return run_id, label


def init_wandb(task: Task, run_dir: Path):
    # An explicit process environment is authoritative for direct worker
    # invocations; the scheduler's serialized task runtime remains the
    # fallback used by normal campaign launches.
    mode = str(os.environ.get("WANDB_MODE", task.runtime.get("wandb_mode", "online"))).lower()
    try:
        import wandb
        run_id, start_label = _resolve_wandb_run_identity(task, run_dir)
        dataset = str(task.dataset.get("name", task.stage))
        return wandb.init(
            project=task.runtime.get("wandb_project", os.environ.get("WANDB_PROJECT", "longtail")),
            entity=task.runtime.get("wandb_entity") or os.environ.get("WANDB_ENTITY") or None,
            # Name by what identifies the row in the table, plus this attempt's
            # start time: the stage name is an internal adapter-grouping label
            # (for example, a generic "core" stage), and an un-dated name is
            # exactly what let a
            # fresh attempt look like a continuation of an old crashed one.
            name=f"{dataset}-{task.method}-s{task.seed}-{start_label}",
            id=run_id,
            resume="allow",
            dir=str(run_dir),
            mode=mode,
            # Group the three seeds of one cell/method so W&B aggregates exactly
            # what the report averages. Grouping by stage instead mixed three
            # different methods into one group.
            group=f"{dataset}-{task.method}",
            job_type="train-eval",
            tags=list(dict.fromkeys(task.tags + [task.adapter, task.method, f"seed-{task.seed}"])),
            config=wandb_config(task),
            settings=wandb.Settings(start_method="thread"),
        )
    except Exception as exc:
        if mode == "online":
            raise RuntimeError(
                "W&B online mode was requested but task logging could not be initialized; "
                "fix WANDB_API_KEY/network/project settings or explicitly set WANDB_MODE=offline "
                "or disabled for a run without remote logging"
            ) from exc
        print(f"[runner] W&B unavailable in mode={mode}: {exc}", flush=True)
        return None


# tqdm redraws with a carriage return, so a 300k-step bar is not "one progress
# line" but 300k of them.  Split the child's output on both terminators so the
# redraws can be thinned out before they reach stdout.log and the W&B artifact.
_OUTPUT_SEGMENT = re.compile(rb"[^\r\n]*[\r\n]")


def run_phase(phase, env: Dict[str, str], log_path: Path, state: StateDB, task_id: str, wb_run,
              progress_seconds: float = 30.0) -> int:
    outputs_exist = phase.skip_if_exists and all(path.exists() for path in phase.skip_if_exists)
    if outputs_exist:
        print(f"[runner] skip phase={phase.name}; outputs exist", flush=True)
        return 0
    phase_env = os.environ.copy()
    phase_env.update(env)
    phase_env.update(phase.env)
    command_text = shell_join(phase.command)
    print(f"[runner] phase={phase.name} cwd={phase.cwd}\n[runner] command={command_text}", flush=True)
    state.heartbeat(task_id, f"phase={phase.name}")
    if wb_run is not None:
        # The resolved command belongs in config (provenance a reader checks
        # once), not in log() — logging the phase *name* pushed a string into
        # the metric stream, where W&B renders it as a junk panel.
        wb_run.config.update({f"command_{phase.name}": command_text}, allow_val_change=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== PHASE {phase.name} =====\n{command_text}\n")
        log.flush()
        proc = subprocess.Popen(
            phase.command, cwd=str(phase.cwd), env=phase_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=False,
        )
        suppressed = 0
        last_progress = 0.0
        last_heartbeat = 0.0

        def emit(segment: bytes) -> None:
            """Write one output segment, keeping progress redraws to a trickle.

            A segment ending in ``\r`` is a progress redraw: it is dropped
            unless `progress_seconds` have passed, and the survivor is stored
            as a real line so stdout.log stays greppable and bounded.  Anything
            ending in ``\n`` is genuine output and always survives.
            """
            nonlocal suppressed, last_progress
            if segment.endswith(b"\r"):
                now = time.monotonic()
                if progress_seconds > 0 and now - last_progress < progress_seconds:
                    suppressed += 1
                    return
                last_progress = now
                segment = segment[:-1] + b"\n"
            text = segment.decode("utf-8", errors="replace")
            # Thinning the output makes the default block buffering visible as
            # a stalled log, so flush what little is left through immediately.
            sys.stdout.write(text)
            sys.stdout.flush()
            log.write(text)
            log.flush()

        try:
            assert proc.stdout is not None
            fd = proc.stdout.fileno()
            pending = b""
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                pending += chunk
                consumed = 0
                for match in _OUTPUT_SEGMENT.finditer(pending):
                    emit(match.group(0))
                    consumed = match.end()
                pending = pending[consumed:]
                # No loss scraping from stdout here: every method writes the
                # same scalars to TensorBoard, which the Monitor mirrors with
                # their true training step. Scraping tqdm's postfix produced a
                # second, step-less copy of the same curve.
                now = time.monotonic()
                if now - last_heartbeat >= 15:
                    last_heartbeat = now
                    state.heartbeat(task_id, f"phase={phase.name} pid={proc.pid}")
            if pending:
                emit(pending + b"\n")
        except KeyboardInterrupt:
            proc.terminate()
            raise
        finally:
            if suppressed:
                note = f"[runner] suppressed {suppressed} progress redraws (one kept per {progress_seconds:g}s)\n"
                sys.stdout.write(note)
                log.write(note)
            log.flush()
        return proc.wait()


def write_provenance(task: Task, run_dir: Path, root: Path) -> None:
    directory = task.repository.get("directory", task.adapter)
    repo = Path(directory)
    if not repo.is_absolute():
        repo = Path(task.runtime["repos_root"]) / repo
    payload = {
        "task_id": task.id, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version, "platform": sys.platform,
        "repo": str(repo),
        "repo_commit": task.repository.get("commit", "vendored-working-tree"),
        "repo_status": "vendored; see third_party/THIRD_PARTY_MANIFEST.json",
        "orchestrator_commit": run_capture(["git", "rev-parse", "HEAD"], root) if (root / ".git").exists() else "unversioned-runpack",
    }
    try:
        import torch
        payload.update({"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                        "cudnn": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available()})
    except Exception as exc:
        payload["torch_error"] = str(exc)
    atomic_write_json(run_dir / "provenance.json", payload)


def resolve_batch_size(task: Task, attempt: int) -> Optional[int]:
    """Pick the batch size for this attempt, honoring the fairness contract.

    A retried task may fall back to a smaller batch after an OOM, but never to
    one larger than the campaign's own configured batch, and never at all for
    a task scored under a locked fairness contract: that table's whole
    point is that every row shares one training budget, so a silently smaller
    (or larger) batch on retry would publish an off-contract row instead.
    """
    contract_batch = int(task.train.get("batch_size", 128))
    if task.eval.get("metric_protocol") in {"unified_cifar_v1", "native_cifar_v1"}:
        return None
    oom_sizes = task.retry.get("oom_batch_sizes", [])
    if attempt <= 1 or not oom_sizes:
        return None
    idx = min(attempt - 1, len(oom_sizes) - 1)
    return min(int(oom_sizes[idx]), contract_batch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    state = StateDB(Path(args.state))
    row = state.get(args.task)
    task = Task(**row["task"])
    state.mark_started(task.id, os.getpid())
    run_dir = Path(task.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    attempt = int(row.get("attempt", 1))
    if attempt > 1:
        print(
            f"[runner] recovery attempt={attempt} task={task.id}; "
            "adapter will resume the newest complete checkpoint",
            flush=True,
        )
    batch_size = resolve_batch_size(task, attempt)
    effective_batch = batch_size if batch_size is not None else int(task.train.get("batch_size", 128))

    resolved = task.to_dict()
    resolved["effective_batch_size"] = effective_batch
    atomic_write_json(run_dir / "task.resolved.json", resolved)
    write_provenance(task, run_dir, Path(args.root))
    (run_dir / "gpu.txt").write_text(str(args.gpu), encoding="utf-8")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_PROJECT", task.runtime.get("wandb_project", "longtail"))
    os.environ.setdefault("WANDB_DIR", str(run_dir / "wandb"))

    wb_run = None
    monitor = None
    stdout_log = run_dir / "stdout.log"
    exit_code = 1
    message = ""
    try:
        wb_run = init_wandb(task, run_dir)
        define_wandb_metrics(wb_run)
        if wb_run is not None:
            wb_run.summary["status"] = "running"
            wb_run.summary["gpu_id"] = args.gpu
            wb_run.summary["effective_batch_size"] = effective_batch

        monitor = Monitor(task.id, run_dir, args.gpu, state, wb_run, interval=int(task.runtime.get("log_system_every_seconds", 30)))
        monitor.start()
        if task.adapter != "ccua":
            raise ValueError(f"unsupported adapter: {task.adapter}")
        adapter = CCUAAdapter(Path(args.root))
        phases = adapter.phases(task, batch_size=batch_size)
        # Vendored trainers may call wandb.init with hardcoded settings during
        # --train, disconnected from this task's run (wrong
        # project, no id, name collides across every task/seed). Disabling wandb
        # inside the subprocess is a no-op for it; the real loss curve already
        # reaches this task's W&B run via the Monitor's tensorboard mirror below.
        env = {"CUDA_VISIBLE_DEVICES": str(args.gpu), "LTX_TASK_ID": task.id, "LTX_RUN_DIR": str(run_dir),
               "WANDB_MODE": "disabled",
               # Thin the redraws out at the source as well, so a 31-hour run
               # does not push a million bar updates through the pipe just to
               # have them dropped here.
               "TQDM_MININTERVAL": str(task.runtime.get("tqdm_min_interval_seconds", 10))}
        progress_seconds = float(task.runtime.get("progress_log_every_seconds", 30))
        for phase in phases:
            code = run_phase(phase, env, stdout_log, state, task.id, wb_run, progress_seconds)
            if code != 0:
                exit_code = code
                raise RuntimeError(f"phase {phase.name} exited with {code}")
        metrics = collect_metrics(run_dir)
        atomic_write_json(run_dir / "metrics.collected.json", metrics)
        if wb_run is not None and metrics:
            wb_run.log(metrics)
            for key, value in metrics.items():
                wb_run.summary[key] = value
        (run_dir / "SUCCESS").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"), encoding="utf-8")
        exit_code = 0
        message = "completed"
        state.finish(task.id, 0, "completed", message)
        if wb_run is not None:
            wb_run.summary["status"] = "completed"
            if os.environ.get("LTX_WANDB_ALERTS", "critical").lower() in {"true", "all"} or (os.environ.get("LTX_WANDB_ALERTS", "critical").lower() == "critical" and "critical" in task.tags):
                try:
                    import wandb
                    wandb.alert(title=f"LongTail finished: {task.method}", text=f"{task.stage}, seed {task.seed}")
                except Exception:
                    pass
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        tail = ""
        if stdout_log.exists():
            tail = stdout_log.read_text(encoding="utf-8", errors="replace")[-10000:]
        is_oom = "out of memory" in (message + tail).lower()
        max_attempts = int(task.retry.get("max_attempts", 2))
        status = state.retry_or_fail(
            task.id,
            exit_code,
            message + (" [OOM]" if is_oom else ""),
            max_attempts=max_attempts,
            retry_exit_codes=task.retry.get("retry_exit_codes"),
            auto_resume_max_attempts=task.retry.get("auto_resume_max_attempts"),
            auto_resume_exit_codes=task.retry.get("auto_resume_exit_codes"),
        )
        if wb_run is not None:
            wb_run.summary["status"] = status
            wb_run.summary["error"] = message
            wb_run.summary["oom_detected"] = is_oom
            if os.environ.get("LTX_WANDB_ALERTS", "critical").lower() not in {"false", "off", "0"}:
                try:
                    import wandb
                    wandb.alert(title=f"LongTail {status}: {task.method}", text=f"{task.stage}, seed {task.seed}: {message}")
                except Exception:
                    pass
    finally:
        if monitor is not None:
            monitor.stop()
            monitor.join(timeout=5)
            monitor.finalize()
        if wb_run is not None:
            try:
                if task.runtime.get("upload_stdout_artifact", True) and stdout_log.exists():
                    import wandb
                    artifact = wandb.Artifact(f"logs-{task.id}", type="run-logs")
                    artifact.add_file(str(stdout_log))
                    artifact.add_file(str(run_dir / "task.resolved.json"))
                    artifact.add_file(str(run_dir / "provenance.json"))
                    if (run_dir / "RESUME_SOURCE.json").exists():
                        artifact.add_file(str(run_dir / "RESUME_SOURCE.json"))
                    if (run_dir / "metrics.collected.json").exists():
                        artifact.add_file(str(run_dir / "metrics.collected.json"))
                    wb_run.log_artifact(artifact)
                wb_run.finish(exit_code=exit_code)
            except Exception:
                pass
        state.close()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
