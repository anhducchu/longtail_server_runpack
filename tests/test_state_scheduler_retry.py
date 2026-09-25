import os
from pathlib import Path

from runner.config import LoadedCampaign, Task
from runner.scheduler import Scheduler
from runner.state import StateDB


TASK_ID = "retry-task"


def make_task(retry):
    return Task(
        id=TASK_ID,
        campaign="retry-campaign",
        stage="train",
        adapter="test",
        method="test",
        seed=0,
        priority=1,
        dataset={},
        train={},
        eval={},
        method_config={},
        repository={},
        runtime={},
        retry=retry,
    )


def make_state(tmp_path: Path, retry):
    state = StateDB(tmp_path / "state.sqlite")
    state.initialize([make_task(retry)])
    return state


class ExitedProcess:
    def __init__(self, exit_code):
        self.exit_code = exit_code

    def poll(self):
        return self.exit_code


def test_kill_exit_defaults_to_auto_resume_codes_when_not_configured(tmp_path):
    state = make_state(tmp_path, {"max_attempts": 2, "retry_exit_codes": [1]})
    try:
        assert state.claim(TASK_ID, gpu_id=0, pid=10)
        assert state.retry_or_fail(TASK_ID, -9, "phase killed", 2, [1]) == "retry"
        row = state.get(TASK_ID)
        assert row["status"] == "retry"
        assert row["exit_code"] == -9

        assert state.claim(TASK_ID, gpu_id=0, pid=11)
        assert state.retry_or_fail(TASK_ID, 137, "phase killed", 2, [1]) == "failed"
        assert state.get(TASK_ID)["exit_code"] == 137
    finally:
        state.close()


def test_retry_or_fail_accepts_explicit_auto_resume_policy(tmp_path):
    state = make_state(tmp_path, {"max_attempts": 4, "retry_exit_codes": [1]})
    try:
        assert state.claim(TASK_ID, gpu_id=0, pid=10)
        assert state.retry_or_fail(
            TASK_ID, -9, "phase killed", max_attempts=4, retry_exit_codes=[1],
            auto_resume_max_attempts=2, auto_resume_exit_codes=[137],
        ) == "retry"

        assert state.claim(TASK_ID, gpu_id=0, pid=11)
        assert state.retry_or_fail(
            TASK_ID, 137, "phase killed", max_attempts=4, retry_exit_codes=[1],
            auto_resume_max_attempts=2, auto_resume_exit_codes=[137],
        ) == "failed"
    finally:
        state.close()


def test_legacy_retry_policy_is_unchanged_for_non_kill_exit(tmp_path):
    state = make_state(tmp_path, {"max_attempts": 2, "retry_exit_codes": [1]})
    try:
        assert state.claim(TASK_ID, gpu_id=0, pid=10)
        assert state.retry_or_fail(TASK_ID, 1, "ordinary failure", 2, [1]) == "retry"

        assert state.claim(TASK_ID, gpu_id=0, pid=11)
        assert state.retry_or_fail(TASK_ID, 2, "ordinary failure", 2, [1]) == "failed"
    finally:
        state.close()


def test_scheduler_reap_records_killed_worker_using_auto_resume_policy(tmp_path):
    task = make_task({
        "max_attempts": 4,
        "retry_exit_codes": [1],
        "auto_resume_exit_codes": [137],
        "auto_resume_max_attempts": 2,
    })
    campaign = LoadedCampaign(
        root=tmp_path,
        raw={"campaign": {"name": "retry-campaign"}},
        server={"runtime": {"runs_root": str(tmp_path / "runs")}},
        tasks=[task],
    )
    scheduler = Scheduler(campaign)
    try:
        assert scheduler.state.claim(TASK_ID, gpu_id=0, pid=10)
        scheduler.processes[TASK_ID] = ExitedProcess(-9)
        scheduler._reap()
        row = scheduler.state.get(TASK_ID)
        assert row["status"] == "retry"
        assert row["exit_code"] == -9

        assert scheduler.state.claim(TASK_ID, gpu_id=0, pid=11)
        scheduler.processes[TASK_ID] = ExitedProcess(137)
        scheduler._reap()
        row = scheduler.state.get(TASK_ID)
        assert row["status"] == "failed"
        assert row["exit_code"] == 137
    finally:
        scheduler.state.close()


def test_stale_workers_are_recovered_as_sigkill_without_unbounded_retries(tmp_path, monkeypatch):
    state = make_state(tmp_path, {
        "max_attempts": 4,
        "retry_exit_codes": [1],
        "auto_resume_max_attempts": 2,
    })

    def no_such_process(pid, signal_number):
        raise ProcessLookupError(pid)

    monkeypatch.setattr(os, "kill", no_such_process)
    try:
        assert state.claim(TASK_ID, gpu_id=0, pid=10)
        assert state.reset_stale_running() == 1
        row = state.get(TASK_ID)
        assert row["status"] == "retry"
        assert row["exit_code"] == -9

        assert state.claim(TASK_ID, gpu_id=0, pid=11)
        assert state.reset_stale_running() == 1
        row = state.get(TASK_ID)
        assert row["status"] == "failed"
        assert row["exit_code"] == -9
        assert row["pid"] is None
        assert row["gpu_id"] is None
    finally:
        state.close()


def test_state_migrates_only_retry_policy_changes(tmp_path):
    state = StateDB(tmp_path / "state.sqlite")
    old_task = make_task({"max_attempts": 2, "retry_exit_codes": [1]})
    new_task = make_task({
        "max_attempts": 2,
        "retry_exit_codes": [1],
        "auto_resume_exit_codes": [-9, 137],
        "auto_resume_max_attempts": 8,
        "checkpoint_interval_steps": 25000,
    })
    try:
        state.initialize([old_task], campaign_fingerprint="old-policy")
        state.initialize([new_task], campaign_fingerprint="new-policy")
        assert state.get(TASK_ID)["task"]["retry"]["auto_resume_max_attempts"] == 8
    finally:
        state.close()
