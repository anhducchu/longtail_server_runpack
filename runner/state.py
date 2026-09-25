"""Persistent task state and retry decisions for one campaign."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import Task


AUTO_RESUME_EXIT_CODES = frozenset((-9, 137))

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    status TEXT NOT NULL,
    priority INTEGER NOT NULL,
    seq INTEGER NOT NULL DEFAULT 0,
    attempt INTEGER NOT NULL DEFAULT 0,
    gpu_id INTEGER,
    pid INTEGER,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    exit_code INTEGER,
    message TEXT,
    heartbeat_at REAL
);
CREATE INDEX IF NOT EXISTS idx_tasks_status_priority
    ON tasks(status, priority DESC);
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _without_retry(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the scientific task identity without its retry policy."""
    result = dict(payload)
    result.pop("retry", None)
    return result


class StateDB:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")

    def close(self) -> None:
        self.conn.close()

    def _stored_fingerprint(self) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM metadata WHERE key='campaign_fingerprint'"
        ).fetchone()
        return None if row is None else str(row[0])

    def _update_retry_policy(
        self,
        tasks: list[Task],
        new_fingerprint: str,
    ) -> None:
        """Allow a changed fingerprint only when retry policy is the sole change."""
        stored = {
            row["id"]: json.loads(row["payload"])
            for row in self.rows()
        }
        incoming = {task.id: task.to_dict() for task in tasks}
        same_experiment = (
            set(stored) == set(incoming)
            and all(
                _without_retry(stored[task_id]) == _without_retry(incoming[task_id])
                for task_id in incoming
            )
        )
        if not same_experiment:
            raise RuntimeError(
                "campaign fingerprint differs from the existing state database; "
                "use a new campaign name or LTX_RUNS_ROOT instead of mixing configurations"
            )

        with self.conn:
            for task_id, payload in incoming.items():
                self.conn.execute(
                    "UPDATE tasks SET payload=? WHERE id=?",
                    (json.dumps(payload, sort_keys=True), task_id),
                )
            self.conn.execute(
                "UPDATE metadata SET value=? WHERE key='campaign_fingerprint'",
                (new_fingerprint,),
            )

    def _check_fingerprint(self, tasks: list[Task], fingerprint: str) -> None:
        stored = self._stored_fingerprint()
        task_count = int(self.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])

        if stored is None:
            if task_count:
                raise RuntimeError(
                    "legacy state database has tasks but no campaign fingerprint; "
                    "use a new campaign name or LTX_RUNS_ROOT for a reproducible launch"
                )
            self.conn.execute(
                "INSERT INTO metadata(key, value) VALUES('campaign_fingerprint', ?)",
                (fingerprint,),
            )
        elif stored != fingerprint:
            self._update_retry_policy(tasks, fingerprint)

    def initialize(
        self,
        tasks: Iterable[Task],
        campaign_fingerprint: str | None = None,
    ) -> None:
        tasks = list(tasks)
        if campaign_fingerprint:
            self._check_fingerprint(tasks, campaign_fingerprint)

        created_at = time.time()
        with self.conn:
            for sequence, task in enumerate(tasks):
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO tasks(
                        id, payload, status, priority, seq, created_at
                    ) VALUES (?, ?, 'pending', ?, ?, ?)
                    """,
                    (
                        task.id,
                        json.dumps(task.to_dict(), sort_keys=True),
                        task.priority,
                        sequence,
                        created_at,
                    ),
                )

    def rows(self, status: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM tasks"
        params: tuple[str, ...] = ()
        if status:
            query += " WHERE status=?"
            params = (status,)
        query += " ORDER BY priority DESC, seq"
        return [dict(row) for row in self.conn.execute(query, params).fetchall()]

    def get(self, task_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT * FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
        if row is None:
            raise KeyError(task_id)
        result = dict(row)
        result["task"] = json.loads(result["payload"])
        return result

    def claim(self, task_id: str, gpu_id: int, pid: int) -> bool:
        now = time.time()
        with self.conn:
            cursor = self.conn.execute(
                """
                UPDATE tasks
                SET status='running', gpu_id=?, pid=?, started_at=?,
                    attempt=attempt+1, heartbeat_at=?, message=NULL
                WHERE id=? AND status IN ('pending', 'retry')
                """,
                (gpu_id, pid, now, now, task_id),
            )
        return cursor.rowcount == 1

    def mark_started(self, task_id: str, pid: int) -> None:
        now = time.time()
        self.conn.execute(
            """
            UPDATE tasks
            SET status='running', pid=?, started_at=COALESCE(started_at, ?), heartbeat_at=?
            WHERE id=?
            """,
            (pid, now, now, task_id),
        )

    def heartbeat(self, task_id: str, message: str = "") -> None:
        self.conn.execute(
            "UPDATE tasks SET heartbeat_at=?, message=? WHERE id=?",
            (time.time(), message[-2000:], task_id),
        )

    def finish(self, task_id: str, exit_code: int, status: str, message: str = "") -> None:
        now = time.time()
        self.conn.execute(
            """
            UPDATE tasks
            SET status=?, exit_code=?, finished_at=?, message=?,
                pid=NULL, gpu_id=NULL, heartbeat_at=?
            WHERE id=?
            """,
            (status, exit_code, now, message[-4000:], now, task_id),
        )

    @staticmethod
    def _exit_code_allowed(exit_code: int, configured_codes) -> bool:
        """Treat shell exit 137 and process exit -9 as the same SIGKILL."""
        if configured_codes is None:
            return True
        allowed = {int(code) for code in configured_codes}
        return exit_code in allowed or (
            exit_code in AUTO_RESUME_EXIT_CODES
            and bool(allowed & AUTO_RESUME_EXIT_CODES)
        )

    def _retry_settings(
        self,
        row: dict[str, Any],
        exit_code: int,
        max_attempts: int | None,
        retry_exit_codes,
        auto_resume_max_attempts: int | None,
        auto_resume_exit_codes,
    ):
        policy = row["task"].get("retry", {})
        attempts = max_attempts
        if attempts is None:
            attempts = policy.get("max_attempts", 2)
        exit_codes = retry_exit_codes
        if exit_codes is None:
            exit_codes = policy.get("retry_exit_codes")

        if exit_code in AUTO_RESUME_EXIT_CODES:
            if auto_resume_max_attempts is None:
                auto_resume_max_attempts = policy.get("auto_resume_max_attempts")
            if auto_resume_exit_codes is None:
                auto_resume_exit_codes = policy.get(
                    "auto_resume_exit_codes", AUTO_RESUME_EXIT_CODES
                )
            if auto_resume_max_attempts is not None:
                attempts = auto_resume_max_attempts
            exit_codes = auto_resume_exit_codes

        return int(attempts), exit_codes

    def reset_stale_running(self) -> int:
        recovered = 0
        for row in self.rows("running"):
            alive = False
            if row.get("pid"):
                try:
                    os.kill(int(row["pid"]), 0)
                    alive = True
                except OSError:
                    pass
            if not alive:
                self.retry_or_fail(
                    row["id"], -9, "Recovered stale worker as SIGKILL"
                )
                recovered += 1
        return recovered

    def retry_or_fail(
        self,
        task_id: str,
        exit_code: int,
        message: str,
        max_attempts: int | None = None,
        retry_exit_codes=None,
        auto_resume_max_attempts: int | None = None,
        auto_resume_exit_codes=None,
    ) -> str:
        row = self.get(task_id)
        max_attempts, retry_exit_codes = self._retry_settings(
            row,
            exit_code,
            max_attempts,
            retry_exit_codes,
            auto_resume_max_attempts,
            auto_resume_exit_codes,
        )
        may_retry = self._exit_code_allowed(exit_code, retry_exit_codes)
        status = "retry" if may_retry and row["attempt"] < max_attempts else "failed"
        self.finish(task_id, exit_code, status, message)
        return status

    def recover_worker_exit(self, task_id: str, exit_code: int, message: str) -> str:
        """Record a reaped worker only if it did not write its own result."""
        row = self.get(task_id)
        if row["status"] != "running":
            return row["status"]
        return self.retry_or_fail(task_id, exit_code, message)

    def retry_failed(self, stage: str | None = None) -> int:
        retried = 0
        with self.conn:
            for row in self.rows("failed"):
                task = json.loads(row["payload"])
                if stage and task["stage"] != stage:
                    continue
                self.conn.execute(
                    """
                    UPDATE tasks
                    SET status='retry', exit_code=NULL, finished_at=?,
                        message='Manual retry requested'
                    WHERE id=?
                    """,
                    (time.time(), row["id"]),
                )
                retried += 1
        return retried
