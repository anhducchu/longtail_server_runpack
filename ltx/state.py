from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import Task

AUTO_RESUME_EXIT_CODES = frozenset((-9, 137))

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY, payload TEXT NOT NULL, status TEXT NOT NULL, priority INTEGER NOT NULL,
    seq INTEGER NOT NULL DEFAULT 0,
    attempt INTEGER NOT NULL DEFAULT 0, gpu_id INTEGER, pid INTEGER, created_at REAL NOT NULL,
    started_at REAL, finished_at REAL, exit_code INTEGER, message TEXT, heartbeat_at REAL
);
CREATE INDEX IF NOT EXISTS idx_tasks_status_priority ON tasks(status, priority DESC);
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class StateDB:
    def __init__(self, path: Path):
        self.path = path; path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
        self.conn.row_factory = sqlite3.Row; self.conn.executescript(SCHEMA)
        self.conn.execute("PRAGMA journal_mode=WAL"); self.conn.execute("PRAGMA synchronous=NORMAL")

    def close(self): self.conn.close()

    def initialize(self, tasks: Iterable[Task], campaign_fingerprint: str | None = None):
        tasks = list(tasks)
        if campaign_fingerprint:
            current = self.conn.execute("SELECT value FROM metadata WHERE key='campaign_fingerprint'").fetchone()
            existing_tasks = int(self.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
            if current is not None and current[0] != campaign_fingerprint:
                # Retry/checkpoint policy is orchestration metadata, not part
                # of the scientific task identity.  Permit an in-place policy
                # upgrade (for example adding SIGKILL recovery) when every
                # other task field is unchanged, and refresh the stored
                # payload so retries use the new policy.  Any protocol,
                # method, dataset, seed, or path change still fails closed.
                stored = self.rows()
                by_id = {row["id"]: json.loads(row["payload"]) for row in stored}
                incoming = {task.id: task.to_dict() for task in tasks}

                def without_retry(payload):
                    value = dict(payload)
                    value.pop("retry", None)
                    return value

                policy_only_change = (
                    set(by_id) == set(incoming)
                    and all(
                        without_retry(by_id[task_id]) == without_retry(incoming[task_id])
                        for task_id in incoming
                    )
                )
                if policy_only_change:
                    with self.conn:
                        for task_id, payload in incoming.items():
                            self.conn.execute(
                                "UPDATE tasks SET payload=? WHERE id=?",
                                (json.dumps(payload, sort_keys=True), task_id),
                            )
                        self.conn.execute(
                            "UPDATE metadata SET value=? WHERE key='campaign_fingerprint'",
                            (campaign_fingerprint,),
                        )
                else:
                    raise RuntimeError(
                        "campaign fingerprint differs from the existing state database; "
                        "use a new campaign name or a new LTX_RUNS_ROOT instead of mixing configurations"
                    )
            if current is None:
                if existing_tasks:
                    raise RuntimeError(
                        "legacy state database has tasks but no campaign fingerprint; "
                        "use a new campaign name or a new LTX_RUNS_ROOT for a reproducible launch"
                    )
                self.conn.execute("INSERT INTO metadata(key,value) VALUES('campaign_fingerprint',?)", (campaign_fingerprint,))
        now = time.time()
        with self.conn:
            # seq preserves the campaign's declared (priority, seed, stage,
            # method) order (see config.py) so a fair-comparison campaign
            # launches seed-major/stage order instead of the task id's hash
            # order, which looked like arbitrary seed jumps to a reader
            # watching the queue drain.
            for seq, task in enumerate(tasks):
                self.conn.execute("INSERT OR IGNORE INTO tasks(id,payload,status,priority,seq,created_at) VALUES(?,?,'pending',?,?,?)",
                                  (task.id, json.dumps(task.to_dict(), sort_keys=True), task.priority, seq, now))

    def rows(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        q = "SELECT * FROM tasks" + (" WHERE status=?" if status else "") + " ORDER BY priority DESC,seq"
        cur = self.conn.execute(q, (status,) if status else ())
        return [dict(r) for r in cur.fetchall()]

    def get(self, task_id: str) -> Dict[str, Any]:
        row = self.conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None: raise KeyError(task_id)
        d = dict(row); d["task"] = json.loads(d["payload"]); return d

    def claim(self, task_id: str, gpu_id: int, pid: int) -> bool:
        now = time.time()
        with self.conn:
            cur = self.conn.execute("""UPDATE tasks SET status='running',gpu_id=?,pid=?,started_at=?,attempt=attempt+1,
                heartbeat_at=?,message=NULL WHERE id=? AND status IN ('pending','retry')""", (gpu_id,pid,now,now,task_id))
        return cur.rowcount == 1

    def mark_started(self, task_id: str, pid: int):
        now=time.time(); self.conn.execute("UPDATE tasks SET status='running',pid=?,started_at=COALESCE(started_at,?),heartbeat_at=? WHERE id=?", (pid,now,now,task_id))

    def heartbeat(self, task_id: str, message: str=""):
        self.conn.execute("UPDATE tasks SET heartbeat_at=?,message=? WHERE id=?", (time.time(),message[-2000:],task_id))

    def finish(self, task_id: str, exit_code: int, status: str, message: str=""):
        self.conn.execute("""UPDATE tasks SET status=?,exit_code=?,finished_at=?,message=?,pid=NULL,gpu_id=NULL,heartbeat_at=? WHERE id=?""",
                          (status,exit_code,time.time(),message[-4000:],time.time(),task_id))

    @staticmethod
    def _exit_code_allowed(exit_code: int, retry_exit_codes) -> bool:
        """Whether an exit code is covered by a retry-code configuration.

        A SIGKILL is reported as ``-9`` by ``Popen`` but often as ``137`` by a
        shell.  They describe the same failure, so either spelling in the
        configuration covers both forms.
        """
        if retry_exit_codes is None:
            return True
        allowed = {int(code) for code in retry_exit_codes}
        return exit_code in allowed or (
            exit_code in AUTO_RESUME_EXIT_CODES and bool(allowed & AUTO_RESUME_EXIT_CODES)
        )

    def _retry_settings(self, row: Dict[str, Any], exit_code: int,
                        max_attempts: Optional[int], retry_exit_codes,
                        auto_resume_max_attempts: Optional[int],
                        auto_resume_exit_codes):
        retry = row["task"].get("retry", {})
        attempts = max_attempts if max_attempts is not None else retry.get("max_attempts", 2)
        exit_codes = retry_exit_codes if retry_exit_codes is not None else retry.get("retry_exit_codes")

        # ``auto_resume_*`` is deliberately limited to the two conventional
        # SIGKILL spellings.  Other failures keep the established retry policy.
        if exit_code in AUTO_RESUME_EXIT_CODES:
            if auto_resume_max_attempts is None:
                auto_resume_max_attempts = retry.get("auto_resume_max_attempts")
            if auto_resume_exit_codes is None:
                auto_resume_exit_codes = retry.get("auto_resume_exit_codes", AUTO_RESUME_EXIT_CODES)
            if auto_resume_max_attempts is not None:
                attempts = auto_resume_max_attempts
            exit_codes = auto_resume_exit_codes
        return int(attempts), exit_codes

    def reset_stale_running(self) -> int:
        import os
        reset=0
        for row in self.rows("running"):
            alive=False
            if row.get("pid"):
                try: os.kill(int(row["pid"]),0); alive=True
                except OSError: pass
            if not alive:
                self.retry_or_fail(
                    row["id"], -9, "Recovered stale worker as SIGKILL",
                )
                reset += 1
        return reset

    def retry_or_fail(self, task_id: str, exit_code: int, message: str,
                      max_attempts: Optional[int] = None, retry_exit_codes=None,
                      auto_resume_max_attempts: Optional[int] = None,
                      auto_resume_exit_codes=None) -> str:
        row = self.get(task_id)
        max_attempts, retry_exit_codes = self._retry_settings(
            row, exit_code, max_attempts, retry_exit_codes,
            auto_resume_max_attempts, auto_resume_exit_codes,
        )
        allowed = self._exit_code_allowed(exit_code, retry_exit_codes)
        status="retry" if allowed and row["attempt"] < max_attempts else "failed"
        self.finish(task_id,exit_code,status,message); return status

    def recover_worker_exit(self, task_id: str, exit_code: int, message: str) -> str:
        """Record a reaped worker only when it did not write its own result."""
        row = self.get(task_id)
        if row["status"] != "running":
            return row["status"]
        return self.retry_or_fail(task_id, exit_code, message)

    def retry_failed(self, stage: str | None = None) -> int:
        n=0
        with self.conn:
            for row in self.rows("failed"):
                task=json.loads(row["payload"])
                if stage and task["stage"] != stage: continue
                self.conn.execute("UPDATE tasks SET status='retry',exit_code=NULL,finished_at=?,message='Manual retry requested' WHERE id=?", (time.time(),row["id"])); n+=1
        return n
