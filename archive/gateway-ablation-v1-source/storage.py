"""SQLite cache, execution lifecycle, checkpoints and request journal.

Evaluation results and task completion are committed in the same transaction.
These local guarantees do not make external API side effects exactly-once.
"""

import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


class EpisodeCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS episodes (key TEXT PRIMARY KEY,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS execution_tasks (
          key TEXT PRIMARY KEY,task_id TEXT NOT NULL,state TEXT NOT NULL,
          created REAL NOT NULL,updated REAL NOT NULL,error TEXT);
        CREATE TABLE IF NOT EXISTS attempts (
          id TEXT PRIMARY KEY,task_key TEXT NOT NULL,ordinal INTEGER NOT NULL,
          state TEXT NOT NULL,started REAL NOT NULL,ended REAL,error TEXT);
        CREATE TABLE IF NOT EXISTS checkpoints (
          task_key TEXT PRIMARY KEY,payload TEXT NOT NULL,updated REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS requests (
          id TEXT PRIMARY KEY,task_key TEXT,task_id TEXT,attempt_id TEXT,role TEXT NOT NULL,
          state TEXT NOT NULL,model TEXT NOT NULL,started REAL NOT NULL,ended REAL,
          prompt_tokens INTEGER,completion_tokens INTEGER,http_status INTEGER,error TEXT);
        CREATE INDEX IF NOT EXISTS request_task_idx ON requests(task_key);
        CREATE TABLE IF NOT EXISTS execution_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,task_key TEXT,event TEXT NOT NULL,
          at REAL NOT NULL,details TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS batches (
          id TEXT PRIMARY KEY,metadata TEXT NOT NULL,report TEXT);
        CREATE TABLE IF NOT EXISTS batch_tasks (
          batch_id TEXT NOT NULL,task_key TEXT NOT NULL,position INTEGER NOT NULL,
          PRIMARY KEY(batch_id,task_key));
        """)
        self.db.commit()

    def get(self, key):
        row = self.db.execute(
            "SELECT payload FROM episodes WHERE key=?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, payload):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO episodes VALUES (?,?)",
                (key, json.dumps(payload, ensure_ascii=False)),
            )

    def event(self, key, event, details=None):
        self.db.execute(
            "INSERT INTO execution_events(task_key,event,at,details) VALUES (?,?,?,?)",
            (key, event, time.time(), json.dumps(details or {}, ensure_ascii=False)),
        )

    def enqueue(self, batch_id, entries, metadata):
        now = time.time()
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO batches(id,metadata) VALUES (?,?)",
                (batch_id, json.dumps(metadata)),
            )
            for position, (key, task, repeat) in enumerate(entries):
                self.db.execute(
                    "INSERT OR IGNORE INTO execution_tasks VALUES (?,?,?,?,?,NULL)",
                    (key, task["id"], "queued", now, now),
                )
                self.db.execute(
                    "INSERT OR IGNORE INTO batch_tasks VALUES (?,?,?)",
                    (batch_id, key, position),
                )
                if self.get(key) is not None:
                    self.db.execute(
                        "UPDATE execution_tasks SET state='completed',updated=? WHERE key=?",
                        (now, key),
                    )
                else:
                    row = self.db.execute(
                        "SELECT state FROM execution_tasks WHERE key=?", (key,)
                    ).fetchone()
                    if row["state"] == "running":
                        self.db.execute(
                            "UPDATE attempts SET state='interrupted',ended=?,error='process_interrupted' WHERE task_key=? AND state='running'",
                            (now, key),
                        )
                        self.db.execute(
                            "UPDATE requests SET state='unknown',ended=?,error='process_interrupted' WHERE task_key=? AND state='started'",
                            (now, key),
                        )
                        self.db.execute(
                            "UPDATE execution_tasks SET state='interrupted',updated=?,error='process_interrupted' WHERE key=?",
                            (now, key),
                        )
                        self.event(key, "recovered_running_task")

    def claim(self, key):
        with self.db:
            row = self.db.execute(
                "SELECT state FROM execution_tasks WHERE key=?", (key,)
            ).fetchone()
            if not row or row["state"] == "cancelled":
                raise RuntimeError(
                    "Task cancelled; explicitly retry it before evaluating"
                )
            if row["state"] == "running":
                raise RuntimeError("Task is already running")
            if row["state"] == "completed":
                raise RuntimeError("Completed task must be served from cache")
            attempt_id = uuid.uuid4().hex
            ordinal = (
                self.db.execute(
                    "SELECT COUNT(*) FROM attempts WHERE task_key=?", (key,)
                ).fetchone()[0]
                + 1
            )
            now = time.time()
            self.db.execute(
                "INSERT INTO attempts VALUES (?,?,?,?,?,?,?)",
                (attempt_id, key, ordinal, "running", now, None, None),
            )
            self.db.execute(
                "UPDATE execution_tasks SET state='running',updated=?,error=NULL WHERE key=?",
                (now, key),
            )
            self.event(
                key, "attempt_started", {"attempt_id": attempt_id, "ordinal": ordinal}
            )
        return attempt_id

    def checkpoint(self, key):
        row = self.db.execute(
            "SELECT payload FROM checkpoints WHERE task_key=?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def save_checkpoint(self, key, snapshot):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO checkpoints VALUES (?,?,?)",
                (key, json.dumps(snapshot, ensure_ascii=False), time.time()),
            )
            self.event(
                key,
                "checkpoint_saved",
                {
                    "steps_done": snapshot["steps_done"],
                    "tool_calls": len(snapshot["env"]["events"]),
                },
            )

    def complete(self, key, attempt_id, episode):
        now = time.time()
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO episodes VALUES (?,?)",
                (key, json.dumps(episode, ensure_ascii=False)),
            )
            self.db.execute(
                "UPDATE attempts SET state='completed',ended=? WHERE id=?",
                (now, attempt_id),
            )
            self.db.execute(
                "UPDATE execution_tasks SET state='completed',updated=?,error=NULL WHERE key=?",
                (now, key),
            )
            self.db.execute("DELETE FROM checkpoints WHERE task_key=?", (key,))
            self.event(key, "task_completed", {"model_error": episode.get("error")})

    def interrupt(self, key, attempt_id, reason, state="interrupted"):
        now = time.time()
        with self.db:
            self.db.execute(
                "UPDATE attempts SET state=?,ended=?,error=? WHERE id=?",
                (state, now, reason, attempt_id),
            )
            self.db.execute(
                "UPDATE execution_tasks SET state=?,updated=?,error=? WHERE key=?",
                (state, now, reason, key),
            )
            self.event(key, "attempt_" + state, {"reason": reason})

    def start_request(self, key, task_id, attempt_id, role, model):
        rid = uuid.uuid4().hex
        with self.db:
            self.db.execute(
                "INSERT INTO requests(id,task_key,task_id,attempt_id,role,state,model,started) VALUES (?,?,?,?,?,?,?,?)",
                (rid, key, task_id, attempt_id, role, "started", model, time.time()),
            )
        return rid

    def end_request(
        self, rid, state, prompt=None, completion=None, status=None, error=None
    ):
        with self.db:
            self.db.execute(
                "UPDATE requests SET state=?,ended=?,prompt_tokens=?,completion_tokens=?,http_status=?,error=? WHERE id=?",
                (state, time.time(), prompt, completion, status, error, rid),
            )

    def request_usage(self, key):
        row = self.db.execute(
            """SELECT COUNT(*) requests,COALESCE(SUM(prompt_tokens),0) prompt_tokens,
          COALESCE(SUM(completion_tokens),0) completion_tokens,
          COALESCE(SUM(CASE WHEN prompt_tokens IS NULL OR completion_tokens IS NULL THEN 1 ELSE 0 END),0) usage_missing
          FROM requests WHERE task_key=?""",
            (key,),
        ).fetchone()
        return dict(row)

    def finish_batch(self, batch_id, report):
        with self.db:
            self.db.execute(
                "UPDATE batches SET report=? WHERE id=?", (json.dumps(report), batch_id)
            )

    def task_report(self, key):
        row = self.db.execute(
            "SELECT * FROM execution_tasks WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            raise ValueError("Unknown task key")
        episode = self.get(key)
        return {
            **dict(row),
            "reported_action": episode.get("output", {}).get("action")
            if episode
            else None,
            "model_error": episode.get("error") if episode else None,
            "completion_note": "completed means execution finished, not business success",
            "attempts": [
                dict(r)
                for r in self.db.execute(
                    "SELECT * FROM attempts WHERE task_key=? ORDER BY ordinal", (key,)
                )
            ],
            "usage": self.request_usage(key),
            "has_checkpoint": self.checkpoint(key) is not None,
        }

    def tasks(self):
        return [
            self.task_report(r[0])
            for r in self.db.execute(
                "SELECT key FROM execution_tasks ORDER BY created,key"
            ).fetchall()
        ]

    def cancel(self, key):
        row = self.db.execute(
            "SELECT state FROM execution_tasks WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            raise ValueError("Unknown task key")
        if row["state"] in ("running", "completed"):
            raise ValueError(
                "Cannot cancel a running or completed task; stop the CLI first"
            )
        with self.db:
            self.db.execute(
                "UPDATE execution_tasks SET state='cancelled',updated=? WHERE key=?",
                (time.time(), key),
            )
            self.event(key, "task_cancelled")

    def retry(self, key):
        row = self.db.execute(
            "SELECT state FROM execution_tasks WHERE key=?", (key,)
        ).fetchone()
        if row is None or row["state"] not in ("cancelled", "blocked", "interrupted"):
            raise ValueError(
                "Retry only applies to cancelled/blocked/interrupted tasks; completed model failures stay cached"
            )
        with self.db:
            self.db.execute(
                "UPDATE execution_tasks SET state='queued',updated=?,error=NULL WHERE key=?",
                (time.time(), key),
            )
            self.event(key, "task_retry_requested")

    def execution_summary(self):
        states = {
            r[0]: r[1]
            for r in self.db.execute(
                "SELECT state,COUNT(*) FROM execution_tasks GROUP BY state"
            )
        }
        request_stats = dict(
            self.db.execute("""SELECT COUNT(*) request_attempts,
          COALESCE(SUM(prompt_tokens),0) known_prompt_tokens,
          COALESCE(SUM(completion_tokens),0) known_completion_tokens,
          COALESCE(SUM(CASE WHEN prompt_tokens IS NULL OR completion_tokens IS NULL THEN 1 ELSE 0 END),0) unknown_usage_requests
          FROM requests""").fetchone()
        )
        return {
            "task_states": states,
            "attempts": self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],
            "additional_attempts": self.db.execute(
                "SELECT COUNT(*) FROM attempts WHERE ordinal>1"
            ).fetchone()[0],
            "requests": request_stats,
            "note": "Unknown requests may be billed; local replay control does not guarantee remote exactly-once.",
        }

    def close(self):
        self.db.close()
