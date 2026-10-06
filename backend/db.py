"""
Application Database
====================
SQLite persistence for threads, messages, tasks and task events. The API and
the worker each open their own connection to the same file (WAL mode allows
concurrent readers alongside a writer).

The agent's own state (LangGraph checkpoints, memory store) lives in separate
SQLite files managed by LangGraph; see backend/resources.py.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import aiosqlite

# Task lifecycle. A task that paused for human approval ends as
# "awaiting_approval"; the approval itself starts a new task on the thread.
ACTIVE_STATUSES = ("queued", "running")
TERMINAL_STATUSES = ("completed", "failed", "cancelled", "awaiting_approval")

SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    steps TEXT NOT NULL DEFAULT '[]',
    files TEXT NOT NULL DEFAULT '{}',
    interrupt TEXT,
    task_id TEXT,
    status TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, created_at);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,              -- "message" | "resume"
    payload TEXT NOT NULL,           -- agent settings + message or decisions
    status TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    result TEXT,
    last_event_id INTEGER NOT NULL DEFAULT 0,
    retry_of TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_thread ON tasks(thread_id, created_at);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);

CREATE TABLE IF NOT EXISTS task_events (
    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    event_id INTEGER NOT NULL,
    thread_id TEXT NOT NULL,
    type TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (task_id, event_id)
);
"""


def now() -> str:
    """Current UTC time as an ISO 8601 string (the format stored in every table)."""
    return datetime.now(timezone.utc).isoformat()


def _loads(value: str | None, default: Any = None) -> Any:
    """Decode a JSON column, returning `default` for NULL or empty values."""
    return json.loads(value) if value else default


class Database:
    """Async access to the app database (one aiosqlite connection per process).

    Methods are grouped by table: threads, messages, tasks, task events.
    Every write commits immediately; there are no multi-statement transactions.
    """

    def __init__(self, path: str):
        """Remember the database path; call connect() before use."""
        self.path = path
        self.conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open the connection, enable WAL/busy-timeout/foreign keys, and create tables."""
        self.conn = await aiosqlite.connect(self.path)
        self.conn.row_factory = aiosqlite.Row
        await self.conn.execute("PRAGMA journal_mode=WAL")
        await self.conn.execute("PRAGMA busy_timeout=5000")
        await self.conn.execute("PRAGMA foreign_keys=ON")
        await self.conn.executescript(SCHEMA)
        await self.conn.commit()

    async def close(self) -> None:
        """Close the connection if it was opened."""
        if self.conn:
            await self.conn.close()

    async def _fetchone(self, sql: str, params: tuple = ()) -> aiosqlite.Row | None:
        """Run a query and return its first row, or None."""
        async with self.conn.execute(sql, params) as cur:
            return await cur.fetchone()

    async def _fetchall(self, sql: str, params: tuple = ()) -> list[aiosqlite.Row]:
        """Run a query and return all rows."""
        async with self.conn.execute(sql, params) as cur:
            return list(await cur.fetchall())

    # ── Threads ────────────────────────────────────────────────────────────
    async def create_thread(self, thread_id: str | None = None, title: str | None = None) -> dict:
        """Create a thread (or do nothing if `thread_id` exists) and return it."""
        thread_id = thread_id or str(uuid.uuid4())
        ts = now()
        await self.conn.execute(
            "INSERT OR IGNORE INTO threads (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (thread_id, title or "New Conversation", ts, ts),
        )
        await self.conn.commit()
        return await self.get_thread(thread_id)

    async def ensure_thread(self, thread_id: str, title: str | None = None) -> dict:
        """Return the thread, creating it first if it does not exist."""
        return await self.get_thread(thread_id) or await self.create_thread(thread_id, title)

    async def get_thread(self, thread_id: str) -> dict | None:
        """Return one thread as a dict, or None if it does not exist."""
        row = await self._fetchone("SELECT * FROM threads WHERE id = ?", (thread_id,))
        return self._thread(row) if row else None

    async def list_threads(self) -> list[dict]:
        """All threads, most recently updated first."""
        rows = await self._fetchall("SELECT * FROM threads ORDER BY updated_at DESC")
        return [self._thread(r) for r in rows]

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread; its messages, tasks and events go with it (ON DELETE CASCADE).

        Returns True if a thread was deleted.
        """
        cur = await self.conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
        await self.conn.commit()
        return cur.rowcount > 0

    async def touch_thread(self, thread_id: str, title: str | None = None) -> None:
        """Bump the thread's updated_at, optionally renaming it at the same time."""
        if title:
            await self.conn.execute(
                "UPDATE threads SET updated_at = ?, title = ? WHERE id = ?", (now(), title, thread_id)
            )
        else:
            await self.conn.execute("UPDATE threads SET updated_at = ? WHERE id = ?", (now(), thread_id))
        await self.conn.commit()

    @staticmethod
    def _thread(row: aiosqlite.Row) -> dict:
        """Convert a threads row to the dict shape the API returns."""
        return {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metadata": _loads(row["metadata"], {}),
        }

    # ── Messages ───────────────────────────────────────────────────────────
    async def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        *,
        steps: list | None = None,
        files: dict | None = None,
        interrupt: dict | None = None,
        task_id: str | None = None,
        status: str | None = None,
    ) -> dict:
        """Append a message to a thread (creating the thread if needed).

        Assistant messages carry the `task_id` that produced them, its final
        `status`, the steps and files shown in the UI, and any approval request
        (`interrupt`). Returns ``{"id": <message id>}``.
        """
        thread = await self.ensure_thread(thread_id)
        msg_id = str(uuid.uuid4())
        await self.conn.execute(
            """INSERT INTO messages (id, thread_id, role, content, steps, files, interrupt,
                                     task_id, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                msg_id, thread_id, role, content,
                json.dumps(steps or []), json.dumps(files or {}),
                json.dumps(interrupt) if interrupt else None,
                task_id, status, now(),
            ),
        )
        await self.conn.commit()
        # Name new threads after their first user message.
        title = None
        if role == "user" and thread["title"] in ("New Conversation", ""):
            title = content[:32] + ("…" if len(content) > 32 else "")
        await self.touch_thread(thread_id, title)
        return {"id": msg_id}

    async def list_messages(self, thread_id: str) -> list[dict]:
        """A thread's messages in the order they were written."""
        rows = await self._fetchall(
            "SELECT * FROM messages WHERE thread_id = ? ORDER BY created_at, rowid", (thread_id,)
        )
        return [
            {
                "id": r["id"],
                "role": r["role"],
                "content": r["content"],
                "steps": _loads(r["steps"], []),
                "files": _loads(r["files"], {}),
                "interrupt": _loads(r["interrupt"]),
                "task_id": r["task_id"],
                "status": r["status"],
                "timestamp": r["created_at"],
            }
            for r in rows
        ]

    async def resolve_pending_interrupt(self, thread_id: str, decisions: list[str]) -> None:
        """Mark the latest unanswered approval request in a thread as answered."""
        rows = await self._fetchall(
            """SELECT id, interrupt FROM messages
               WHERE thread_id = ? AND interrupt IS NOT NULL
               ORDER BY created_at DESC, rowid DESC LIMIT 1""",
            (thread_id,),
        )
        if not rows:
            return
        interrupt = _loads(rows[0]["interrupt"], {})
        if interrupt.get("resolved"):
            return
        interrupt["resolved"] = decisions
        await self.conn.execute(
            "UPDATE messages SET interrupt = ? WHERE id = ?", (json.dumps(interrupt), rows[0]["id"])
        )
        await self.conn.commit()

    # ── Tasks ──────────────────────────────────────────────────────────────
    async def create_task(
        self, thread_id: str, kind: str, payload: dict, retry_of: str | None = None
    ) -> dict:
        """Insert a new task in status "queued" and return it.

        `kind` is "message" (payload has the user message) or "resume" (payload
        has approval decisions); `retry_of` links a retry to the task it repeats.
        """
        task_id = str(uuid.uuid4())
        await self.conn.execute(
            """INSERT INTO tasks (id, thread_id, kind, payload, status, retry_of, created_at)
               VALUES (?, ?, ?, ?, 'queued', ?, ?)""",
            (task_id, thread_id, kind, json.dumps(payload), retry_of, now()),
        )
        await self.conn.commit()
        return await self.get_task(task_id)

    async def get_task(self, task_id: str) -> dict | None:
        """Return one task as a dict, or None if it does not exist."""
        row = await self._fetchone("SELECT * FROM tasks WHERE id = ?", (task_id,))
        return self._task(row) if row else None

    async def list_tasks(self, thread_id: str) -> list[dict]:
        """A thread's tasks, oldest first."""
        rows = await self._fetchall(
            "SELECT * FROM tasks WHERE thread_id = ? ORDER BY created_at", (thread_id,)
        )
        return [self._task(r) for r in rows]

    async def active_task_for_thread(self, thread_id: str) -> dict | None:
        """The thread's queued or running task, if any (there is at most one)."""
        row = await self._fetchone(
            f"""SELECT * FROM tasks WHERE thread_id = ? AND status IN {ACTIVE_STATUSES}
                ORDER BY created_at DESC LIMIT 1""",
            (thread_id,),
        )
        return self._task(row) if row else None

    async def tasks_with_status(self, status: str) -> list[dict]:
        """All tasks currently in `status`, oldest first."""
        rows = await self._fetchall(
            "SELECT * FROM tasks WHERE status = ? ORDER BY created_at", (status,)
        )
        return [self._task(r) for r in rows]

    async def claim_task(self, task_id: str) -> bool:
        """Atomically move a queued task to running. False if someone else got it
        first or it was cancelled while queued."""
        cur = await self.conn.execute(
            "UPDATE tasks SET status = 'running', started_at = ? WHERE id = ? AND status = 'queued'",
            (now(), task_id),
        )
        await self.conn.commit()
        return cur.rowcount == 1

    async def finish_task(
        self, task_id: str, status: str, *, result: dict | None = None, error: str | None = None
    ) -> None:
        """Move a task to a terminal status and store its result or error."""
        progress_sql = ", progress = 100" if status == "completed" else ""
        await self.conn.execute(
            f"""UPDATE tasks SET status = ?, result = ?, error = ?, completed_at = ?{progress_sql}
                WHERE id = ?""",
            (status, json.dumps(result) if result is not None else None, error, now(), task_id),
        )
        await self.conn.commit()

    async def cancel_queued_task(self, task_id: str) -> bool:
        """Cancel a task that has not started yet.

        Returns False if the task already left "queued" (a worker claimed it, so the
        worker has to cancel it instead).
        """
        cur = await self.conn.execute(
            "UPDATE tasks SET status = 'cancelled', completed_at = ? WHERE id = ? AND status = 'queued'",
            (now(), task_id),
        )
        await self.conn.commit()
        return cur.rowcount == 1

    async def set_progress(self, task_id: str, progress: int) -> None:
        """Store a task's progress percentage (derived from its todo plan)."""
        await self.conn.execute("UPDATE tasks SET progress = ? WHERE id = ?", (progress, task_id))
        await self.conn.commit()

    @staticmethod
    def _task(row: aiosqlite.Row) -> dict:
        """Convert a tasks row to the dict shape used by the API and worker."""
        return {
            "task_id": row["id"],
            "thread_id": row["thread_id"],
            "kind": row["kind"],
            "payload": _loads(row["payload"], {}),
            "status": row["status"],
            "progress": row["progress"],
            "error": row["error"],
            "result": _loads(row["result"]),
            "last_event_id": row["last_event_id"],
            "retry_of": row["retry_of"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
        }

    # ── Task events ────────────────────────────────────────────────────────
    async def append_event(self, task_id: str, thread_id: str, type_: str, data: dict) -> dict:
        """Store an event with the next per-task event_id and return its envelope."""
        ts = now()
        payload = json.dumps(data, default=str)
        # Single statement, so the id is allocated atomically even with two writers.
        async with self.conn.execute(
            """INSERT INTO task_events (task_id, event_id, thread_id, type, data, created_at)
               SELECT ?, COALESCE(MAX(event_id), 0) + 1, ?, ?, ?, ?
               FROM task_events WHERE task_id = ?
               RETURNING event_id""",
            (task_id, thread_id, type_, payload, ts, task_id),
        ) as cur:
            event_id = (await cur.fetchone())[0]
        await self.conn.execute(
            "UPDATE tasks SET last_event_id = ? WHERE id = ?", (event_id, task_id)
        )
        await self.conn.commit()
        return {
            "event_id": event_id,
            "task_id": task_id,
            "thread_id": thread_id,
            "type": type_,
            "timestamp": ts,
            "data": json.loads(payload),
        }

    async def events_after(self, task_id: str, after: int, limit: int = 1000) -> list[dict]:
        """A task's stored events with event_id > `after`, in order (at most `limit`)."""
        rows = await self._fetchall(
            """SELECT * FROM task_events WHERE task_id = ? AND event_id > ?
               ORDER BY event_id LIMIT ?""",
            (task_id, after, limit),
        )
        return [
            {
                "event_id": r["event_id"],
                "task_id": r["task_id"],
                "thread_id": r["thread_id"],
                "type": r["type"],
                "timestamp": r["created_at"],
                "data": _loads(r["data"], {}),
            }
            for r in rows
        ]
