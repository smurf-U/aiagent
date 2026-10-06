"""
Task Service (API side)
=======================
Creates tasks, enqueues them for the worker, cancels and retries them, and
reads their events (replay from SQLite + live from Redis). Used by the chat,
task and WebSocket routers.
"""

import asyncio
from collections.abc import AsyncIterator

from fastapi import HTTPException

from backend import resources
from backend.db import TERMINAL_STATUSES
from backend.events import REDIS_ERRORS

TERMINAL_EVENTS = ("task_completed", "task_failed", "task_cancelled")


def public_task(task: dict) -> dict:
    """Task status as returned by the API (internal payload omitted)."""
    return {k: v for k, v in task.items() if k not in ("payload", "result")}


async def get_task_or_404(task_id: str) -> dict:
    """Return the task, or raise HTTP 404 if it does not exist."""
    task = await resources.db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


async def ensure_thread_idle(thread_id: str) -> None:
    """Raise HTTP 409 if the thread already has a queued or running task.

    One task per thread at a time keeps the agent's conversation state linear.
    """
    if await resources.db.active_task_for_thread(thread_id):
        raise HTTPException(
            status_code=409,
            detail="This thread already has a task in progress. Wait for it or cancel it.",
        )


async def emit(task: dict, type_: str, data: dict | None = None) -> dict:
    """Store an event for a task and publish it (API-side events such as task_queued)."""
    event = await resources.db.append_event(task["task_id"], task["thread_id"], type_, data or {})
    await resources.bus.publish(event)
    return event


async def create_task(thread_id: str, kind: str, payload: dict, retry_of: str | None = None) -> dict:
    """Create a task on a thread, record task_queued, and queue it for a worker."""
    await ensure_thread_idle(thread_id)
    await resources.db.ensure_thread(thread_id)
    task = await resources.db.create_task(thread_id, kind, payload, retry_of=retry_of)
    await emit(task, "task_queued", {"kind": kind})
    await resources.bus.enqueue(task["task_id"])
    return task


async def cancel_task(task_id: str) -> dict:
    """Cancel a task: finish it here if still queued, else signal its worker.

    Returns the task as it is right after the request (a running task may still
    show "running" until the worker processes the cancel).
    """
    task = await get_task_or_404(task_id)
    if task["status"] in TERMINAL_STATUSES:
        return task
    if await resources.db.cancel_queued_task(task_id):
        # Never reached the worker; finish it here.
        await resources.db.add_message(
            task["thread_id"], "assistant", "⏹ Stopped before it started.",
            task_id=task_id, status="cancelled",
        )
        await emit(task, "task_cancelled", {})
    else:
        # Running: the worker that owns it cancels it and emits task_cancelled.
        await resources.bus.request_cancel(task_id)
    return await resources.db.get_task(task_id)


async def retry_task(task_id: str) -> dict:
    """Run a failed or cancelled task again as a new task with the same payload."""
    task = await get_task_or_404(task_id)
    if task["status"] not in ("failed", "cancelled"):
        raise HTTPException(status_code=409, detail=f"Only failed or cancelled tasks can be retried (status: {task['status']})")
    return await create_task(task["thread_id"], task["kind"], task["payload"], retry_of=task_id)


async def follow_events(task_id: str, after: int, idle_timeout: float) -> AsyncIterator[dict | None]:
    """Yield a task's events with event_id > after: first the stored history,
    then live events as they are published, ending after the terminal event.

    Yields None after `idle_timeout` seconds without events (callers use it for
    keep-alives or to end a long-poll).
    """
    # Subscribe before replaying so nothing published in between is missed;
    # duplicates are filtered by event_id.
    sub = await resources.bus.subscribe(task_id)
    try:
        last = after
        for event in await resources.db.events_after(task_id, last):
            last = event["event_id"]
            yield event
            if event["type"] in TERMINAL_EVENTS:
                return

        task = await resources.db.get_task(task_id)
        if task and task["status"] in TERMINAL_STATUSES and task["last_event_id"] <= last:
            return

        while True:
            try:
                event = await sub.next_event(timeout=idle_timeout)
            except REDIS_ERRORS:
                # Live channel unavailable: degrade to checking the database.
                await asyncio.sleep(1)
                event = None
            if event is None:
                # Cover a publish lost while Redis reconnected: check the DB.
                missed = await resources.db.events_after(task_id, last)
                if not missed:
                    yield None
                    continue
                events = missed
            elif event["event_id"] <= last:
                continue
            elif event["event_id"] > last + 1:
                events = await resources.db.events_after(task_id, last)  # fill a gap
            else:
                events = [event]

            for ev in events:
                last = ev["event_id"]
                yield ev
                if ev["type"] in TERMINAL_EVENTS:
                    return
    finally:
        await sub.close()
