"""
Tasks API Router
================
Observe and control task executions. All transports deliver the same event
envelope ({event_id, task_id, thread_id, type, timestamp, data}) and can
resume from a last seen event_id:

- WebSocket  /ws/tasks/{task_id}?last_event_id=N   primary; accepts {"type": "cancel"}
- SSE        /api/tasks/{task_id}/stream           EventSource, resumes via Last-Event-ID
- Polling    /api/tasks/{task_id}/events?after=N   fallback; each request returns within
                                                   TASK_POLL_WAIT_SECONDS
"""

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Header, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketDisconnected, WebSocketState
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend import resources, tasks
from backend.config import settings
from backend.db import TERMINAL_STATUSES

router = APIRouter(prefix="/api/tasks", tags=["tasks"])
ws_router = APIRouter(tags=["tasks"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
KEEPALIVE_SECONDS = 15


class TaskEvents(BaseModel):
    """One polling response: events after the requested id, and whether the task is over."""

    task_id: str
    status: str
    events: list[dict[str, Any]]
    last_event_id: int
    done: bool


@router.get("/{task_id}")
async def get_task(task_id: str):
    """Current status of a task: queued, running, completed, failed, cancelled or awaiting_approval."""
    return tasks.public_task(await tasks.get_task_or_404(task_id))


@router.get("/{task_id}/result")
async def get_task_result(task_id: str):
    """Final answer and files of a finished task."""
    task = await tasks.get_task_or_404(task_id)
    return {
        "task_id": task_id,
        "status": task["status"],
        "result": task["result"],
        "error": task["error"],
    }


@router.get("/{task_id}/events", response_model=TaskEvents)
async def get_task_events(task_id: str, after: int = 0, wait: float | None = None):
    """Events with event_id > `after`. Waits up to `wait` seconds (capped at
    TASK_POLL_WAIT_SECONDS) for the first new event, then briefly for more, so
    a burst of tokens comes back in one response."""
    await tasks.get_task_or_404(task_id)
    wait = settings.TASK_POLL_WAIT_SECONDS if wait is None else min(wait, settings.TASK_POLL_WAIT_SECONDS)

    events: list[dict] = []
    stream = tasks.follow_events(task_id, after, idle_timeout=wait)
    try:
        deadline = None
        while True:
            timeout = None if deadline is None else max(0.0, deadline - asyncio.get_running_loop().time())
            try:
                event = await asyncio.wait_for(anext(stream), timeout=timeout)
            except (StopAsyncIteration, asyncio.TimeoutError):
                break
            if event is None:  # idle for `wait` seconds
                break
            events.append(event)
            if deadline is None:  # linger 250 ms after the first event to batch
                deadline = asyncio.get_running_loop().time() + 0.25
    finally:
        await stream.aclose()

    task = await resources.db.get_task(task_id)
    last = events[-1]["event_id"] if events else after
    return TaskEvents(
        task_id=task_id,
        status=task["status"],
        events=events,
        last_event_id=last,
        done=task["status"] in TERMINAL_STATUSES and last >= task["last_event_id"],
    )


@router.get("/{task_id}/stream")
async def stream_task_events(task_id: str, last_event_id: int = Header(0)):
    """Server-Sent Events, resuming after the Last-Event-ID header if sent.
    Sends a comment ping every 15 s of silence for idle-timeout proxies."""
    await tasks.get_task_or_404(task_id)

    async def event_stream():
        """Yield SSE frames: one per event, with ': ping' comments while idle."""
        async for event in tasks.follow_events(task_id, last_event_id, KEEPALIVE_SECONDS):
            if event is None:
                yield ": ping\n\n"
            else:
                yield f"id: {event['event_id']}\nevent: {event['type']}\ndata: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Stop a queued or running task. Running tasks stop at the worker's next await."""
    return tasks.public_task(await tasks.cancel_task(task_id))


@router.post("/{task_id}/retry", status_code=202)
async def retry_task(task_id: str):
    """Run a failed or cancelled task again as a new task on the same thread."""
    task = await tasks.retry_task(task_id)
    return {"task_id": task["task_id"], "thread_id": task["thread_id"], "status": task["status"]}


@ws_router.websocket("/ws/tasks/{task_id}")
async def task_websocket(websocket: WebSocket, task_id: str, last_event_id: int = 0):
    """Live task events. Replays events after `last_event_id`, then streams new
    ones and closes after the terminal event. Client messages:
    {"type": "cancel"} stops the task."""
    await websocket.accept()
    if not await resources.db.get_task(task_id):
        await websocket.close(code=4404, reason="Task not found")
        return

    async def receive_commands():
        """Handle client messages until the client disconnects or we close."""
        try:
            while True:
                try:
                    msg = json.loads(await websocket.receive_text())
                except json.JSONDecodeError:
                    continue  # ignore malformed messages
                if isinstance(msg, dict) and msg.get("type") == "cancel":
                    await tasks.cancel_task(task_id)
        except (WebSocketDisconnect, WebSocketDisconnected):
            pass  # client closed the socket, or we did

    receiver = asyncio.create_task(receive_commands())
    finished = False
    try:
        async for event in tasks.follow_events(task_id, last_event_id, KEEPALIVE_SECONDS):
            if receiver.done():  # client went away
                break
            if event is None:
                await websocket.send_json({"type": "ping"})
                continue
            await websocket.send_json(event)
        else:
            finished = True
    except (WebSocketDisconnect, WebSocketDisconnected):
        pass  # client went away mid-send
    finally:
        # Stop receiving before closing, so no receive runs on a closed socket.
        receiver.cancel()
        await asyncio.gather(receiver, return_exceptions=True)
        if finished and websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(code=1000)
            except (RuntimeError, WebSocketDisconnect):
                pass  # client closed at the same moment
