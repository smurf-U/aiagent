"""
Agent Worker
============
Runs DeepAgents tasks outside the API process:

    uv run python -m backend.worker

Pops task ids from the Redis queue, runs the agent, stores every event in
SQLite (the source of truth) and publishes it on Redis for live transports.
Listens for cancel requests and cancels the matching running task.

On startup it recovers from a previous crash: tasks left "running" are marked
failed (the user can retry them) and "queued" tasks missing from the Redis
queue are re-enqueued. Run a single worker process; increase
WORKER_CONCURRENCY for parallelism within it.
"""

import asyncio
import logging
import signal

from backend import resources
from backend.agent import AgentConfigModel, execute_agent_stream, seed_store
from backend.config import settings
from backend.events import REDIS_ERRORS

log = logging.getLogger("deepagents.worker")

RESULT_FILE_LIMIT = 20_000  # characters kept per file in the task result


class TaskRunner:
    """Runs tasks popped from the queue, up to `concurrency` at a time."""

    def __init__(self, concurrency: int):
        """Create the concurrency limiter and the registry of running tasks."""
        self.slots = asyncio.Semaphore(concurrency)
        self.running: dict[str, asyncio.Task] = {}

    async def emit(self, task: dict, type_: str, data: dict | None = None) -> dict:
        """Persist an event, then publish it. Persisting first means a client
        that subscribes late can always replay it from the database."""
        event = await resources.db.append_event(task["task_id"], task["thread_id"], type_, data or {})
        try:
            await resources.bus.publish(event)
        except REDIS_ERRORS as exc:
            # The event is already stored; live clients pick it up from the
            # database on their next idle check, so the task can carry on.
            log.warning("publish failed for task %s event %s: %s", task["task_id"], event["event_id"], exc)
        return event

    async def run(self, task_id: str) -> None:
        """Claim a task and execute it, always freeing its concurrency slot.

        A task that another worker already claimed, or that was cancelled while
        queued, is skipped.
        """
        try:
            if not await resources.db.claim_task(task_id):
                log.info("task %s already claimed or cancelled; skipping", task_id)
                return
            task = await resources.db.get_task(task_id)
            log.info("task %s started (thread %s, %s)", task_id, task["thread_id"], task["kind"])
            await self._execute(task)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("task %s crashed in the worker", task_id)
        finally:
            self.running.pop(task_id, None)
            self.slots.release()

    async def _execute(self, task: dict) -> None:
        """Run the agent for a claimed task and record its outcome.

        Forwards every agent event (stored, then published), collects the steps,
        answer and files for the final assistant message, and finishes the task as
        completed, awaiting_approval, failed or cancelled.
        """
        payload = task["payload"]
        cfg = AgentConfigModel(**payload["settings"])
        if task["kind"] == "resume":
            message, resume = None, {"decisions": payload["decisions"]}
        else:
            message, resume = payload["message"], None

        steps: list[dict] = []
        answer = ""
        files: dict[str, str] = {}
        interrupt = None
        error = None

        await self.emit(task, "task_started", {"kind": task["kind"]})
        try:
            async for event in execute_agent_stream(
                message,
                task["thread_id"],
                cfg,
                checkpointer=resources.checkpointer,
                store=resources.store,
                resume=resume,
            ):
                type_, data = event["event"], event.get("data", {})
                if type_ == "done":
                    answer = data.get("answer") or answer
                    files.update(data.get("files") or {})
                    continue
                if type_ == "error":
                    error = data.get("error", "Unknown error")
                    continue

                await self.emit(task, type_, data)
                if type_ in ("plan", "subagent", "tool_started", "tool_completed", "status"):
                    steps.append({"type": type_, "data": data})
                if type_ == "plan":
                    todos = data.get("todos") or []
                    if todos:
                        done = sum(1 for t in todos if t.get("status") == "completed")
                        await resources.db.set_progress(task["task_id"], int(100 * done / len(todos)))
                elif type_ == "message_delta":
                    answer += data.get("delta", "")
                elif type_ == "message_reset":
                    answer = ""
                elif type_ == "file_updated":
                    files[data["path"]] = data.get("content", "")
                elif type_ == "approval_required":
                    interrupt = data
        except asyncio.CancelledError:
            await self._finish(task, "cancelled", answer or "⏹ Stopped.", steps, files)
            await self.emit(task, "task_cancelled", {})
            log.info("task %s cancelled", task["task_id"])
            return
        except Exception as exc:  # agent code raised outside its own handler
            error = str(exc)

        if error:
            await self._finish(task, "failed", f"❌ Error: {error}", steps, files, error=error)
            await self.emit(task, "task_failed", {"error": error})
            log.warning("task %s failed: %s", task["task_id"], error)
        elif interrupt:
            content = answer or "⏸ Waiting for your approval."
            await self._finish(task, "awaiting_approval", content, steps, files, interrupt=interrupt)
            await self.emit(task, "task_completed", {"answer": content, "files": files, "interrupted": True})
        else:
            content = answer or "Task completed."
            await self._finish(task, "completed", content, steps, files)
            await self.emit(task, "task_completed", {"answer": content, "files": files, "interrupted": False})
            log.info("task %s completed", task["task_id"])

    async def _finish(self, task, status, content, steps, files, *, error=None, interrupt=None):
        # Record the assistant message before the terminal event goes out, so a
        # client that reloads the thread on that event sees the final message.
        """Store the assistant message and mark the task finished.

        Runs before the terminal event is emitted, so a client that reloads the
        thread on that event already sees the final message.
        """
        files = {path: text[:RESULT_FILE_LIMIT] for path, text in files.items()}
        await resources.db.add_message(
            task["thread_id"], "assistant", content,
            steps=steps, files=files, interrupt=interrupt,
            task_id=task["task_id"], status=status,
        )
        await resources.db.finish_task(
            task["task_id"], status, result={"answer": content, "files": files}, error=error
        )

    async def listen_for_cancels(self) -> None:
        """Cancel running tasks when the API asks (runs for the worker's lifetime)."""
        async for task_id in resources.bus.cancel_requests():
            running = self.running.get(task_id)
            if running:
                log.info("cancelling task %s", task_id)
                running.cancel()

    async def recover(self) -> None:
        """Clean up after a previous worker that stopped mid-task."""
        for task in await resources.db.tasks_with_status("running"):
            error = "The worker restarted while this task was running."
            await resources.db.add_message(
                task["thread_id"], "assistant", f"❌ Error: {error}",
                task_id=task["task_id"], status="failed",
            )
            await resources.db.finish_task(task["task_id"], "failed", error=error)
            await self.emit(task, "task_failed", {"error": error})
            log.warning("recovered interrupted task %s as failed", task["task_id"])

        in_queue = set(await resources.bus.queued_ids())
        for task in await resources.db.tasks_with_status("queued"):
            if task["task_id"] not in in_queue:
                await resources.bus.enqueue(task["task_id"])
                log.info("re-enqueued task %s", task["task_id"])


async def main() -> None:
    """Worker entry point: open resources, serve until stopped, always close them."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    await resources.open()
    try:
        await serve()
    finally:
        # Always close: open SQLite connections run in non-daemon threads and
        # would keep a crashed worker process alive.
        await resources.close()


async def serve() -> None:
    """Prepare the worker (Redis check, store seeding, crash recovery) and run the queue loop.

    Stops on SIGINT/SIGTERM, then cancels running tasks and waits for them to
    record their outcome.
    """
    if not await resources.bus.ping():
        raise SystemExit(f"Cannot reach Redis at {settings.REDIS_URL}. Start it with: docker compose up -d redis")

    added = await seed_store(resources.store)
    if added:
        log.info("seeded %d memory store entries", added)

    runner = TaskRunner(settings.WORKER_CONCURRENCY)
    await runner.recover()
    cancel_listener = asyncio.create_task(runner.listen_for_cancels())

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    log.info("worker ready (concurrency=%d, redis=%s)", settings.WORKER_CONCURRENCY, settings.REDIS_URL)
    redis_down_since = None
    try:
        while not stop.is_set():
            await runner.slots.acquire()
            try:
                task_id = await resources.bus.dequeue()
            except REDIS_ERRORS as exc:
                # Redis restarting or a stalled read: keep the worker (and its
                # running tasks) alive and try again.
                runner.slots.release()
                if redis_down_since is None:
                    redis_down_since = asyncio.get_running_loop().time()
                    log.warning("Redis unavailable (%s); retrying", exc)
                await asyncio.sleep(2)
                continue
            if redis_down_since is not None:
                log.info("Redis reachable again")
                redis_down_since = None
            if task_id is None:
                runner.slots.release()
                continue
            runner.running[task_id] = asyncio.create_task(runner.run(task_id))
    finally:
        # Runs on normal shutdown and on any crash, so tasks finish their
        # bookkeeping while the database is still open.
        log.info("shutting down; cancelling %d running task(s)", len(runner.running))
        for running in list(runner.running.values()):
            running.cancel()
        await asyncio.gather(*runner.running.values(), return_exceptions=True)
        cancel_listener.cancel()


if __name__ == "__main__":
    asyncio.run(main())
