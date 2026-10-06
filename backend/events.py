"""
Event Bus (Redis)
=================
- Task queue: the API pushes task ids onto a Redis list; workers pop them.
- Live events: the worker publishes each task event on a per-task channel;
  API transports (WebSocket, SSE, long-poll) subscribe to it.
- Cancellation: the API publishes task ids on a cancel channel; the worker
  running that task cancels it.

Redis only carries signals. Events are persisted in SQLite first, so a client
that missed a live message replays it from the database by event_id.
"""

import asyncio
import json
from collections.abc import AsyncIterator

import redis.asyncio as redis
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError

from backend.config import settings

QUEUE_KEY = "deepagents:task_queue"
DEQUEUE_BLOCK_SECONDS = 5
# Read timeout for every Redis reply. Must exceed DEQUEUE_BLOCK_SECONDS, and
# leaves headroom for an event loop that is briefly busy (e.g. under a debugger).
SOCKET_TIMEOUT_SECONDS = 30
REDIS_ERRORS = (RedisConnectionError, RedisTimeoutError)
CANCEL_CHANNEL = "deepagents:cancel"


def task_channel(task_id: str) -> str:
    """Redis pub/sub channel carrying one task's live events."""
    return f"deepagents:task:{task_id}"


class EventBus:
    """Redis client wrapper for the task queue, live events and cancel signals."""

    def __init__(self, url: str = settings.REDIS_URL):
        """Create the Redis client (connects lazily on first command)."""
        self.redis = redis.from_url(
            url,
            decode_responses=True,
            socket_timeout=SOCKET_TIMEOUT_SECONDS,
            socket_connect_timeout=5,
            socket_keepalive=True,
            health_check_interval=30,
            retry=Retry(ExponentialBackoff(cap=5, base=0.2), retries=3),
            retry_on_error=list(REDIS_ERRORS),
        )

    async def close(self) -> None:
        """Close the Redis connection pool."""
        await self.redis.aclose()

    async def ping(self) -> bool:
        """True if Redis answers, False on any error (used by health checks)."""
        try:
            return await self.redis.ping()
        except Exception:
            return False

    # ── Queue ──────────────────────────────────────────────────────────────
    async def enqueue(self, task_id: str) -> None:
        """Add a task id to the queue for a worker to pick up."""
        await self.redis.lpush(QUEUE_KEY, task_id)

    async def dequeue(self, timeout: float = DEQUEUE_BLOCK_SECONDS) -> str | None:
        """Wait up to `timeout` seconds for the next task id; None if the queue stayed empty."""
        item = await self.redis.brpop([QUEUE_KEY], timeout=timeout)
        return item[1] if item else None

    async def queued_ids(self) -> list[str]:
        """Task ids currently waiting in the queue (used by worker crash recovery)."""
        return await self.redis.lrange(QUEUE_KEY, 0, -1)

    # ── Live events ────────────────────────────────────────────────────────
    async def publish(self, event: dict) -> None:
        """Broadcast a stored event to everyone following its task."""
        await self.redis.publish(task_channel(event["task_id"]), json.dumps(event))

    async def subscribe(self, task_id: str) -> "Subscription":
        """Start receiving a task's live events."""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(task_channel(task_id))
        return Subscription(pubsub)

    # ── Cancellation ───────────────────────────────────────────────────────
    async def request_cancel(self, task_id: str) -> None:
        """Ask whichever worker runs `task_id` to cancel it."""
        await self.redis.publish(CANCEL_CHANNEL, task_id)

    async def cancel_requests(self) -> AsyncIterator[str]:
        """Yield task ids whose cancellation was requested. Runs forever,
        resubscribing if the Redis connection drops."""
        while True:
            pubsub = self.redis.pubsub()
            try:
                await pubsub.subscribe(CANCEL_CHANNEL)
                while True:
                    # Short reads (not listen()) so an idle channel never hits the socket timeout.
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=DEQUEUE_BLOCK_SECONDS)
                    if message:
                        yield message["data"]
            except REDIS_ERRORS:
                await asyncio.sleep(2)
            finally:
                await pubsub.aclose()


class Subscription:
    """A live subscription to one task's events."""

    def __init__(self, pubsub):
        """Wrap an already-subscribed Redis PubSub object."""
        self._pubsub = pubsub

    async def next_event(self, timeout: float) -> dict | None:
        """Return the next published event, or None if none arrives within `timeout`."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while (remaining := deadline - loop.time()) > 0:
            # get_message also returns None, immediately, when the message it read
            # was an ignored subscribe confirmation; keep waiting until the deadline.
            message = await self._pubsub.get_message(ignore_subscribe_messages=True, timeout=remaining)
            if message:
                return json.loads(message["data"])
        return None

    async def close(self) -> None:
        """Unsubscribe and release the connection."""
        await self._pubsub.aclose()
