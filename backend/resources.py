"""
Process Resources
=================
Opens the connections a process needs (app DB, Redis bus, LangGraph
checkpointer and store) and exposes them as module-level singletons.
Both the API (FastAPI lifespan) and the worker call open()/close().
"""

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore

from backend.config import settings
from backend.db import Database
from backend.events import EventBus

db: Database | None = None
bus: EventBus | None = None
checkpointer: AsyncSqliteSaver | None = None
store: AsyncSqliteStore | None = None

_extra_conns: list[aiosqlite.Connection] = []


async def _sqlite(path: str, **kwargs) -> aiosqlite.Connection:
    """Open an aiosqlite connection in WAL mode and track it for close()."""
    conn = await aiosqlite.connect(path, **kwargs)
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA busy_timeout=5000")
    _extra_conns.append(conn)
    return conn


async def open() -> None:
    """Open every shared resource for this process. Call once at startup."""
    global db, bus, checkpointer, store
    db = Database(settings.APP_DB_PATH)
    await db.connect()
    bus = EventBus(settings.REDIS_URL)

    checkpointer = AsyncSqliteSaver(await _sqlite(settings.CHECKPOINT_DB_PATH))
    await checkpointer.setup()
    # The store manages its own transactions (explicit BEGIN), so it needs an
    # autocommit connection, as AsyncSqliteStore.from_conn_string would open.
    store = AsyncSqliteStore(await _sqlite(settings.STORE_DB_PATH, isolation_level=None))
    await store.setup()


async def close() -> None:
    """Close everything open() opened. Safe to call after a partial open()."""
    if bus:
        await bus.close()
    if db:
        await db.close()
    for conn in _extra_conns:
        await conn.close()
    _extra_conns.clear()
