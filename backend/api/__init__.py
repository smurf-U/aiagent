"""HTTP and WebSocket routers, mounted by backend/main.py."""

from backend.api.chat import router as chat_router
from backend.api.threads import router as threads_router
from backend.api.files import router as files_router
from backend.api.skills import router as skills_router
from backend.api.memory import router as memory_router
from backend.api.tasks import router as tasks_router, ws_router as tasks_ws_router

__all__ = [
    "chat_router",
    "threads_router",
    "files_router",
    "skills_router",
    "memory_router",
    "tasks_router",
    "tasks_ws_router",
]
