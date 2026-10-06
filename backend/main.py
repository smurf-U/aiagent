"""
FastAPI Application Entry Point
================================
Mounts all API routers, configures CORS, and serves the frontend
static files. Runs with uvicorn.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.config import WORKSPACE_ROOT, settings
from backend.api import (
    chat_router,
    threads_router,
    files_router,
    skills_router,
    memory_router,
    tasks_router,
    tasks_ws_router,
)
from backend import resources
from backend.agent import seed_store

# ---------------------------------------------------------------------------
# Lifespan: open the database, Redis bus, checkpointer and store
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open shared resources at startup and close them at shutdown."""
    await resources.open()
    await seed_store(resources.store)
    yield
    await resources.close()


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------
app = FastAPI(
    lifespan=lifespan,
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production FastAPI backend for the Deep Agents conversational chatbot. "
        "Implements planning, virtual filesystem, context engineering, skills, "
        "subagents, swappable backends, and persistent thread memory — all exposed "
        "as REST + Server-Sent Events endpoints."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# API Routers
# ---------------------------------------------------------------------------
app.include_router(chat_router)
app.include_router(threads_router)
app.include_router(files_router)
app.include_router(skills_router)
app.include_router(memory_router)
app.include_router(tasks_router)
app.include_router(tasks_ws_router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    """Liveness check plus Redis reachability and which provider API keys are set."""
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "redis": await resources.bus.ping(),
        "available_models": [m["id"] for m in settings.AVAILABLE_MODELS],
        "api_keys": {
            name: bool(getattr(settings, name))
            for name in (
                "OPENAI_API_KEY",
                "GROQ_API_KEY",
                "ANTHROPIC_API_KEY",
                "GEMINI_API_KEY",
                "TAVILY_API_KEY",
            )
        },
    }


@app.get("/api/models")
async def list_models():
    """Return list of supported models with metadata."""
    return settings.AVAILABLE_MODELS


# ---------------------------------------------------------------------------
# Serve React frontend (Vite build at frontend/dist)
# ---------------------------------------------------------------------------
FRONTEND_DIST = WORKSPACE_ROOT / "frontend" / "dist"

# Mount static assets only if build exists (must be registered before the SPA catch-all)
if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend(full_path: str):
    """Serve the Vite React build for all non-API routes (SPA fallback)."""
    if full_path.startswith(("api/", "ws/")):
        raise HTTPException(status_code=404, detail="Not found")
    if not FRONTEND_DIST.exists():
        return {"message": "Frontend not built yet. Run: cd frontend && npm install && npm run build"}
    file_path = (FRONTEND_DIST / full_path).resolve()
    if file_path.is_file() and file_path.is_relative_to(FRONTEND_DIST.resolve()):
        return FileResponse(str(file_path))
    return FileResponse(str(FRONTEND_DIST / "index.html"))


# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info",
    )
