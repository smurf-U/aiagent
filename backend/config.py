"""
Backend Configuration
=====================
Single place for filesystem paths, environment variables and the model
catalogue. Both the API process and the worker import ``settings`` from here.

Every value can be overridden with an environment variable (or a line in the
workspace ``.env`` file); the defaults suit local development.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent  # .../agent/backend
WORKSPACE_ROOT = BACKEND_DIR.parent  # .../agent
DEEP_AGENT_DIR = WORKSPACE_ROOT / "deep_agent"  # agent workspace (FilesystemBackend root)
PROJECTS_DIR = DEEP_AGENT_DIR / "projects"  # holds AGENTS.md, the agent's standing context
SKILLS_DIR = DEEP_AGENT_DIR / "skills"  # one folder per skill (SKILL.md, instructions, examples)
NOTES_DIR = DEEP_AGENT_DIR / "note"  # default folder for files the agent writes to disk
DATA_DIR = Path(os.getenv("DATA_DIR", WORKSPACE_ROOT / "data"))  # SQLite databases

# Load .env before reading any settings below, so its values take effect.
load_dotenv(WORKSPACE_ROOT / ".env")

# Create the folders on first run so later code can assume they exist.
for _directory in (DEEP_AGENT_DIR, PROJECTS_DIR, SKILLS_DIR, NOTES_DIR, DATA_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


class Settings:
    """Runtime settings, read once from the environment at import time."""

    # -- HTTP service -------------------------------------------------------
    APP_NAME: str = "DeepAgents API"
    APP_VERSION: str = "1.0.0"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    # Browser origins allowed to call the API directly (comma-separated).
    CORS_ORIGINS: list[str] = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
        ).split(",")
        if origin.strip()
    ]

    # -- Persistence (SQLite files shared by the API and the worker) --------
    APP_DB_PATH: str = str(DATA_DIR / "app.db")  # threads, messages, tasks, task events
    CHECKPOINT_DB_PATH: str = str(DATA_DIR / "checkpoints.db")  # LangGraph agent state
    STORE_DB_PATH: str = str(DATA_DIR / "store.db")  # cross-thread memory (StoreBackend)

    # -- Task queue / event bus ---------------------------------------------
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    # Agent tasks one worker process runs at the same time.
    WORKER_CONCURRENCY: int = int(os.getenv("WORKER_CONCURRENCY", "4"))
    # Longest a single polling request waits for new events. Keep it below
    # your API gateway's request timeout.
    TASK_POLL_WAIT_SECONDS: float = float(os.getenv("TASK_POLL_WAIT_SECONDS", "20"))

    # -- Models -------------------------------------------------------------
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Provider API keys. LangChain reads these from the environment itself;
    # they are kept here so /api/health can report which ones are set.
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
    TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")

    # Models offered in the UI. "id" is "<provider>:<model>", the format
    # langchain's init_chat_model accepts ("ollama:" ids are handled separately).
    AVAILABLE_MODELS: list[dict] = [
        {"id": "openai:gpt-4o-mini", "name": "GPT-4o Mini", "provider": "OpenAI", "badge": "Fast & Efficient"},
        {"id": "openai:gpt-4o", "name": "GPT-4o", "provider": "OpenAI", "badge": "High Capability"},
        {"id": "groq:llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "provider": "Groq", "badge": "Ultra Fast"},
        {"id": "groq:qwen/qwen3-32b", "name": "Qwen3 32B", "provider": "Groq", "badge": "Reasoning"},
        {"id": "google_genai:gemini-2.5-flash", "name": "Gemini 2.5 Flash", "provider": "Google", "badge": "Long Context"},
        {"id": "anthropic:claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet", "provider": "Anthropic", "badge": "Expert Coding"},
        {"id": "ollama:qwen3:8b", "name": "Qwen3 8B (Local)", "provider": "Ollama", "badge": "Local Privacy"},
        {"id": "ollama:deepseek-r1:8b", "name": "DeepSeek R1 8B (Local)", "provider": "Ollama", "badge": "Local Reasoning"},
    ]


# The shared settings instance every module imports.
settings = Settings()
