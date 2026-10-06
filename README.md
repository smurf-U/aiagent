# Deep Agents Chatbot

A conversational chatbot built on [LangChain Deep Agents](https://github.com/langchain-ai/deepagents) showcasing planning (`write_todos`), the virtual filesystem, context engineering (`AGENTS.md`), skills, subagents (incl. structured output), swappable backends (State / Filesystem / Store), thread memory and human-in-the-loop approval.

Two UIs share the same agent features:

| UI | Stack | Entry point |
|----|-------|-------------|
| Production app | React/Vite → FastAPI → Redis queue → agent worker | `frontend/`, `backend/` |
| Dashboard | Streamlit (self-contained) | `streamlit_app.py` |

## Architecture

```text
React ──REST / WebSocket / SSE / polling──► FastAPI (API)
                                              │  POST /api/chat → task_id (returns at once)
                                              ▼
                                         Redis queue ──► Agent worker (python -m backend.worker)
                                              ▲               │ DeepAgents + LangGraph + Ollama
                                   live events│◄──────────────┤ publishes each event
                                              │               ▼
                                         SQLite (data/): threads, messages, tasks, task events,
                                                         LangGraph checkpoints, memory store
```

- **`thread_id`** is the conversation (messages, agent state, files). **`task_id`** is one execution on it.
- The agent runs in the worker, not in an HTTP request. Closing the browser, a dropped connection or an API restart doesn't stop it.
- Every event is stored in SQLite **before** it is published on Redis, with a per-task `event_id`. Any client can reconnect and replay from its last `event_id`.
- WebSocket and SSE are only transports. The task status in the database is the source of truth.

## Setup

```bash
uv sync
cp .env.example .env              # add the API keys you use
docker compose up -d redis        # task queue + event bus
cd frontend && npm install && cd ..
```

`.env` keys (all optional except for the providers you use): `OPENAI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `TAVILY_API_KEY` (web search).

Optional settings: `REDIS_URL` (default `redis://localhost:6379/0`), `DATA_DIR` (default `./data`), `WORKER_CONCURRENCY` (default 4), `TASK_POLL_WAIT_SECONDS` (default 20; keep below your API gateway timeout), `OLLAMA_BASE_URL`, `HOST`, `PORT`, `CORS_ORIGINS` (comma-separated).

Local models need [Ollama](https://ollama.com) running with e.g. `ollama pull qwen3:8b`.

## Run

Three processes: Redis, the API and the worker.

**Production-style (API serves the built frontend):**

```bash
docker compose up -d redis
cd frontend && npm run build && cd ..
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000   # terminal 1
uv run python -m backend.worker                              # terminal 2
# open http://localhost:8000  (API docs at /docs)
```

**Development (hot reload):**

```bash
docker compose up -d redis
uv run uvicorn backend.main:app --reload --port 8000   # terminal 1
uv run python -m backend.worker                        # terminal 2 (restart after agent code changes)
cd frontend && npm run dev                             # terminal 3 → http://localhost:5173
```

In VS Code, the **Full stack** launch configuration starts the API, worker and frontend together.

**Streamlit dashboard:**

```bash
uv run python main.py    # or: uv run streamlit run streamlit_app.py
```

## API

**Chat and tasks**

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/chat` | Queue a turn; returns `{task_id, thread_id, status: "queued"}` (202) |
| POST | `/api/chat/resume` | Answer a pending approval (`decisions`: approve / edit / reject) as a new task |
| GET | `/api/tasks/{id}` | Status (`queued`, `running`, `completed`, `failed`, `cancelled`, `awaiting_approval`), progress, timestamps, error |
| GET | `/api/tasks/{id}/result` | Final answer and files |
| POST | `/api/tasks/{id}/cancel` | Stop a queued or running task |
| POST | `/api/tasks/{id}/retry` | Run a failed or cancelled task again (new task, same thread) |
| WS | `/ws/tasks/{id}?last_event_id=N` | **Primary live transport.** Replays after N, streams new events, closes after the final one. Send `{"type": "cancel"}` to stop. |
| GET | `/api/tasks/{id}/stream` | Same events as SSE; resumes from `Last-Event-ID`; works with `EventSource` |
| GET | `/api/tasks/{id}/events?after=N` | Polling fallback; each request returns within `TASK_POLL_WAIT_SECONDS` |

Events share one envelope:

```json
{"event_id": 12, "task_id": "…", "thread_id": "…", "type": "tool_started", "timestamp": "…", "data": {…}}
```

Types: `task_queued`, `task_started`, `status`, `plan`, `subagent`, `tool_started`, `tool_completed`, `file_updated`, `message_delta`, `message_reset`, `approval_required`, then one of `task_completed`, `task_failed`, `task_cancelled`.

**Threads and workspace**

| Method | Path | Purpose |
|--------|------|---------|
| GET/POST | `/api/threads` | List / create threads |
| GET/DELETE | `/api/threads/{id}` | A thread with its messages and `active_task_id` / delete it |
| GET | `/api/threads/{id}/tasks` | Task history of a thread |
| GET | `/api/threads/{id}/files` | The agent's virtual files for the thread |
| GET/POST/DELETE | `/api/files` | Browse and edit files under `deep_agent/` |
| GET | `/api/skills` | Skill catalogue from `deep_agent/skills/` |
| GET/POST/DELETE | `/api/memory` | Cross-thread memory store items |
| GET | `/api/health`, `/api/models` | Health (incl. Redis), API key status, model list |

## Failure handling

- **Browser closes or the connection drops:** the task keeps running. Reopening the thread reconnects and replays missed events.
- **WebSocket unavailable:** the frontend falls back to polling after 3 failed connection attempts.
- **API restarts:** tasks in the worker are unaffected. Their history is in SQLite.
- **Worker restarts:** on startup it marks tasks it left `running` as `failed` (shown with a Retry button) and re-queues `queued` tasks missing from Redis.

## Limitations

- **Single worker process.** Use `WORKER_CONCURRENCY` for parallelism inside it. Its crash recovery assumes no other worker is running. Multiple workers would need per-worker heartbeats, so only a dead worker's tasks are failed.
- **SQLite on one machine.** The API and worker must share the `data/` directory. For several hosts, move the app database and LangGraph checkpointer/store to PostgreSQL.
- **No authentication or rate limiting yet.**
