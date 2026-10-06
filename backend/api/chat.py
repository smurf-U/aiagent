"""
Chat API Router
===============
Starts agent turns as tasks. The request returns immediately with a task_id;
the worker runs the agent, and clients follow the task through
/ws/tasks/{task_id} (WebSocket), /api/tasks/{task_id}/stream (SSE) or
/api/tasks/{task_id}/events (polling). Turns paused for human approval are
continued through /api/chat/resume, which starts a new task on the thread.
"""

from typing import Any, Literal
from fastapi import APIRouter
from pydantic import BaseModel

from backend import resources, tasks
from backend.agent import AgentConfigModel

router = APIRouter(prefix="/api/chat", tags=["chat"])


class AgentSettings(BaseModel):
    """Agent options sent with every chat or resume request (see AgentConfigModel)."""

    thread_id: str
    model: str = "ollama:qwen3:8b"
    backend: str = "StateBackend"  # "StateBackend", "FilesystemBackend", "StoreBackend"
    use_agents_md: bool = True
    use_skills: bool = True
    use_subagents: bool = True
    require_approval: bool = False
    system_prompt: str | None = None

    def agent_settings(self) -> dict:
        """Just the agent-config fields, validated, as a dict to store with the task."""
        return AgentConfigModel(**self.model_dump(exclude={"thread_id", "message", "decisions"})).model_dump()


class ChatRequest(AgentSettings):
    """A new user message for a thread."""

    message: str


class Decision(BaseModel):
    """A human's answer to one pending tool call."""

    type: Literal["approve", "edit", "reject"]
    message: str | None = None  # reject: reason passed back to the agent
    edited_action: dict[str, Any] | None = None  # edit: {"name": ..., "args": {...}}


class ResumeRequest(AgentSettings):
    """Decisions for a turn paused for approval, one per pending action."""

    decisions: list[Decision]


class TaskCreated(BaseModel):
    """Response for a newly queued task."""

    task_id: str
    thread_id: str
    status: str


@router.post("", response_model=TaskCreated, status_code=202)
async def chat_endpoint(req: ChatRequest):
    """Queue an agent turn for a thread and return its task id."""
    await tasks.ensure_thread_idle(req.thread_id)
    await resources.db.add_message(req.thread_id, "user", req.message)
    task = await tasks.create_task(
        req.thread_id, "message", {"settings": req.agent_settings(), "message": req.message}
    )
    return TaskCreated(**task)


@router.post("/resume", response_model=TaskCreated, status_code=202)
async def resume_endpoint(req: ResumeRequest):
    """Continue a turn paused for human approval as a new task.

    Send one decision per pending action, in the order they were requested.
    The agent settings must match the ones used for the paused turn.
    """
    await tasks.ensure_thread_idle(req.thread_id)
    decisions = [d.model_dump(exclude_none=True) for d in req.decisions]
    await resources.db.resolve_pending_interrupt(req.thread_id, [d["type"] for d in decisions])
    task = await tasks.create_task(
        req.thread_id, "resume", {"settings": req.agent_settings(), "decisions": decisions}
    )
    return TaskCreated(**task)
