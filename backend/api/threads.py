"""
Thread & Conversation Management Router
========================================
Threads (conversations), their messages, their task history, and the agent's
virtual filesystem for the thread. All persisted: threads/messages/tasks in
the app database, files in the LangGraph checkpoint.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend import resources, tasks
from backend.agent import AgentConfigModel, get_agent

router = APIRouter(prefix="/api/threads", tags=["threads"])


class ThreadModel(BaseModel):
    """A thread as returned by the API."""

    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    active_task_id: str | None = None  # set while a task is queued or running


class CreateThreadRequest(BaseModel):
    """Body for creating a thread; the title defaults to "New Conversation"."""

    title: str | None = None


async def _thread_model(thread: dict, with_messages: bool) -> ThreadModel:
    """Build the API view of a thread, with or without its messages."""
    messages = await resources.db.list_messages(thread["id"])
    active = await resources.db.active_task_for_thread(thread["id"])
    return ThreadModel(
        id=thread["id"],
        title=thread["title"],
        created_at=thread["created_at"],
        updated_at=thread["updated_at"],
        message_count=len(messages),
        messages=messages if with_messages else [],
        active_task_id=active["task_id"] if active else None,
    )


async def _get_thread_or_404(thread_id: str) -> dict:
    """Return the thread, or raise HTTP 404 if it does not exist."""
    thread = await resources.db.get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


@router.get("", response_model=List[ThreadModel])
async def list_threads():
    """List conversation threads, most recently active first."""
    return [await _thread_model(t, with_messages=False) for t in await resources.db.list_threads()]


@router.post("", response_model=ThreadModel)
async def create_thread(req: CreateThreadRequest = CreateThreadRequest()):
    """Create a new conversation thread."""
    thread = await resources.db.create_thread(title=req.title)
    return await _thread_model(thread, with_messages=False)


@router.get("/{thread_id}", response_model=ThreadModel)
async def get_thread(thread_id: str):
    """A thread with its full message history."""
    thread = await resources.db.ensure_thread(thread_id)
    return await _thread_model(thread, with_messages=True)


@router.delete("/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete a thread, its messages, tasks, events and agent checkpoints."""
    await _get_thread_or_404(thread_id)
    active = await resources.db.active_task_for_thread(thread_id)
    if active:
        await tasks.cancel_task(active["task_id"])
    await resources.db.delete_thread(thread_id)
    await resources.checkpointer.adelete_thread(thread_id)
    return {"status": "deleted", "thread_id": thread_id}


@router.get("/{thread_id}/tasks")
async def list_thread_tasks(thread_id: str):
    """Task (execution) history of a thread, oldest first."""
    await _get_thread_or_404(thread_id)
    return [tasks.public_task(t) for t in await resources.db.list_tasks(thread_id)]


@router.get("/{thread_id}/files")
async def list_thread_files(thread_id: str):
    """Files in the agent's virtual filesystem for this thread (StateBackend),
    from the thread's current agent state. Skill seed files are left out."""
    await _get_thread_or_404(thread_id)
    # Checkpoints store only the channels each step changed; the compiled graph
    # rebuilds the full state. The graph is only used to read, never run.
    agent, _ = get_agent(AgentConfigModel(), resources.checkpointer, resources.store)
    state = await agent.aget_state({"configurable": {"thread_id": thread_id}})
    files = state.values.get("files") or {}
    result = []
    for path, data in sorted(files.items()):
        if path.startswith("/skills/") or path == "/projects/AGENTS.md":
            continue
        content = data.get("content", "") if isinstance(data, dict) else str(data)
        if isinstance(content, list):  # older deepagents stored a list of lines
            content = "\n".join(content)
        result.append({
            "path": path,
            "content": content,
            "size": len(content),
            "modified_at": data.get("modified_at") if isinstance(data, dict) else None,
        })
    return result
