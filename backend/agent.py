"""
Deep Agent Factory and Event Stream
===================================
Builds DeepAgents (``create_deep_agent``) from a small config object and runs
them, translating LangGraph's raw stream into simple UI events.

Used by the worker (to run tasks) and by the API (only to read a thread's
agent state, e.g. its virtual files). Neither process creates its own
checkpointer or store here: both are passed in, see backend/resources.py.
"""

import asyncio
import json
from typing import Any, AsyncGenerator

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, StateBackend, StoreBackend
from deepagents.backends.utils import create_file_data
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from langgraph.types import Command
from pydantic import BaseModel, Field

from backend.config import DEEP_AGENT_DIR, PROJECTS_DIR, SKILLS_DIR, settings
from backend.tools.search import internet_search


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
class ResearchFindings(BaseModel):
    """Structured output the "structured-researcher" subagent must return."""

    summary: str = Field(description="Comprehensive summary of findings")
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0 based on source consensus"
    )
    sources: list[str] = Field(description="List of cited source URLs")
    key_takeaways: list[str] = Field(
        default_factory=list, description="Bullet-point list of primary takeaways"
    )


class AgentConfigModel(BaseModel):
    """Everything that changes how an agent is built. Two equal configs produce
    the same agent, which is what lets get_agent() cache by config."""

    model: str = "ollama:qwen3:8b"  # "<provider>:<model>", see settings.AVAILABLE_MODELS
    backend: str = "StateBackend"  # where the agent's files live: StateBackend | FilesystemBackend | StoreBackend
    use_agents_md: bool = True  # load deep_agent/projects/AGENTS.md as standing memory
    use_skills: bool = True  # expose deep_agent/skills/ to the agent
    use_subagents: bool = True  # add the research subagents
    require_approval: bool = False  # pause before APPROVAL_TOOLS for a human decision
    system_prompt: str | None = None  # None = DEFAULT_SYSTEM_PROMPT


# Tools that pause for human approval when require_approval is enabled.
APPROVAL_TOOLS = ("write_file", "edit_file", "task")

# Store namespace holding the agent's files under StoreBackend (shared by all threads).
MEMORY_NAMESPACE = ("memories",)

# Path of the standing-context file inside the agent's virtual filesystem.
AGENTS_MD_PATH = "/projects/AGENTS.md"


# ---------------------------------------------------------------------------
# 2. Seed files (AGENTS.md + skills)
# ---------------------------------------------------------------------------
def load_agents_md() -> str:
    """Return AGENTS.md from disk, or a short built-in default if it is missing."""
    path = PROJECTS_DIR / "AGENTS.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return (
        "# Deep Agent Context\n"
        "Plan your work using write_todos. Offload bulky context to virtual files. "
        "Use subagents for in-depth research."
    )


def load_skill_seed_files() -> dict[str, dict]:
    """Read every skill Markdown file into deepagents file records, keyed by
    virtual path (deep_agent/skills/aws/SKILL.md -> /skills/aws/SKILL.md)."""
    files = {}
    if SKILLS_DIR.exists():
        for file_path in SKILLS_DIR.rglob("*.md"):  # rglob: any depth below skills/
            rel_path = file_path.relative_to(SKILLS_DIR).as_posix()
            files[f"/skills/{rel_path}"] = create_file_data(file_path.read_text(encoding="utf-8"))
    return files


async def seed_store(store: BaseStore) -> int:
    """Copy AGENTS.md and skill files into the memory store, skipping keys that
    already exist so edits made through StoreBackend are never overwritten.
    Returns how many entries were added."""
    seeds = {AGENTS_MD_PATH: create_file_data(load_agents_md())}
    seeds.update(load_skill_seed_files())
    added = 0
    for key, value in seeds.items():
        if await store.aget(MEMORY_NAMESPACE, key) is None:
            await store.aput(MEMORY_NAMESPACE, key, value)
            added += 1
    return added


# ---------------------------------------------------------------------------
# 3. Models
# ---------------------------------------------------------------------------
def resolve_model(model_str: str):
    """Create a chat model from a "<provider>:<model>" id.

    Ollama ids are built directly so model names containing ":" (qwen3:8b)
    survive and the configured base URL is used; everything else goes through
    LangChain's init_chat_model.
    """
    if model_str.startswith("ollama:"):
        model_name = model_str.split(":", 1)[1]  # keep "qwen3:8b" intact
        return ChatOllama(model=model_name, base_url=settings.OLLAMA_BASE_URL, temperature=0)
    return init_chat_model(model_str)


# ---------------------------------------------------------------------------
# 4. Agent factory
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are a versatile, highly autonomous Deep Agent built on LangGraph.\n"
    "Core operating principles:\n"
    "1. Plan first: When a task requires 2+ steps, call `write_todos` and update statuses.\n"
    "2. Offload aggressively: Store large data, code drafts, and research notes in files.\n"
    "3. Use Skills: Check your /skills/ folder when asked about AWS, LangGraph, Python, or Report Writing.\n"
    "4. Delegate wisely: Delegate specialized research to your subagents.\n"
    "5. Always synthesize clear, helpful responses with cited sources."
)

# Subagents the main agent can delegate to with its "task" tool. Each runs in
# its own context window, so long research does not crowd the main conversation.
SUBAGENTS = [
    {
        "name": "research-agent",
        "description": (
            "Performs comprehensive deep-dive internet research with Tavily "
            "in an isolated context window to prevent parent context pollution."
        ),
        "system_prompt": (
            "You are a specialized research agent. Perform exhaustive research, "
            "synthesize findings clearly, and cite authoritative source URLs."
        ),
        "tools": [internet_search],
    },
    {
        "name": "structured-researcher",
        "description": (
            "Researches complex topics and strictly returns structured findings: "
            "summary, confidence score (0-1), list of source URLs, and key takeaways."
        ),
        "system_prompt": (
            "You are an analytical researcher. Investigate the topic and output "
            "strictly formatted structured research findings."
        ),
        "tools": [internet_search],
        "response_format": ResearchFindings,
    },
]

# Compiled agents by config. Building one compiles a graph and creates a model
# client, so it is done once per distinct config per process.
_agent_cache: dict[str, tuple[Any, dict]] = {}


def get_agent(cfg: AgentConfigModel, checkpointer: BaseCheckpointSaver, store: BaseStore):
    """Return ``(compiled_agent, seed_files)`` for a config, building it on first use.

    ``seed_files`` are the AGENTS.md/skill files to inject into a thread's state
    (StateBackend only; the other backends read them from disk or the store).
    The checkpointer and store are per-process singletons, so caching by
    config alone is safe.
    """
    cache_key = json.dumps(cfg.model_dump(), sort_keys=True)
    if cache_key in _agent_cache:
        return _agent_cache[cache_key]

    seed_files: dict[str, dict] = {}
    memory_paths = [AGENTS_MD_PATH] if cfg.use_agents_md else None
    extra_kwargs: dict[str, Any] = {}

    if cfg.backend == "StateBackend":
        # Files live in the thread's graph state; seed them with each message.
        backend = StateBackend()
        if cfg.use_agents_md:
            seed_files[AGENTS_MD_PATH] = create_file_data(load_agents_md())
        if cfg.use_skills:
            seed_files.update(load_skill_seed_files())
    elif cfg.backend == "FilesystemBackend":
        # Files are real files under deep_agent/; virtual_mode keeps paths inside it.
        backend = FilesystemBackend(root_dir=str(DEEP_AGENT_DIR), virtual_mode=True)
    else:  # StoreBackend
        # Files live in the shared store (seeded at startup, see seed_store).
        backend = StoreBackend(store=store, namespace=lambda runtime: MEMORY_NAMESPACE)
        extra_kwargs["store"] = store

    if cfg.use_subagents:
        extra_kwargs["subagents"] = SUBAGENTS
    if cfg.use_skills:
        extra_kwargs["skills"] = ["/skills/"]
    if memory_paths:
        extra_kwargs["memory"] = memory_paths
    if cfg.require_approval:
        extra_kwargs["interrupt_on"] = {name: True for name in APPROVAL_TOOLS}

    agent = create_deep_agent(
        model=resolve_model(cfg.model),
        tools=[internet_search],
        system_prompt=cfg.system_prompt or DEFAULT_SYSTEM_PROMPT,
        backend=backend,
        checkpointer=checkpointer,
        **extra_kwargs,
    )
    _agent_cache[cache_key] = (agent, seed_files)
    return agent, seed_files


# ---------------------------------------------------------------------------
# 5. Event stream
# ---------------------------------------------------------------------------
# Filesystem tools whose calls are summarised as "Filesystem: <tool> <path>".
FILESYSTEM_TOOLS = ("write_file", "read_file", "edit_file", "ls", "glob", "grep")


def extract_text(content) -> str:
    """Flatten message content to plain text. Providers return either a string
    or a list of content blocks ({"type": "text", "text": ...}, tool blocks, ...)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _event(type_: str, **data) -> dict:
    """Build one event in the shape the worker consumes."""
    return {"event": type_, "data": data}


def _tool_call_events(tool_name: str, tool_args: dict) -> list[dict]:
    """Translate one tool call from the model into UI events."""
    if tool_name == "write_todos":
        todos = tool_args.get("todos", [])
        return [
            _event("plan", todos=todos),
            _event("status", status=f"Updating execution plan ({len(todos)} steps)"),
        ]
    if tool_name == "task":
        name = tool_args.get("subagent_type", "subagent")
        return [
            _event("subagent", name=name, description=tool_args.get("description", "")),
            _event("status", status=f"Delegating task to {name}"),
        ]
    started = _event("tool_started", name=tool_name, args=tool_args)
    if tool_name == "internet_search":
        return [started, _event("status", status=f"Searching web for: {tool_args.get('query', '')}")]
    if tool_name in FILESYSTEM_TOOLS:
        path = tool_args.get("file_path") or tool_args.get("path") or ""
        return [started, _event("status", status=f"Filesystem: {tool_name} {path}")]
    return [started]


async def execute_agent_stream(
    message: str | None,
    thread_id: str,
    cfg: AgentConfigModel,
    *,
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
    resume: dict | None = None,
) -> AsyncGenerator[dict, None]:
    """Run one agent turn and yield its events as ``{"event": ..., "data": ...}``.

    Event types: status, plan, subagent, tool_started, tool_completed,
    file_updated, message_delta, message_reset, approval_required, and finally
    exactly one of "done" (answer + files) or "error".

    Pass ``message`` to start a new turn, or ``resume`` (``{"decisions": [...]}``)
    to continue a turn that paused for human approval. Either way the turn runs
    on the thread's saved state (``thread_id``).
    """
    # Building an agent (graph compile, model init, reading skill files) is
    # synchronous; do it in a thread so the event loop stays responsive.
    agent, seed_files = await asyncio.to_thread(get_agent, cfg, checkpointer, store)

    yield _event("status", status="Resuming after review…" if resume is not None else "Thinking…")

    if resume is not None:
        payload = Command(resume=resume)  # answers the pending interrupt
    else:
        payload = {"messages": [{"role": "user", "content": message}]}
        if seed_files and cfg.backend == "StateBackend":
            payload["files"] = seed_files

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}

    try:
        final_answer = ""
        written_files: dict[str, str] = {}
        interrupted = False

        # "messages" mode gives token chunks; "updates" gives each node's state
        # change (tool calls, tool results, files, interrupts).
        async for mode, chunk in agent.astream(payload, config=config, stream_mode=["updates", "messages"]):
            if mode == "messages":
                msg_chunk, metadata = chunk
                # Stream tokens from the main agent's model node only; subagents
                # run nested under the tools node, so their namespace contains "|".
                if (
                    metadata.get("langgraph_node") == "model"
                    and "|" not in metadata.get("langgraph_checkpoint_ns", "")
                    and getattr(msg_chunk, "type", "") == "AIMessageChunk"
                ):
                    token = extract_text(msg_chunk.content)
                    if token:
                        yield _event("message_delta", delta=token)
                continue

            for node_name, update in chunk.items():
                # Human-in-the-loop: the graph paused before an approval tool.
                if node_name == "__interrupt__":
                    request = update[0].value
                    # Round-trip through JSON so any non-serialisable values become strings.
                    interrupt_data = json.loads(json.dumps({
                        "action_requests": request.get("action_requests", []),
                        "review_configs": request.get("review_configs", []),
                    }, default=str))
                    interrupted = True
                    yield _event("approval_required", **interrupt_data)
                    continue

                if not isinstance(update, dict):  # middleware hooks may emit None
                    continue

                # Files the agent wrote this step (seed files are not news).
                for path, data in (update.get("files") or {}).items():
                    if path in seed_files:
                        continue
                    content = data.get("content", "") if isinstance(data, dict) else str(data)
                    written_files[path] = content
                    yield _event("file_updated", path=path, content=content[:2000])

                for msg in update.get("messages", []):
                    msg_type = getattr(msg, "type", "")
                    tool_calls = getattr(msg, "tool_calls", None)

                    if msg_type == "ai" and tool_calls:
                        # Only report tool calls from the model node: the approval
                        # middleware re-emits the same message when a turn resumes.
                        if node_name != "model":
                            continue
                        # Text streamed before a tool call was narration, not the answer.
                        yield _event("message_reset")
                        for call in tool_calls:
                            for event in _tool_call_events(call.get("name", "tool"), call.get("args", {})):
                                yield event

                    elif msg_type == "tool":
                        result = extract_text(msg.content)
                        yield _event("tool_completed", name=getattr(msg, "name", "tool"), result=result[:1500])

                    elif msg_type == "ai":
                        # An AI message without tool calls is the turn's answer.
                        text = extract_text(msg.content)
                        if text:
                            final_answer = text

        yield _event("done", answer=final_answer, files=written_files, interrupted=interrupted)

    except Exception as exc:  # model/provider/tool errors end the turn, not the worker
        yield _event("error", error=str(exc))
