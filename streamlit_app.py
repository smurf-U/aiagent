"""
Deep Agents Conversational Chatbot — Streamlit Application
===========================================================
A conversational chatbot built with `deepagents` and `LangGraph` that showcases
EVERY feature demonstrated in the deep agent notebooks:

1. Core Architecture (Notebook 1):
   - Planning with `write_todos` (pending, in_progress, completed tracking)
   - Virtual File System & Context Offloading (`write_file`, `read_file`, `edit_file`, `ls`, `glob`, `grep`)
   - Custom tools: Tavily web search (`internet_search`)
   - Custom system prompt & custom LLMs (OpenAI, Groq, Anthropic, Google GenAI, Ollama)

2. Context Engineering & Memory (Notebook 2):
   - Durable system instructions via `AGENTS.md` loaded using `memory=["/projects/AGENTS.md"]`
   - Conversational persistence using LangGraph `MemorySaver` checkpointer and `thread_id`
   - Agent Skills (`/skills/`): `aws`, `langgraph`, `python`, `report-writer`

3. Swappable Backends (Notebook 3):
   - `StateBackend`: Ephemeral in-state virtual filesystem per thread
   - `FilesystemBackend`: Real files persisted to disk inside `deep_agent/`
   - `StoreBackend`: Cross-thread persistent memory using LangGraph `InMemoryStore`

4. Subagents & Delegation (Notebook 4):
   - Isolated context delegation via the `task` tool
   - `research-agent`: In-depth web research subagent
   - `structured-researcher`: Structured output subagent enforcing Pydantic `ResearchFindings`

Run with:
    streamlit run streamlit_app.py
"""

import os
import uuid
from pathlib import Path
from typing import Literal

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path & Environment Setup
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = Path(__file__).resolve().parent
DEEP_AGENT_DIR = WORKSPACE_ROOT / "deep_agent"
PROJECTS_DIR = DEEP_AGENT_DIR / "projects"
SKILLS_DIR = DEEP_AGENT_DIR / "skills"

load_dotenv(WORKSPACE_ROOT / ".env")

# Ensure required folders exist
DEEP_AGENT_DIR.mkdir(parents=True, exist_ok=True)
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
SKILLS_DIR.mkdir(parents=True, exist_ok=True)

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, StateBackend, StoreBackend
from deepagents.backends.utils import create_file_data
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from tavily import TavilyClient

# ---------------------------------------------------------------------------
# 1. Custom Tools: Tavily Internet Search (Notebook 1 & 4)
# ---------------------------------------------------------------------------
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a live web search for current information, documentation, and news."""
    if not tavily_client:
        return "Error: TAVILY_API_KEY is not set in .env"
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


# ---------------------------------------------------------------------------
# 2. Structured Output Schema for Subagents (Notebook 4)
# ---------------------------------------------------------------------------
class ResearchFindings(BaseModel):
    """Structured findings from a research task."""

    summary: str = Field(description="Comprehensive summary of findings")
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0 based on source consensus"
    )
    sources: list[str] = Field(description="List of cited source URLs")
    key_takeaways: list[str] = Field(
        default_factory=list, description="Bullet-point list of primary takeaways"
    )


# ---------------------------------------------------------------------------
# 3. Context Engineering Helpers: AGENTS.md & Skills (Notebook 2)
# ---------------------------------------------------------------------------
def load_agents_md() -> str:
    """Read AGENTS.md from the projects directory."""
    path = PROJECTS_DIR / "AGENTS.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return (
        "# Deep Agent Context\n"
        "Plan your work using write_todos. Offload bulky context to virtual files. "
        "Use subagents for in-depth research."
    )


def load_skill_seed_files() -> dict:
    """Read all skill files (SKILL.md, instructions.md, examples.md) under deep_agent/skills/

    and convert them to virtual file data under the /skills/ prefix.
    """
    files = {}
    if SKILLS_DIR.exists():
        for file_path in SKILLS_DIR.rglob("*.md"):
            rel_path = file_path.relative_to(SKILLS_DIR).as_posix()
            virtual_path = f"/skills/{rel_path}"
            files[virtual_path] = create_file_data(
                file_path.read_text(encoding="utf-8")
            )
    return files


# ---------------------------------------------------------------------------
# 4. Agent Factory: Assembles all Deep Agent Features
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


def build_agent(cfg: dict):
    """Build and compile a Deep Agent instance configured from UI parameters."""
    seed_files = {}
    memory_paths = None
    store_instance = None

    # --- A. Backend Selection (Notebook 3) ---
    backend_choice = cfg["backend"]

    if backend_choice == "StateBackend (in-state, ephemeral per thread)":
        backend = StateBackend()
        # Seed files into in-state virtual filesystem for the thread
        if cfg["use_agents_md"]:
            seed_files["/projects/AGENTS.md"] = create_file_data(load_agents_md())
            memory_paths = ["/projects/AGENTS.md"]
        if cfg["use_skills"]:
            seed_files.update(load_skill_seed_files())

    elif backend_choice == "FilesystemBackend (real disk in deep_agent/)":
        backend = FilesystemBackend(root_dir=str(DEEP_AGENT_DIR), virtual_mode=True)
        if cfg["use_agents_md"]:
            memory_paths = ["/projects/AGENTS.md"]

    else:  # StoreBackend (cross-thread persistent memory)
        store = st.session_state.get("store")
        store_instance = store
        backend = StoreBackend(store=store, namespace=lambda rt: ("memories",))

        # Seed initial documents into store once per session
        if not st.session_state.get("store_seeded"):
            if cfg["use_agents_md"]:
                store.put(
                    ("memories",),
                    "/projects/AGENTS.md",
                    create_file_data(load_agents_md()),
                )
            if cfg["use_skills"]:
                for vpath, vdata in load_skill_seed_files().items():
                    store.put(("memories",), vpath, vdata)
            st.session_state["store_seeded"] = True

        if cfg["use_agents_md"]:
            memory_paths = ["/projects/AGENTS.md"]

    # --- B. Subagents Configuration (Notebook 4) ---
    subagents = []
    if cfg["use_subagents"]:
        subagents.append(
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
            }
        )
        subagents.append(
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
            }
        )

    # --- C. Assemble Deep Agent ---
    agent_kwargs = {
        "model": cfg["model"],
        "tools": [internet_search],
        "system_prompt": cfg["system_prompt"],
        "backend": backend,
        "checkpointer": st.session_state.get("checkpointer"),
    }

    if subagents:
        agent_kwargs["subagents"] = subagents
    if cfg["use_skills"]:
        agent_kwargs["skills"] = ["/skills/"]
    if memory_paths:
        agent_kwargs["memory"] = memory_paths
    if store_instance:
        agent_kwargs["store"] = store_instance

    agent = create_deep_agent(**agent_kwargs)
    return agent, seed_files


# ---------------------------------------------------------------------------
# 5. UI Rendering Helpers
# ---------------------------------------------------------------------------
def extract_text(content) -> str:
    """Extract plain string content from string or list of content blocks."""
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


def render_steps(messages):
    """Render intermediate execution artifacts: planning, tools, files, and subagents."""
    for msg in messages:
        msg_type = getattr(msg, "type", "")

        # AI Tool Calls
        if msg_type == "ai" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                name, args = tc.get("name", "tool"), tc.get("args", {})

                # 1. Planning Tool (write_todos)
                if name == "write_todos":
                    with st.expander(
                        "📋 **Planning Tool (`write_todos`)**", expanded=True
                    ):
                        todos = args.get("todos", [])
                        if not todos:
                            st.info("Todo list cleared / empty.")
                        for item in todos:
                            status = (
                                item.get("status")
                                if isinstance(item, dict)
                                else "pending"
                            )
                            content = (
                                item.get("content")
                                if isinstance(item, dict)
                                else str(item)
                            )
                            badge = {
                                "completed": "✅ **Completed**",
                                "in_progress": "🔄 **In Progress**",
                                "pending": "⬜ **Pending**",
                            }.get(status, "⬜ **Pending**")
                            st.markdown(f"- {badge}: {content}")

                # 2. Subagent Delegation (task)
                elif name == "task":
                    subagent_name = args.get("subagent_type", "subagent")
                    description = args.get(
                        "description", "Executing delegated task..."
                    )
                    with st.expander(
                        f"🤖 **Subagent Execution (`{subagent_name}`)**", expanded=True
                    ):
                        st.markdown(
                            f"**Delegation Task:**\n> {description.strip()}"
                        )

                # 3. Web Search Tool
                elif name == "internet_search":
                    query = args.get("query", "")
                    with st.expander(
                        f"🔍 **Internet Search (`{query}`)**", expanded=False
                    ):
                        st.json(args)

                # 4. File System Operations
                elif name in (
                    "write_file",
                    "read_file",
                    "edit_file",
                    "ls",
                    "glob",
                    "grep",
                ):
                    target = (
                        args.get("file_path")
                        or args.get("path")
                        or args.get("pattern")
                        or ""
                    )
                    with st.expander(
                        f"📁 **File System: `{name}` `{target}`**", expanded=False
                    ):
                        st.json(args)

                # Other Tools
                else:
                    with st.expander(f"🛠️ **Tool Call (`{name}`)**", expanded=False):
                        st.json(args)

        # Tool Execution Results
        elif msg_type == "tool":
            tool_name = getattr(msg, "name", "tool")
            content_str = extract_text(msg.content)
            display_text = (
                content_str[:1200] + " …[truncated]"
                if len(content_str) > 1200
                else content_str
            )
            with st.expander(
                f"↩️ **Tool Result (`{tool_name}`)**", expanded=False
            ):
                st.code(display_text, language="markdown")


def render_virtual_files(files_dict: dict):
    """Render virtual file tree and contents stored in agent state."""
    if not files_dict:
        return
    with st.expander(
        f"🗂️ **Virtual Files in State ({len(files_dict)})**", expanded=False
    ):
        for vpath, vdata in files_dict.items():
            content = (
                vdata.get("content", "") if isinstance(vdata, dict) else str(vdata)
            )
            st.markdown(f"📄 **`{vpath}`**")
            preview = (
                content[:1200] + " …[truncated]"
                if len(content) > 1200
                else content
            )
            st.code(preview, language="python" if vpath.endswith(".py") else "text")


# ---------------------------------------------------------------------------
# 6. Streamlit Page Configuration & Session State
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Deep Agents Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State Variables
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = MemorySaver()
if "store" not in st.session_state:
    st.session_state.store = InMemoryStore()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []  # List of (role, text, steps_messages, files)

# ---------------------------------------------------------------------------
# 7. Sidebar Controls & Feature Toggles
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🧠 Deep Agent Controls")

    # Model Selection
    model_options = [
        "openai:gpt-4o-mini",
        "openai:gpt-4o",
        "groq:llama-3.3-70b-versatile",
        "groq:qwen/qwen3-32b",
        "anthropic:claude-3-5-sonnet-20241022",
        "anthropic:claude-3-5-haiku-20241022",
        "google_genai:gemini-2.5-flash",
        "google_genai:gemini-2.5-pro",
        "custom",
    ]
    selected_model = st.selectbox(
        "Model Provider & Architecture",
        model_options,
        index=0,
        help="Notebook 1: Deep agents support any model initialized via langchain's init_chat_model.",
    )

    if selected_model == "custom":
        model_name = st.text_input(
            "Enter custom model string",
            value="openai:gpt-4o-mini",
            help="Format: 'provider:model_name', e.g., 'ollama:qwen3:8b' or 'groq:llama-3.3-70b-versatile'",
        )
    else:
        model_name = selected_model

    # Backend Selection (Notebook 3)
    st.subheader("💾 File & Memory Backend")
    backend_choice = st.radio(
        "Select Backend",
        [
            "StateBackend (in-state, ephemeral per thread)",
            "FilesystemBackend (real disk in deep_agent/)",
            "StoreBackend (cross-thread persistent store)",
        ],
        index=0,
        help=(
            "• StateBackend: Virtual filesystem stored inside LangGraph state for this thread only.\n"
            "• FilesystemBackend: Real files written to/read from deep_agent/ on disk.\n"
            "• StoreBackend: LangGraph BaseStore (InMemoryStore) shared across all threads."
        ),
    )

    st.subheader("🧩 Feature Flags")
    use_agents_md = st.checkbox(
        "Load AGENTS.md Context (`memory=`)",
        value=True,
        help="Notebook 2: Pre-loads durable project architecture and behavioral guidelines into the context window.",
    )
    use_skills = st.checkbox(
        "Enable Agent Skills (`/skills/`)",
        value=True,
        help="Notebook 2: Exposes domain expertise in AWS, LangGraph, Python, and Report Writing.",
    )
    use_subagents = st.checkbox(
        "Enable Subagents (`task` tool)",
        value=True,
        help="Notebook 4: Equips the agent with 'research-agent' (quarantined web search) and 'structured-researcher' (Pydantic schema).",
    )

    with st.expander("📝 Custom System Prompt", expanded=False):
        system_prompt = st.text_area(
            "System Prompt",
            DEFAULT_SYSTEM_PROMPT,
            height=140,
            help="Notebook 1: Custom project instructions layered over the core Deep Agent system prompt.",
        )

    st.divider()

    # Thread & Session Management (Notebook 2 & LangGraph Checkpointer)
    st.subheader("🧵 Thread & Memory Session")
    st.caption(f"Current `thread_id`: **`{st.session_state.thread_id[:8]}…`**")

    col1, col2 = st.columns(2)
    if col1.button(
        "🆕 New Thread",
        use_container_width=True,
        help="Generates a fresh thread_id. Thread memory resets, but StoreBackend cross-thread memories remain!",
    ):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()

    if col2.button(
        "🗑️ Reset All",
        use_container_width=True,
        help="Clears checkpointer, store, chat history, and starts completely fresh.",
    ):
        for key in (
            "checkpointer",
            "store",
            "store_seeded",
            "thread_id",
            "history",
            "cfg_key",
            "agent",
        ):
            st.session_state.pop(key, None)
        st.rerun()

    # API Key Health Check
    st.divider()
    st.subheader("🔑 API Key Status")
    keys_status = {
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
        "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY")),
        "TAVILY_API_KEY": bool(os.getenv("TAVILY_API_KEY")),
    }
    for k, status in keys_status.items():
        st.markdown(f"{'🟢' if status else '🔴'} `{k}`")

# ---------------------------------------------------------------------------
# 8. Agent Construction / Cache Management
# ---------------------------------------------------------------------------
config_payload = {
    "model": model_name,
    "backend": backend_choice,
    "use_agents_md": use_agents_md,
    "use_skills": use_skills,
    "use_subagents": use_subagents,
    "system_prompt": system_prompt,
}

config_key = str(sorted(config_payload.items()))
if st.session_state.get("cfg_key") != config_key:
    with st.spinner("🔄 Compiling Deep Agent graph..."):
        try:
            agent, seed_files = build_agent(config_payload)
            st.session_state.agent = agent
            st.session_state.seed_files = seed_files
            st.session_state.cfg_key = config_key
        except Exception as e:
            st.error(f"Error initializing Deep Agent: {e}")
            st.stop()

# ---------------------------------------------------------------------------
# 9. Main Chat Interface & Header
# ---------------------------------------------------------------------------
st.title("🧠 Deep Agents Conversational Chatbot")
st.markdown(
    "Experience every pillar of Deep Agents: **Planning** (`write_todos`), **File System Offloading**, "
    "**Context Engineering** (`AGENTS.md`), **Domain Skills** (`/skills/`), **Subagents** (standard & structured), "
    "**Swappable Backends**, and **Persistent Thread Memory**."
)

# Active Features Badge Bar
col_b1, col_b2, col_b3, col_b4 = st.columns(4)
with col_b1:
    st.info(f"**Model:** `{model_name}`")
with col_b2:
    st.info(f"**Backend:** `{backend_choice.split()[0]}`")
with col_b3:
    st.info(
        f"**Skills:** `{'Enabled (4)' if use_skills else 'Disabled'}` | **Subagents:** `{'Enabled (2)' if use_subagents else 'Disabled'}`"
    )
with col_b4:
    st.info(f"**Memory:** `Thread ({st.session_state.thread_id[:6]}…)`")

# Quick Demo Prompts to Test Every Feature
with st.expander("⚡ Quick Test Prompts (Click to copy or explore features)"):
    st.markdown("""
    - **📋 Planning (`write_todos`):**
      `"Plan a 4-step migration strategy for transitioning a monolithic Flask API to modern FastAPI microservices. Save the plan to /note/migration_plan.txt."`
    - **🗂️ Domain Skill (LangGraph / AWS / Python):**
      `"How do I build a LangGraph workflow with conditional routing and a MemorySaver checkpointer? Show a runnable code snippet."`
    - **🤖 Subagent Research:**
      `"Use your research subagent to investigate the latest features in LangChain Deep Agents and summarize how context offloading works."`
    - **📊 Structured Researcher Subagent:**
      `"Use your structured-researcher subagent to evaluate the current state of Small Language Models (SLMs) in production. Return findings with confidence score and sources."`
    - **💾 Context Offloading (Virtual / Disk Files):**
      `"Create a file /note/benchmarks.txt comparing Python async vs threading, then read it back and summarize it."`
    - **🧵 Thread Memory Test:**
      Turn 1: `"Remember that my name is Kaushal and I am building Deep Agents with LangChain."`
      Turn 2: `"What is my name and what project am I working on?"`
    """)

# Replay Existing Conversation History
for role, text, steps, files in st.session_state.history:
    with st.chat_message(role):
        if steps:
            render_steps(steps)
        st.markdown(text)
        if files:
            render_virtual_files(files)

# ---------------------------------------------------------------------------
# 10. User Prompt Processing & Agent Invocation
# ---------------------------------------------------------------------------
if user_prompt := st.chat_input(
    "Ask anything — research a topic, ask for code, query AWS/LangGraph skills, or plan tasks..."
):
    # Render and store user message
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.history.append(("user", user_prompt, None, None))

    # Prepare agent payload
    invoke_payload = {"messages": [{"role": "user", "content": user_prompt}]}

    # Seed files into in-state virtual filesystem if StateBackend is active
    if st.session_state.seed_files:
        invoke_payload["files"] = st.session_state.seed_files

    invoke_config = {
        "configurable": {"thread_id": st.session_state.thread_id},
        "recursion_limit": 100,
    }

    # Execute Deep Agent invocation
    with st.chat_message("assistant"):
        with st.spinner("🧠 Deep Agent thinking, planning, and delegating..."):
            try:
                result = st.session_state.agent.invoke(
                    invoke_payload, config=invoke_config
                )
            except Exception as exc:
                st.error(f"❌ Execution failed: {exc}")
                st.stop()

        # Isolate new messages from this interaction turn
        all_messages = result.get("messages", [])
        turn_start_idx = max(
            (
                i
                for i, m in enumerate(all_messages)
                if getattr(m, "type", "") == "human"
            ),
            default=0,
        )
        turn_messages = all_messages[turn_start_idx + 1 :]

        # Render intermediate planning, subagent, and tool steps
        render_steps(turn_messages)

        # Extract final AI response
        last_message = all_messages[-1] if all_messages else None
        final_answer = (
            extract_text(getattr(last_message, "content", ""))
            if last_message
            else "*(No response generated)*"
        )
        st.markdown(final_answer)

        # Extract and display new virtual files generated in state
        virtual_files = {}
        for p, d in result.get("files", {}).items():
            if p not in st.session_state.seed_files:
                virtual_files[p] = d
        if virtual_files:
            render_virtual_files(virtual_files)

    # Persist interaction into chat history
    st.session_state.history.append(
        ("assistant", final_answer, turn_messages, virtual_files)
    )
