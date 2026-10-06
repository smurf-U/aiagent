"""
Human-in-the-loop demo (terminal).

The agent pauses before write_file / edit_file and asks you to approve,
reject, or edit the tool call.

    uv run python hitl_demo.py
    uv run python hitl_demo.py "Create /note/shopping.txt with: milk, eggs, bread"
"""

import json
import sys

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from backend.agent import resolve_model

MODEL = "ollama:qwen3:8b"
DEFAULT_QUERY = "Create a file /note/todo.txt with three tasks for learning LangGraph."

agent = create_deep_agent(
    model=resolve_model(MODEL),
    system_prompt="You are a helpful assistant. Use your file tools when asked to create files.",
    interrupt_on={"write_file": True, "edit_file": True},
    checkpointer=MemorySaver(),  # required: the paused state is saved here
)


def ask_decision(action: dict) -> dict:
    """Show one pending tool call and ask the user to approve, reject or edit it.

    Returns the decision in the format HumanInTheLoopMiddleware expects.
    """
    print(f"\n⏸  Agent wants to call: {action['name']}")
    print(json.dumps(action["args"], indent=2))
    choice = input("[a]pprove / [r]eject / [e]dit args? ").strip().lower()
    if choice.startswith("r"):
        reason = input("Reason (sent to the agent): ").strip() or "Rejected by user."
        return {"type": "reject", "message": reason}
    if choice.startswith("e"):
        new_args = json.loads(input("New args as JSON: "))
        return {"type": "edit", "edited_action": {"name": action["name"], "args": new_args}}
    return {"type": "approve"}


def main():
    """Run one query, answering every approval request in the terminal."""
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    config = {"configurable": {"thread_id": "hitl-demo"}}
    print(f"👤 {query}")

    result = agent.invoke({"messages": [{"role": "user", "content": query}]}, config=config)

    # Each interrupt pauses the graph; resume it with your decisions.
    while result.get("__interrupt__"):
        request = result["__interrupt__"][0].value
        decisions = [ask_decision(a) for a in request["action_requests"]]
        result = agent.invoke(Command(resume={"decisions": decisions}), config=config)

    print(f"\n🤖 {result['messages'][-1].content}")
    files = result.get("files", {})
    for path, data in files.items():
        print(f"\n📄 {path}\n{data.get('content', '')}")


if __name__ == "__main__":
    main()
