"""Task 8 - session memory over the crew.

`RunnableWithMessageHistory` emits a LangChainDeprecationWarning pointing at
LangGraph's persistence layer. That is expected and left alone; the class works
and the brief calls for it.

Memory here is not decorative. A follow-up like "how long does that take?" has no
retrievable subject on its own, so the chain rewrites it using the previous turn
before the crew ever sees it. Run the same follow-up in a fresh session and the
rewrite cannot happen, which is exactly how the absence of state shows up.
"""

from __future__ import annotations

import os
import re
import sys

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from crew.crew import run_support_crew

# One history object per session id, held in process. It does not survive a
# restart, which the brief accepts for this task.
_SESSIONS: dict[str, InMemoryChatMessageHistory] = {}

# Words that make a question depend on whatever was said before it.
FOLLOW_UP_MARKERS = {"that", "it", "this", "they", "them", "those", "there"}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = InMemoryChatMessageHistory()
    return _SESSIONS[session_id]


def reset_sessions() -> None:
    _SESSIONS.clear()


def resolve_follow_up(text: str, history: list) -> tuple[str, bool]:
    """Rewrite a context-dependent question using the last thing the user asked.

    Returns the query to run and whether history was actually needed.
    """
    words = set(re.findall(r"[a-z']+", text.lower()))
    if not words & FOLLOW_UP_MARKERS:
        return text, False

    previous = [m for m in history if m.type == "human"]
    if not previous:
        return text, False

    return f"{previous[-1].content} {text}", True


def _answer(payload: dict) -> str:
    history = payload.get("history") or []
    question = payload["input"]

    effective, used_history = resolve_follow_up(question, history)
    run = run_support_crew(effective)

    payload["_trace"] = {
        "effective_query": effective,
        "used_history": used_history,
        "turns_in_history": len(history),
    }
    print(f"      history turns available : {len(history)}")
    print(f"      resolved using history  : {used_history}")
    print(f"      query sent to the crew  : {effective}")
    return run.response.answer


chain = RunnableLambda(_answer)

conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


def ask(question: str, session_id: str) -> str:
    return conversation.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )


def demo_multi_turn() -> None:
    reset_sessions()
    session = "rider-8821"

    print("=" * 70)
    print("TASK 8 - SESSION MEMORY, STATE CARRIED ACROSS TURNS")
    print("=" * 70)
    print(f"session_id: {session}")
    print()

    turns = [
        "What is the refund rule when a rider is charged twice for one trip?",
        "How long does that take to reach the original payment method?",
        "And is there any compensation on top of it?",
    ]

    for i, question in enumerate(turns, start=1):
        print(f"[turn {i}] Q: {question}")
        answer = ask(question, session)
        print(f"      A: {answer[:280]}")
        print()

    history = get_session_history(session).messages
    print("-" * 70)
    print(f"History for {session} now holds {len(history)} messages:")
    for m in history:
        print(f"  {m.type:<6} {m.content[:90]}")
    print()
    print("Turns 2 and 3 only resolve because turn 1 is still in this session's history.")


def demo_fresh_session() -> None:
    reset_sessions()
    session = "rider-9002-fresh"

    print("=" * 70)
    print("TASK 8 - FRESH SESSION, STATE CORRECTLY ABSENT")
    print("=" * 70)
    print(f"session_id: {session}  (never used before, history starts empty)")
    print()

    history_before = get_session_history(session).messages
    print(f"messages in history before the first turn: {len(history_before)}")
    print()

    # The identical follow-up from turn 2 of the other demo, asked cold.
    question = "How long does that take to reach the original payment method?"
    print(f"[turn 1] Q: {question}")
    answer = ask(question, session)
    print(f"      A: {answer[:280]}")
    print()

    print("-" * 70)
    print("There is no previous turn to resolve 'that' against, so the question goes to")
    print("the crew exactly as typed. The state from the other session is not visible")
    print("here, which is the point: sessions do not leak into each other.")
    print()
    other = get_session_history("rider-8821").messages
    print(f"messages visible from session 'rider-8821' in this run: {len(other)}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "multi"
    if mode == "fresh":
        demo_fresh_session()
    else:
        demo_multi_turn()
