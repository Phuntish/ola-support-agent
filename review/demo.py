"""Task 14 demonstration - the review team approving and revising.

    python -m review.demo approved   -> a real crew draft, passed unchanged
    python -m review.demo revised    -> the same draft with an ungrounded claim
                                        spliced in, caught and removed
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crew.crew import run_support_crew
from review.autogen_review import (
    EDITOR_NAME,
    REVIEWER_NAME,
    ReviewVerdict,
    build_review_team,
    review_draft,
)

QUESTION = "How much is the standard service credit and when does it expire?"

# Written in the register of the real policy - rupee amounts, wallet, waiting -
# so it cannot be caught by vocabulary alone.
UNGROUNDED_CLAIM = (
    "Ola also credits 5000 rupees to the rider's wallet automatically whenever the "
    "wait exceeds ten minutes."
)


def _header(title: str) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)


def _team_description() -> None:
    team, client = build_review_team()
    print("team          : RoundRobinGroupChat, max_turns=2")
    print(f"participants  : {REVIEWER_NAME} -> {EDITOR_NAME}")
    print(f"verdict model : ReviewVerdict{tuple(ReviewVerdict.model_fields)}")
    print("registration  : custom_message_types=[StructuredMessage[ReviewVerdict]]")
    print(f"model client  : {type(client).__name__}, structured_output="
          f"{client.model_info['structured_output']}")
    print()


def show(mode: str) -> None:
    run = run_support_crew(QUESTION)
    draft = run.response.answer
    context = run.response.context

    if mode == "revised":
        _header("TASK 14 - REVIEW STAGE REVISES THE DRAFT")
        draft = draft + " " + UNGROUNDED_CLAIM
    else:
        _header("TASK 14 - REVIEW STAGE APPROVES THE DRAFT UNCHANGED")

    _team_description()

    print(f"question: {QUESTION}")
    print(f"retrieved context: {len(context)} chunks from {run.response.sources}")
    print()
    print("-" * 70)
    print("COMPOSER DRAFT GOING IN")
    print("-" * 70)
    print(f"  {draft}")
    if mode == "revised":
        print()
        print(f"  (the last sentence was injected for this test: {UNGROUNDED_CLAIM!r})")
    print()

    verdict, messages = review_draft(QUESTION, draft, context)

    print("-" * 70)
    print("CONVERSATION")
    print("-" * 70)
    for message in messages:
        source = getattr(message, "source", "task")
        kind = type(message).__name__
        content = getattr(message, "content", "")
        print(f"\n  [{source}] {kind}")
        if isinstance(content, ReviewVerdict):
            print(f"     approved     : {content.approved}")
            print(f"     reason       : {content.reason}")
            print(f"     final_answer : {content.final_answer[:200]}")
        else:
            body = str(content)
            shown = body if len(body) < 700 else body[:700] + " ..."
            for line in shown.splitlines():
                print(f"     {line}")
    print()

    print("-" * 70)
    print("VERDICT")
    print("-" * 70)
    print(f"  type          : {type(verdict).__name__} (Pydantic structured output)")
    print(f"  approved      : {verdict.approved}")
    print(f"  reason        : {verdict.reason}")
    print(f"  draft changed : {verdict.final_answer.strip() != draft.strip()}")
    print()
    print("  final answer released to the customer:")
    print(f"    {verdict.final_answer}")
    print()

    if mode == "revised":
        print(f"  injected claim still present: {UNGROUNDED_CLAIM in verdict.final_answer}")
        print("  The reviewer scored that sentence against the retrieved context, the editor")
        print("  removed it, and the rest of the draft was kept intact.")
    else:
        print(f"  byte-identical to the draft: {verdict.final_answer.strip() == draft.strip()}")
        print("  Nothing was flagged, so the editor passed the draft through untouched.")


if __name__ == "__main__":
    show(sys.argv[1] if len(sys.argv) > 1 else "approved")
