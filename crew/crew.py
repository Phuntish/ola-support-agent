"""Task 7 - assemble and run the crew, then validate what comes out (Task 9).

The lookup task is only added when the question actually names a ticket. A crew
that always runs every agent regardless of the question does more work, makes
noisier transcripts, and gives the Lookup Agent a record_id it does not have.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
# CrewAI 1.9 asks an interactive yes/no about execution tracing on first run,
# which would hang an unattended run_all.py. Turn it off before importing.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai import Crew, Process, Task

from crew.agents import COMPOSER_AGENT, LOOKUP_AGENT, RETRIEVAL_AGENT, build_agents
from crew.schema import SupportResponse, validate_crew_output
from llm.mock_llm import get_llm

RECORD_ID_PATTERN = re.compile(r"\bOLA-\d{4}\b", re.IGNORECASE)


@dataclass
class CrewRun:
    question: str
    response: SupportResponse
    raw: str
    llm_calls: int
    tools_used: list[str] = field(default_factory=list)


def mentions_ticket(question: str) -> bool:
    return RECORD_ID_PATTERN.search(question) is not None


def build_crew(question: str, llm=None) -> tuple[Crew, Any]:
    llm = llm or get_llm()

    # Task 15: tools come from the least-autonomy registry, which refuses any
    # pairing it has not declared. Importing a tool directly here would bypass it,
    # so nothing in this module does.
    from governance.least_autonomy import tools_for

    agents = build_agents(
        llm=llm,
        retrieval_tools=tools_for(RETRIEVAL_AGENT),
        lookup_tools=tools_for(LOOKUP_AGENT),
    )

    retrieval_task = Task(
        description=f"{question}",
        expected_output="The Ola support policy that answers the question, as JSON from the rag_lookup tool.",
        agent=agents[RETRIEVAL_AGENT],
    )

    tasks = [retrieval_task]
    if mentions_ticket(question):
        tasks.append(
            Task(
                description=f"{question}",
                expected_output="The ticket record as JSON from the check_support_ticket_status tool.",
                agent=agents[LOOKUP_AGENT],
            )
        )

    tasks.append(
        Task(
            description=(
                f"Write the reply to this customer question: {question}\n"
                "Use only the policy text and ticket details supplied to you."
            ),
            expected_output=(
                "A JSON object with keys answer, sources, retrieval_similarity, refused and ticket."
            ),
            agent=agents[COMPOSER_AGENT],
            context=tasks[:],
        )
    )

    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )
    return crew, llm


def run_guarded(question: str, llm=None) -> CrewRun:
    """The full request path: input guardrails, crew, output guardrail.

    This is what the API calls. Order matters - injection is checked before
    anything else runs, and PII is masked before the text reaches either the
    agents or the logger.
    """
    from governance.budget import BudgetExceeded, enforce_budget
    from guardrails.groundedness import REFUSAL_TEXT, check_groundedness
    from guardrails.injection import BLOCK_MESSAGE, detect_injection
    from guardrails.pii import mask_pii

    triggered: list[str] = []

    # Task 15, runtime layer. First thing checked, because rejecting a request
    # after the crew has run has already spent what the cap exists to protect.
    try:
        enforce_budget(question)
    except BudgetExceeded as exc:
        return CrewRun(
            question=question,
            response=SupportResponse(
                query=question,
                answer=str(exc),
                refused=True,
                retrieval_similarity=0.0,
                guardrails_triggered=["budget:request_too_large"],
            ),
            raw="",
            llm_calls=0,
        )

    verdict = detect_injection(question)
    if verdict.blocked:
        return CrewRun(
            question=question,
            response=SupportResponse(
                query=question,
                answer=BLOCK_MESSAGE,
                refused=True,
                retrieval_similarity=0.0,
                guardrails_triggered=[f"injection:{label}" for label in verdict.labels],
            ),
            raw="",
            llm_calls=0,
        )

    masked = mask_pii(question)
    if masked.fired:
        triggered.append("pii:phone_masked")

    run = run_support_crew(masked.masked, llm=llm)
    response = run.response

    if not response.refused and response.context:
        grounding = check_groundedness(response.answer, response.context)
        if not grounding.grounded:
            triggered.append("groundedness:unsupported_sentence")
            response = response.model_copy(
                update={"answer": REFUSAL_TEXT, "refused": True, "sources": []}
            )

    run.response = response.model_copy(update={"guardrails_triggered": triggered})
    return run


def run_support_crew(question: str, llm=None) -> CrewRun:
    """Run the crew end to end and validate the result against SupportResponse."""
    crew, llm = build_crew(question, llm=llm)
    result = crew.kickoff()
    raw = result.raw

    response = validate_crew_output(raw, query=question)
    return CrewRun(
        question=question,
        response=response,
        raw=raw,
        llm_calls=getattr(llm, "call_count", 0),
        tools_used=[c["tool"] for c in getattr(llm, "tool_calls", [])],
    )


DEMO_QUERIES = [
    ("RAG tool path", "What is the refund rule when a rider is charged twice for one trip?"),
    ("Lookup tool path", "What is the status of ticket OLA-0006 and should we escalate it?"),
    ("Both tools", "Ticket OLA-0005 has been open a while - what does the escalation policy say?"),
]


def main() -> None:
    from crew.agents import COMPOSER_AGENT, LOOKUP_AGENT, RETRIEVAL_AGENT

    print("=" * 70)
    print("TASK 7 - CREWAI CREW WITH TOOL USE")
    print("=" * 70)
    print("agents:")
    print(f"  1. {RETRIEVAL_AGENT:<32} tool: rag_lookup")
    print(f"  2. {LOOKUP_AGENT:<32} tool: check_support_ticket_status")
    print(f"  3. {COMPOSER_AGENT:<32} tools: none (works only from the other two)")
    print()
    print("run via Crew.kickoff(), process=sequential, MOCK_LLM=1")
    print()

    for label, question in DEMO_QUERIES:
        print("-" * 70)
        print(f"{label}: {question}")
        print("-" * 70)
        run = run_support_crew(question)

        print(f"  tools actually invoked : {run.tools_used}")
        print(f"  mock LLM calls         : {run.llm_calls}")
        print(f"  schema validation      : {type(run.response).__name__} OK (Task 9)")
        print(f"  sources                : {run.response.sources}")
        print(f"  retrieval similarity   : {run.response.retrieval_similarity}")
        if run.response.ticket is not None:
            t = run.response.ticket
            print(f"  ticket                 : {t.record_id} status={t.status} "
                  f"score={t.escalation_score} escalate={t.escalate_recommended}")
        else:
            print("  ticket                 : none (question named no record_id)")
        print(f"  answer:\n      {run.response.answer[:420]}")
        print()

    print("=" * 70)
    print("Note on the tool names: the retrieval tool is called `rag_lookup`, which")
    print("contains the substring 'lookup'. Dispatch is done on each tool's declared")
    print("argument schema (record_id vs query), so the name never enters into it.")


if __name__ == "__main__":
    main()
