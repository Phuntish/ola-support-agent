"""Regenerate every transcript in transcripts/ by actually running the code.

Each step runs as its own subprocess so a failure in one task cannot leave a
half-written transcript behind for another. Run it with the project venv:

    .venv/bin/python run_all.py
    .venv/bin/python run_all.py task04        # one step, by transcript prefix
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRANSCRIPTS = ROOT / "transcripts"

# (transcript name, argv for python -m, human label)
STEPS: list[tuple[str, list[str], str]] = [
    # Part 1 - dataset and RAG core
    ("task01_dataset", ["dataset"], "Task 1 - support ticket dataset"),
    ("task02_knowledge_base", ["rag.kb_report"], "Task 2 - knowledge base inventory"),
    ("task03a_chunking", ["rag.chunking"], "Task 3 - chunking strategies"),
    ("task03b_indexing", ["rag.index"], "Task 3 - embedding and Chroma collections"),
    ("task04_grounded_generation", ["rag.generate"], "Task 4 - grounded generation"),
    ("task05_chunking_eval", ["rag.eval_chunks"], "Task 5 - chunking comparison"),
    # Part 2 - orchestration, memory, schema, guardrails
    ("task06_ticket_tool", ["tools.ticket_status"], "Task 6 - ticket lookup and escalation score"),
    ("task07_crew_tool_use", ["crew.crew"], "Task 7 - CrewAI crew with tool use"),
    ("task08a_memory_multi_turn", ["crew.memory", "multi"], "Task 8 - memory carried across turns"),
    ("task08b_memory_fresh_session", ["crew.memory", "fresh"], "Task 8 - fresh session, state absent"),
    ("task09_schema_validation", ["crew.schema"], "Task 9 - structured response schema"),
    ("task10a_guardrail_pii", ["guardrails.pii"], "Task 10 - PII masking"),
    ("task10b_guardrail_injection", ["guardrails.injection"], "Task 10 - prompt injection"),
    ("task10c_guardrail_groundedness", ["guardrails.groundedness"], "Task 10 - groundedness"),
    ("task10d_guardrails_end_to_end", ["guardrails.pipeline"], "Task 10 - all three on the live path"),
    # Part 3 - API, observability, evaluation
    ("task11_api_endpoints", ["api.demo", "endpoints"], "Task 11 - FastAPI endpoints and WebSocket"),
    ("task12_logging", ["api.demo", "logging"], "Task 12 - JSON-Lines logging and PII audit"),
    ("task13a_testset", ["evaluation.testset"], "Task 13 - the 15-query test set"),
    ("task13b_judge_eval", ["evaluation.judge"], "Task 13 - LLM-as-judge scoring"),
    # Part 4 - review stage, governance, caching
    ("task14a_review_approved", ["review.demo", "approved"], "Task 14 - review approves unchanged"),
    ("task14b_review_revised", ["review.demo", "revised"], "Task 14 - review revises the draft"),
    ("task15a_least_autonomy", ["governance.least_autonomy"], "Task 15 - least autonomy + risk"),
    ("task15b_budget_cap", ["governance.budget"], "Task 15 - runtime budget cap"),
    ("task16_response_cache", ["governance.cache"], "Task 16 - response caching"),
]


def child_env() -> dict[str, str]:
    """Everything runs offline and telemetry-free, and the child must inherit that."""
    env = os.environ.copy()
    env.setdefault("MOCK_LLM", "1")
    env.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    env.setdefault("OTEL_SDK_DISABLED", "true")
    env.setdefault("CREWAI_TRACING_ENABLED", "false")
    # Suppresses CrewAI's first-run tracing prompt, which otherwise appears for
    # anyone running this from a differently named directory. See crew/crew.py.
    env.setdefault("CREWAI_TESTING", "true")
    env.setdefault("ANONYMIZED_TELEMETRY", "False")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["PYTHONPATH"] = str(ROOT)
    return env


def run_step(name: str, argv: list[str], label: str) -> bool:
    target = TRANSCRIPTS / f"{name}.txt"
    started = time.perf_counter()

    proc = subprocess.run(
        [sys.executable, "-m", *argv],
        cwd=ROOT,
        env=child_env(),
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started

    header = (
        f"# {label}\n"
        f"# command: python -m {' '.join(argv)}\n"
        f"# MOCK_LLM=1, CREWAI_DISABLE_TELEMETRY=true, no network calls\n"
        f"# exit code: {proc.returncode}, wall clock: {elapsed:.1f}s\n"
        + "#" * 70
        + "\n\n"
    )
    body = proc.stdout
    if proc.stderr.strip():
        body += "\n--- stderr ---\n" + proc.stderr

    target.write_text(header + body, encoding="utf-8")

    status = "ok " if proc.returncode == 0 else "FAIL"
    print(f"  [{status}] {name:<30} {elapsed:>6.1f}s  -> transcripts/{name}.txt")
    if proc.returncode != 0:
        print(proc.stderr.strip()[-800:])
    return proc.returncode == 0


def main() -> int:
    TRANSCRIPTS.mkdir(exist_ok=True)

    wanted = sys.argv[1:]
    steps = [s for s in STEPS if not wanted or any(s[0].startswith(w) for w in wanted)]
    if not steps:
        print(f"no steps matched {wanted}; known steps: {[s[0] for s in STEPS]}")
        return 1

    print(f"Regenerating {len(steps)} transcript(s) into {TRANSCRIPTS}\n")
    results = [run_step(*step) for step in steps]

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} steps passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
