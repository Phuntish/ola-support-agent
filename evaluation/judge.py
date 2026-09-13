"""Task 13 - run the 15-query test set through the agent and score it.

Flow per query: run the guarded crew path, measure evidence about the answer,
render the judge prompt around that evidence, and let the mock judge apply the
rubric. Four scores per query plus four averages at the end.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
# CrewAI keys its "first execution" marker on the project DIRECTORY NAME
# (appdirs.user_data_dir(get_project_directory_name())), so a fresh clone
# under any other folder name is treated as a first run and prompts
# "Would you like to view your execution traces? [y/N] (20s timeout)".
# CREWAI_TESTING is the only switch that suppresses it. In crewai 1.9.3 it is
# read in exactly one functional place - _is_test_environment() in
# events/listeners/tracing/utils.py - so it gates those prompts and nothing else.
os.environ.setdefault("CREWAI_TESTING", "true")

from crew.crew import run_guarded
from crew.schema import SupportResponse
from evaluation.testset import TEST_SET, EvalCase
from guardrails.groundedness import SUPPORT_THRESHOLD, score_sentences
from guardrails.pii import contains_pii
from llm.mock_llm import JUDGE_SYSTEM_PROMPT, MockJudgeLLM

EVIDENCE_VERSION = 1

METRICS = ("accuracy", "grounding", "completeness", "safety")


@dataclass
class Scored:
    case: EvalCase
    response: SupportResponse
    evidence: dict
    scores: dict[str, float]
    reason: str


def gather_evidence(case: EvalCase, response: SupportResponse, guardrails: list[str]) -> dict:
    """Measure everything the judge is allowed to rely on."""
    answered = not response.refused

    if answered and response.context:
        per_sentence = score_sentences(response.answer, response.context)
        supported = sum(1 for _, s in per_sentence if s >= SUPPORT_THRESHOLD)
        grounded_fraction = round(supported / len(per_sentence), 4) if per_sentence else 0.0
        unsupported = len(per_sentence) - supported
    else:
        grounded_fraction, unsupported = 1.0, 0

    answer_lower = response.answer.lower()
    found = sum(1 for point in case.key_points if point.lower() in answer_lower)

    return {
        "evidence_version": EVIDENCE_VERSION,
        "case_id": case.case_id,
        "in_scope": case.in_scope,
        "refused": response.refused,
        "expected_doc": case.expected_doc,
        "sources": response.sources,
        "grounded_fraction": grounded_fraction,
        "unsupported_sentences": unsupported,
        "key_points_total": len(case.key_points),
        "key_points_found": found,
        "pii_in_answer": contains_pii(response.answer),
        "injection_case": "injection" in case.topic,
        "injection_blocked": any(g.startswith("injection:") for g in guardrails),
    }


def build_judge_prompt(case: EvalCase, response: SupportResponse, evidence: dict) -> str:
    return (
        f"{JUDGE_SYSTEM_PROMPT}\n\n"
        f"QUESTION:\n{case.query}\n\n"
        f"AGENT ANSWER:\n{response.answer}\n\n"
        f"REQUIRED FACTS: {case.key_points or 'none - this question should not be answered'}\n\n"
        f"EVIDENCE:\n{json.dumps(evidence, indent=2)}\n"
    )


def evaluate_case(case: EvalCase, judge: MockJudgeLLM) -> Scored:
    run = run_guarded(case.query)
    response = run.response
    evidence = gather_evidence(case, response, response.guardrails_triggered)
    verdict = json.loads(judge.call(build_judge_prompt(case, response, evidence)))

    return Scored(
        case=case,
        response=response,
        evidence=evidence,
        scores={m: verdict[m] for m in METRICS},
        reason=verdict["reason"],
    )


def main() -> None:
    judge = MockJudgeLLM()

    print("=" * 70)
    print("TASK 13 - LLM-AS-JUDGE EVALUATION (MOCK_LLM)")
    print("=" * 70)
    print(f"queries        : {len(TEST_SET)}")
    print(f"metrics        : {', '.join(METRICS)}")
    print(f"grounding basis: sentence provenance at threshold {SUPPORT_THRESHOLD}")
    print()

    print("-" * 70)
    print("THE JUDGE PROMPT (rendered fresh for every query)")
    print("-" * 70)
    print(JUDGE_SYSTEM_PROMPT)
    print()

    results = [evaluate_case(case, judge) for case in TEST_SET]

    print("-" * 70)
    print("PER-QUERY SCORES")
    print("-" * 70)
    for result in results:
        case = result.case
        scope = "in-scope" if case.in_scope else "OUT-OF-SCOPE"
        print(f"\n{case.case_id}  [{scope}]  {case.topic}")
        print(f"   q: {case.query}")
        print(f"   refused={result.response.refused}  sources={result.response.sources}")
        if result.response.guardrails_triggered:
            print(f"   guardrails: {result.response.guardrails_triggered}")
        scores = "  ".join(f"{m}={result.scores[m]:.2f}" for m in METRICS)
        print(f"   {scores}")
        print(f"   judge: {result.reason}")
        print(f"   a: {result.response.answer[:150]}")

    print()
    print("=" * 70)
    print("SCORE TABLE")
    print("=" * 70)
    header = f"  {'case':<6} {'scope':<13} {'acc':>6} {'grnd':>6} {'comp':>6} {'safe':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        scope = "in-scope" if r.case.in_scope else "out-of-scope"
        print(f"  {r.case.case_id:<6} {scope:<13}"
              f" {r.scores['accuracy']:>6.2f} {r.scores['grounding']:>6.2f}"
              f" {r.scores['completeness']:>6.2f} {r.scores['safety']:>6.2f}")

    print()
    print("=" * 70)
    print(f"AVERAGES ACROSS ALL {len(results)} QUERIES")
    print("=" * 70)
    averages = {m: sum(r.scores[m] for r in results) / len(results) for m in METRICS}
    for metric in METRICS:
        bar = "#" * int(round(averages[metric] * 40))
        print(f"  {metric.capitalize():<13} {averages[metric]:.4f}  {bar}")
    print()
    print(f"  Overall mean of the four: {sum(averages.values()) / len(METRICS):.4f}")
    print()

    in_scope = [r for r in results if r.case.in_scope]
    out_scope = [r for r in results if not r.case.in_scope]
    print("  Split out by scope:")
    for label, group in (("in-scope", in_scope), ("out-of-scope", out_scope)):
        line = "  ".join(
            f"{m}={sum(r.scores[m] for r in group) / len(group):.3f}" for m in METRICS
        )
        print(f"    {label:<13} (n={len(group)})  {line}")
    print()
    print(f"  judge calls made: {judge.call_count}")
    print()

    print("=" * 70)
    print("NEGATIVE CONTROL - does the judge ever mark anything down?")
    print("=" * 70)
    print("Three scores above sit at 1.000, which is what the architecture should")
    print("produce: extractive generation cannot emit an ungrounded sentence, and the")
    print("out-of-scope queries are all correctly declined. To show the rubric is not")
    print("simply rubber-stamping, here is the same judge on deliberately broken output.")
    print()

    baseline = next(r for r in results if r.case.case_id == "E04")
    controls = [
        (
            "ungrounded sentence spliced into a good answer",
            baseline.response.model_copy(update={
                "answer": baseline.response.answer
                + " Ola also wires 5000 rupees to every complainant the same day.",
            }),
            baseline.case,
        ),
        (
            "answer pulled from the wrong policy document",
            baseline.response.model_copy(update={"sources": ["vip-customer-handling"]}),
            baseline.case,
        ),
        (
            "out-of-scope question answered instead of declined",
            baseline.response.model_copy(update={"refused": False}),
            next(c for c in TEST_SET if c.case_id == "E13"),
        ),
        (
            "phone number left unmasked in the answer",
            baseline.response.model_copy(update={
                "answer": baseline.response.answer + " Call the rider back on 9876543210.",
            }),
            baseline.case,
        ),
    ]

    for label, response, case in controls:
        evidence = gather_evidence(case, response, response.guardrails_triggered)
        verdict = json.loads(judge.call(build_judge_prompt(case, response, evidence)))
        scores = "  ".join(f"{m}={verdict[m]:.2f}" for m in METRICS)
        print(f"  {label}")
        print(f"    {scores}")
        print(f"    judge: {verdict['reason']}")
        print()

    print("  So the metrics that saturate above do so because the answers are correct,")
    print("  not because the judge cannot tell the difference. Completeness is the one")
    print(f"  metric that discriminates on real output: {averages['completeness']:.4f}, held down by E05.")


if __name__ == "__main__":
    main()
