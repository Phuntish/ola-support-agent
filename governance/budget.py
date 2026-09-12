"""Task 15, runtime layer - a per-request token and cost cap.

The cap is enforced before any agent runs. Rejecting up front is the point: once
the crew has started, a runaway request has already cost the tokens you were
trying not to spend.

Under MOCK_LLM nothing is actually billed, so the rate below is a stated
assumption rather than a measurement - it is what the cap would cost against a
small hosted model. The token counts themselves are real, in the sense that they
are measured from the text that would be sent.
"""

from __future__ import annotations

from dataclasses import dataclass

# ~4 characters per token is the usual rule of thumb for English and it is close
# enough for a budget guard, which only has to catch the order of magnitude.
CHARS_PER_TOKEN = 4

MAX_REQUEST_TOKENS = 1500
USD_PER_1K_TOKENS = 0.00025
MAX_REQUEST_USD = round(MAX_REQUEST_TOKENS / 1000 * USD_PER_1K_TOKENS, 6)

# A crew run makes several model calls over the same question plus retrieved
# context, so the question alone understates what the request will cost.
EXPECTED_CALLS_PER_REQUEST = 5
CONTEXT_TOKENS_PER_CALL = 180


class BudgetExceeded(Exception):
    """Raised when a request would cost more than one request is allowed to."""


@dataclass
class BudgetReport:
    question_tokens: int
    projected_tokens: int
    projected_usd: float
    limit_tokens: int
    limit_usd: float

    @property
    def within_budget(self) -> bool:
        return self.projected_tokens <= self.limit_tokens

    def describe(self) -> str:
        return (
            f"{self.projected_tokens} projected tokens "
            f"(${self.projected_usd:.6f}) against a cap of {self.limit_tokens} "
            f"(${self.limit_usd:.6f})"
        )


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def project_request(question: str) -> BudgetReport:
    """What the whole crew run is expected to consume, not just the question."""
    question_tokens = estimate_tokens(question)
    projected = (question_tokens + CONTEXT_TOKENS_PER_CALL) * EXPECTED_CALLS_PER_REQUEST
    return BudgetReport(
        question_tokens=question_tokens,
        projected_tokens=projected,
        projected_usd=round(projected / 1000 * USD_PER_1K_TOKENS, 6),
        limit_tokens=MAX_REQUEST_TOKENS,
        limit_usd=MAX_REQUEST_USD,
    )


def enforce_budget(question: str) -> BudgetReport:
    """Check before spending. Raises BudgetExceeded with something a caller can act on."""
    report = project_request(question)
    if not report.within_budget:
        raise BudgetExceeded(
            f"Request rejected: it would use about {report.projected_tokens} tokens "
            f"(${report.projected_usd:.6f}), over the per-request cap of "
            f"{report.limit_tokens} tokens (${report.limit_usd:.6f}). "
            f"The question is {report.question_tokens} tokens on its own. "
            f"Shorten it or split it into separate questions."
        )
    return report


def main() -> None:
    from crew.crew import run_guarded

    print("=" * 70)
    print("TASK 15 - RUNTIME LAYER: PER-REQUEST BUDGET CAP")
    print("=" * 70)
    print(f"  cap per request      : {MAX_REQUEST_TOKENS} tokens  (${MAX_REQUEST_USD:.6f})")
    print(f"  assumed rate         : ${USD_PER_1K_TOKENS} per 1K tokens")
    print(f"  token estimate       : len(text) // {CHARS_PER_TOKEN}")
    print(f"  projection           : (question + {CONTEXT_TOKENS_PER_CALL} context)"
          f" x {EXPECTED_CALLS_PER_REQUEST} calls")
    print()
    print("  The cap is checked before the crew starts. Rejecting after the fact would")
    print("  mean the tokens were already spent, which defeats the purpose.")
    print()

    normal = "What is the refund rule when a rider is charged twice?"

    # A support agent pasting an entire chat log into the box is the realistic
    # version of this, not an attack.
    oversized = (
        "Here is the whole conversation with the rider, please read all of it and tell me "
        "what our refund and escalation policy says we should do. "
    ) + ("The rider said they were charged twice and then the driver cancelled again. " * 120)

    print("-" * 70)
    print("1. NORMAL REQUEST - passes the cap and runs")
    print("-" * 70)
    report = project_request(normal)
    print(f"  question         : {normal}")
    print(f"  question chars   : {len(normal)}")
    print(f"  {report.describe()}")
    print(f"  within budget    : {report.within_budget}")
    run = run_guarded(normal)
    print(f"  crew ran         : yes, {run.llm_calls} model calls")
    print(f"  answer           : {run.response.answer[:140]}")
    print()

    print("-" * 70)
    print("2. OVERSIZED REQUEST - rejected before anything runs")
    print("-" * 70)
    report = project_request(oversized)
    print(f"  question chars   : {len(oversized)}")
    print(f"  question preview : {oversized[:110]}...")
    print(f"  {report.describe()}")
    print(f"  within budget    : {report.within_budget}")
    print()
    try:
        enforce_budget(oversized)
    except BudgetExceeded as exc:
        print("  BudgetExceeded raised:")
        print(f"    {exc}")
    print()

    print("  and through the live request path, which applies the cap itself:")
    run = run_guarded(oversized)
    print(f"    refused              : {run.response.refused}")
    print(f"    guardrails_triggered : {run.response.guardrails_triggered}")
    print(f"    model calls made     : {run.llm_calls}   <- nothing was spent")
    print(f"    reply                : {run.response.answer[:200]}")
    print()

    print("-" * 70)
    print("3. WHERE THE CAP BITES")
    print("-" * 70)
    print(f"  {'question chars':>16} {'projected tokens':>18} {'within cap':>12}")
    for size in (100, 400, 800, 1000, 1200, 2000):
        probe = "x" * size
        r = project_request(probe)
        print(f"  {size:>16} {r.projected_tokens:>18} {str(r.within_budget):>12}")


if __name__ == "__main__":
    main()
