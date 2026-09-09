"""Task 6 - ticket lookup with a designed escalation score.

The score is deliberately not `escalated OR stale`. It is a weighted blend of the
escalation flag and how far the ticket has aged, so that three different
situations come out with three different numbers:

    a flagged ticket                     -> at least 0.55
    an untouched ticket that went stale  -> up to 0.45
    a fresh untouched ticket             -> 0.0

That matters because the second case is the one a bare boolean misses: nobody has
escalated the ticket, which is exactly why it needs a human to look at it.
"""

from __future__ import annotations

import math
from typing import Any

from dataset import SUPPORT_TICKETS, get_ticket

# Weights. The flag is worth more than the age because an explicit escalation is
# a human decision, but the age is worth enough that a stale ticket can still
# clear the threshold on its own.
WEIGHT_FLAG = 0.55
WEIGHT_AGE = 0.45

# The age term is normalised against the dataset's own ceiling.
MAX_AGE_DAYS = 30

# Recommend escalation at the 80th percentile of days_since_created - see
# ESCALATION_THRESHOLD below, which is derived from the data rather than typed in.
THRESHOLD_PERCENTILE = 80


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile, so the threshold moves if the dataset does."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a percentile of an empty sequence")
    k = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(ordered[lo])
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def escalation_score(escalated: bool, days_since_created: int) -> float:
    """escalation_score = 0.55 * escalated + 0.45 * min(days_since_created / 30, 1)"""
    normalised_age = min(days_since_created / MAX_AGE_DAYS, 1.0)
    return round(WEIGHT_FLAG * float(escalated) + WEIGHT_AGE * normalised_age, 4)


# Derived at import time from the generated dataset, not hand-picked: the score a
# ticket would have if nobody had escalated it but it had aged past the 80th
# percentile of days_since_created.
P80_DAYS = percentile([t["days_since_created"] for t in SUPPORT_TICKETS], THRESHOLD_PERCENTILE)
ESCALATION_THRESHOLD = round(WEIGHT_AGE * min(P80_DAYS / MAX_AGE_DAYS, 1.0), 4)


def check_support_ticket_status(record_id: str) -> dict[str, Any]:
    """Look up one ticket and score how badly it needs escalating.

    Returns status, resolution_time_hours and escalation_score. An unknown id
    comes back as a dict with an `error` key rather than raising, because this is
    called by an agent that has to say something sensible either way.
    """
    ticket = get_ticket(record_id)
    if ticket is None:
        return {
            "record_id": record_id,
            "error": "not_found",
            "message": f"No support ticket matches {record_id!r}.",
        }

    score = escalation_score(ticket["escalated"], ticket["days_since_created"])
    return {
        "record_id": ticket["record_id"],
        "category": ticket["category"],
        "status": ticket["status"],
        "resolution_time_hours": ticket["resolution_time_hours"],
        "escalation_score": score,
        "escalate_recommended": score >= ESCALATION_THRESHOLD,
        "escalation_threshold": ESCALATION_THRESHOLD,
        "days_since_created": ticket["days_since_created"],
        "escalated_flag": ticket["escalated"],
    }


def ticket_status_tool():
    """Build the CrewAI tool wrapper.

    Deliberately a factory rather than a module-level `@tool` object: Task 15
    hands this out through a registry so that only the Lookup Agent can ever hold
    it, and a module-level singleton would let any agent import it directly.
    """
    from crewai.tools import tool

    @tool("check_support_ticket_status")
    def check_support_ticket_status_tool(record_id: str) -> str:
        """Look up one Ola support ticket by its record_id (for example OLA-0006).

        Returns JSON with the ticket's status, resolution_time_hours and a
        designed escalation_score between 0 and 1.
        """
        import json

        return json.dumps(check_support_ticket_status(record_id), indent=2)

    return check_support_ticket_status_tool


def _distribution() -> list[float]:
    return [escalation_score(t["escalated"], t["days_since_created"]) for t in SUPPORT_TICKETS]


def main() -> None:
    print("=" * 70)
    print("TASK 6 - TICKET LOOKUP AND ESCALATION SCORE")
    print("=" * 70)
    print("Formula")
    print(f"  escalation_score = {WEIGHT_FLAG} * escalated"
          f" + {WEIGHT_AGE} * min(days_since_created / {MAX_AGE_DAYS}, 1)")
    print(f"  range [0, 1]: fresh and unflagged = 0.0, flagged and 30 days old = 1.0")
    print()

    print("Threshold, derived from this dataset")
    print(f"  p{THRESHOLD_PERCENTILE} of days_since_created over {len(SUPPORT_TICKETS)} tickets = {P80_DAYS:.2f} days")
    print(f"  score of an unflagged ticket at that age  = {WEIGHT_AGE} * {P80_DAYS:.2f}/{MAX_AGE_DAYS}"
          f" = {ESCALATION_THRESHOLD}")
    print(f"  ESCALATION_THRESHOLD = {ESCALATION_THRESHOLD}")
    print()

    scores = _distribution()
    flagged = [t for t in SUPPORT_TICKETS
               if escalation_score(t["escalated"], t["days_since_created"]) >= ESCALATION_THRESHOLD]
    escalated_only = sum(1 for t in flagged if t["escalated"])
    stale_only = sum(1 for t in flagged if not t["escalated"])

    print("What that threshold actually selects")
    print(f"  score range across the dataset : {min(scores):.4f} - {max(scores):.4f}")
    print(f"  lowest score of a flagged ticket : "
          f"{min(s for s, t in zip(scores, SUPPORT_TICKETS) if t['escalated']):.4f}")
    print(f"  highest score of an unflagged one: "
          f"{max(s for s, t in zip(scores, SUPPORT_TICKETS) if not t['escalated']):.4f}")
    print(f"  tickets at or above threshold  : {len(flagged)}/{len(SUPPORT_TICKETS)}"
          f" = {100 * len(flagged) / len(SUPPORT_TICKETS):.1f}%")
    print(f"    - carrying escalated=True    : {escalated_only}")
    print(f"    - stale but never escalated  : {stale_only}   <- what a bare boolean OR would miss")
    print()

    print("Sample lookups")
    print("-" * 70)
    # OLA-0006 is flagged, OLA-0005 is the interesting one - never escalated but
    # 26 days old, so it clears the threshold on age alone. OLA-0010 is fresh and
    # quiet, OLA-9999 does not exist.
    samples = ["OLA-0006", "OLA-0005", "OLA-0010", "OLA-9999"]
    for record_id in samples:
        result = check_support_ticket_status(record_id)
        print(f"\n  check_support_ticket_status({record_id!r})")
        for key, value in result.items():
            print(f"      {key:<24} {value}")


if __name__ == "__main__":
    main()
