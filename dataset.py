"""Task 1 - seeded support-ticket dataset for the Ola support agent.

The generator is fully deterministic: same seed in, same SUPPORT_TICKETS out.
Nothing here is hand-edited; the structural targets in the brief are hit by
choosing the seed and the category/status weights, then re-running.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

SEED = 20260908
N_TICKETS = 60

CATEGORIES = ["Billing", "Technical Issue", "Account Access", "Product Defect", "General Inquiry"]
STATUSES = ["Open", "In Progress", "Escalated", "Resolved", "Closed"]

# Roughly the mix an Ola city-ops queue sees: fare/refund disputes dominate,
# app faults next, then login/KYC problems, with defects the rarest.
CATEGORY_WEIGHTS = [0.30, 0.22, 0.16, 0.12, 0.20]

# Most tickets are already finished; only a thin slice is live at any moment.
STATUS_WEIGHTS = [0.18, 0.20, 0.10, 0.34, 0.18]

# (low, mode, high) hours for a triangular draw, per category.
# Range 0.25-72 h: a single-touch FAQ reply at the bottom, the three-business-day
# ceiling of our lowest severity tier at the top, so every SLA tier fits inside it
# without implying week-long backlogs.
RESOLUTION_HOURS = {
    "Billing": (0.5, 6.0, 48.0),  # needs a finance-side confirmation before closure
    "Technical Issue": (1.0, 12.0, 72.0),  # may wait on an app release train
    "Account Access": (0.5, 3.0, 24.0),  # OTP/KYC resets are near-scripted
    "Product Defect": (2.0, 24.0, 72.0),  # reproduce, file, verify the fix
    "General Inquiry": (0.25, 2.0, 12.0),  # answered from the knowledge base
}

# Age windows per status. A ticket sitting in Open for three weeks is not
# realistic, while Closed tickets are mostly old ones.
AGE_WINDOWS = {
    "Open": (0, 14),
    "In Progress": (1, 21),
    "Escalated": (2, 25),
    "Resolved": (3, 28),
    "Closed": (7, 30),
}

MAX_RESOLUTION_HOURS = 72.0
STALE_AFTER_DAYS = 14


def _draw_escalated(rng: random.Random, status: str, days_since_created: int) -> bool:
    """Escalation follows the ticket's own history rather than an independent coin flip."""
    if status == "Escalated":
        return True
    if status in ("Open", "In Progress") and days_since_created >= STALE_AFTER_DAYS:
        # Aged live tickets are the ones that get pulled into an escalation review.
        return rng.random() < 0.35
    if status in ("Resolved", "Closed"):
        # A small tail of finished tickets went through escalation on the way.
        return rng.random() < 0.06
    return False


def generate_tickets(seed: int = SEED, n: int = N_TICKETS) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    tickets: list[dict[str, Any]] = []

    for i in range(1, n + 1):
        category = rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0]
        status = rng.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]

        low, high = AGE_WINDOWS[status]
        days_since_created = rng.randint(low, high)

        escalated = _draw_escalated(rng, status, days_since_created)

        lo, mode, hi = RESOLUTION_HOURS[category]
        hours = rng.triangular(lo, hi, mode)
        if escalated:
            # An escalation adds a hand-off, so the clock runs longer.
            hours = min(hours * 1.5, MAX_RESOLUTION_HOURS)

        tickets.append(
            {
                "record_id": f"OLA-{i:04d}",
                "category": category,
                "status": status,
                "resolution_time_hours": round(hours, 1),
                "days_since_created": days_since_created,
                "escalated": escalated,
            }
        )

    return tickets


SUPPORT_TICKETS: list[dict[str, Any]] = generate_tickets()


def get_ticket(record_id: str) -> dict[str, Any] | None:
    """Case-insensitive lookup by record_id."""
    wanted = record_id.strip().upper()
    return next((t for t in SUPPORT_TICKETS if t["record_id"] == wanted), None)


def escalated_percentage(tickets: list[dict[str, Any]] | None = None) -> float:
    rows = SUPPORT_TICKETS if tickets is None else tickets
    return 100.0 * sum(t["escalated"] for t in rows) / len(rows)


def validate(tickets: list[dict[str, Any]] | None = None) -> list[str]:
    """Return a list of failed structural checks; empty list means the dataset is good."""
    rows = SUPPORT_TICKETS if tickets is None else tickets
    problems: list[str] = []

    if len(rows) < 40:
        problems.append(f"need >=40 tickets, generated {len(rows)}")

    by_category = Counter(t["category"] for t in rows)
    for category in CATEGORIES:
        if by_category[category] < 3:
            problems.append(f"category {category!r} has {by_category[category]} records, need >=3")

    by_status = Counter(t["status"] for t in rows)
    for status in STATUSES:
        if by_status[status] < 1:
            problems.append(f"status {status!r} is unused, need >=1")

    pct = escalated_percentage(rows)
    if not 10.0 <= pct <= 30.0:
        problems.append(f"escalated share {pct:.1f}% is outside the 10-30% band")

    for t in rows:
        if not 0 <= t["days_since_created"] <= 30:
            problems.append(f"{t['record_id']}: days_since_created={t['days_since_created']} out of 0-30")

    return problems


def report(tickets: list[dict[str, Any]] | None = None) -> None:
    rows = SUPPORT_TICKETS if tickets is None else tickets

    print("=" * 70)
    print("TASK 1 - SUPPORT TICKET DATASET")
    print("=" * 70)
    print(f"seed                 : {SEED}")
    print(f"records generated    : {len(rows)}")
    print(f"resolution_time_hours: {min(t['resolution_time_hours'] for t in rows):.1f}"
          f" - {max(t['resolution_time_hours'] for t in rows):.1f} h"
          f" (design range 0.25 - {MAX_RESOLUTION_HOURS:.0f} h)")
    print()

    by_category = Counter(t["category"] for t in rows)
    print("Count per category (each given category needs >=3)")
    for category in CATEGORIES:
        print(f"  {category:<16} {by_category[category]:>3}")
    extra = sorted(set(by_category) - set(CATEGORIES))
    for category in extra:
        print(f"  {category:<16} {by_category[category]:>3}  (added)")
    print()

    by_status = Counter(t["status"] for t in rows)
    print("Count per status (each given status needs >=1)")
    for status in STATUSES:
        print(f"  {status:<16} {by_status[status]:>3}")
    print()

    n_escalated = sum(t["escalated"] for t in rows)
    print(f"Escalated: {n_escalated}/{len(rows)} = {escalated_percentage(rows):.1f}%  (target band 10-30%)")
    print()

    print("Mean resolution_time_hours by category")
    for category in CATEGORIES:
        hours = [t["resolution_time_hours"] for t in rows if t["category"] == category]
        print(f"  {category:<16} {sum(hours) / len(hours):>6.1f} h   (n={len(hours)})")
    print()

    print("First 10 records")
    header = f"  {'record_id':<10} {'category':<16} {'status':<12} {'hours':>6} {'days':>5} {'esc':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for t in rows[:10]:
        print(f"  {t['record_id']:<10} {t['category']:<16} {t['status']:<12}"
              f" {t['resolution_time_hours']:>6.1f} {t['days_since_created']:>5} {str(t['escalated']):>5}")
    print()

    problems = validate(rows)
    if problems:
        print("VALIDATION FAILED")
        for p in problems:
            print(f"  - {p}")
    else:
        print("VALIDATION PASSED - all structural thresholds met")


if __name__ == "__main__":
    report()
