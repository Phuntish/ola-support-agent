"""Task 2 - inventory of the hand-written knowledge base.

Checks that every required topic from the brief has a document behind it and that
each document sits in the 2-5 sentence band the brief asks for.
"""

from __future__ import annotations

from rag.chunking import load_documents, split_sentences

# Every topic the brief requires, mapped to the document that covers it.
REQUIRED_TOPICS = {
    "ticket-priority classification rules": "ticket-priority-classification",
    "SLA-by-severity policy": "sla-by-severity",
    "escalation matrix": "escalation-matrix",
    "refund/compensation policy": "refund-and-compensation",
    "customer-communication-channel policy": "communication-channels",
    "business-hours/holiday-support policy": "business-hours-and-holidays",
    "repeat-complaint-handling policy": "repeat-complaint-handling",
    "service-credit policy": "service-credit-policy",
    "feedback-collection process": "feedback-collection",
    "VIP-customer handling policy": "vip-customer-handling",
    "outage-communication protocol": "outage-communication",
    "ticket data-retention policy": "ticket-data-retention",
}

MIN_SENTENCES = 2
MAX_SENTENCES = 5


def main() -> None:
    documents = load_documents()
    by_id = {d.doc_id: d for d in documents}

    print("=" * 70)
    print("TASK 2 - KNOWLEDGE BASE INVENTORY")
    print("=" * 70)
    print(f"documents authored : {len(documents)}  (brief requires >=12)")
    print(f"sentence band      : {MIN_SENTENCES}-{MAX_SENTENCES} sentences per document")
    print()

    print(f"{'doc_id':<34} {'sent':>5} {'words':>6}  title")
    print("-" * 100)
    problems: list[str] = []
    for doc in documents:
        n_sentences = len(split_sentences(doc.text))
        n_words = len(doc.text.split())
        print(f"{doc.doc_id:<34} {n_sentences:>5} {n_words:>6}  {doc.title}")
        if not MIN_SENTENCES <= n_sentences <= MAX_SENTENCES:
            problems.append(f"{doc.doc_id}: {n_sentences} sentences, outside the {MIN_SENTENCES}-{MAX_SENTENCES} band")
    print()

    print("Required topic coverage")
    print("-" * 70)
    for topic, doc_id in REQUIRED_TOPICS.items():
        covered = doc_id in by_id
        print(f"  [{'x' if covered else ' '}] {topic:<42} -> {doc_id}")
        if not covered:
            problems.append(f"required topic {topic!r} has no document")
    print()

    extra = sorted(set(by_id) - set(REQUIRED_TOPICS.values()))
    if extra:
        print(f"Additional documents beyond the required list: {extra}")
        print()

    if problems:
        print("VALIDATION FAILED")
        for p in problems:
            print(f"  - {p}")
    else:
        print(f"VALIDATION PASSED - {len(documents)} documents, all {len(REQUIRED_TOPICS)} required topics covered")


if __name__ == "__main__":
    main()
