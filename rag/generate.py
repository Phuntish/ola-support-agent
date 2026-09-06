"""Task 4 - grounded generation with an empirically calibrated refusal threshold.

Under MOCK_LLM there is no model that can be trusted to stay inside the context,
so the generator is extractive: it only ever re-emits sentences that came back
from the retriever. That makes groundedness a property of the code rather than
something we hope the model respects, and it leaves retrieval similarity as the
only signal for "do I actually know this?".

The threshold below is not a tutorial default. It is the midpoint of the gap
measured by `calibrate()` between in-scope and out-of-scope top-1 similarities on
this specific knowledge base and embedding model. Re-run `python -m rag.generate`
to reproduce the numbers.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from rag.chunking import SENTENCE_STRATEGY, load_documents, split_sentences
from rag.index import DEFAULT_TOP_K, Hit, embed, retrieve

# Measured on this KB with all-MiniLM-L6-v2 - see calibrate() output in
# transcripts/task04_grounded_generation.txt and the table in README.md.
SIMILARITY_THRESHOLD = 0.3060

REFUSAL_TEXT = (
    "I don't know. The Ola support knowledge base I have access to does not cover that, "
    "so I would rather say nothing than guess."
)

MAX_ANSWER_SENTENCES = 3

# top_k always returns k chunks, and the weakest of them is often from an
# unrelated policy that merely shares vocabulary. Only chunks within this much of
# the best hit are allowed to contribute sentences to the answer.
CONTEXT_MARGIN = 0.15

CALIBRATION_IN_SCOPE = [
    "What is the first response time for a P1 ticket?",
    "How long does a refund take to reach the original payment method?",
    "When does a ticket escalate from Tier 1 to Tier 2?",
    "How much is a standard service credit worth?",
    "How long are closed support tickets kept before deletion?",
    "Are billing queries handled on a public holiday?",
]

CALIBRATION_OUT_OF_SCOPE = [
    "What is the boiling point of water at sea level?",
    "Who won the cricket world cup in 2011?",
    "Write me a Python function that reverses a linked list.",
]

DEMO_IN_SCOPE = [
    "How quickly must we reply to a P2 billing complaint?",
    "A rider was charged twice for the same trip, what is the refund rule?",
    "When should a support agent phone the customer instead of using chat?",
    "What happens if the same customer complains about the same thing three times?",
    "What do we tell riders while a city-wide outage is still going on?",
]

DEMO_OUT_OF_SCOPE = "What is the current share price of Ola Electric?"


@dataclass
class GroundedAnswer:
    query: str
    answer: str
    refused: bool
    top_similarity: float
    sources: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "refused": self.refused,
            "top_similarity": self.top_similarity,
            "sources": self.sources,
        }


def _document_titles() -> set[str]:
    return {doc.title for doc in load_documents()}


def _context_sentences(hits: list[Hit], titles: set[str]) -> list[str]:
    """Flatten retrieved chunks into unique sentences, dropping the title prefix."""
    seen: set[str] = set()
    sentences: list[str] = []
    for hit in hits:
        for sentence in split_sentences(hit.text):
            stripped = sentence.rstrip(".").strip()
            if stripped in titles:
                continue
            if sentence in seen:
                continue
            seen.add(sentence)
            sentences.append(sentence)
    return sentences


def compose_answer(query: str, hits: list[Hit]) -> str:
    """Pick the retrieved sentences that answer the query, keeping their original order.

    Nothing is written here that did not come out of the retriever, which is what
    makes the output grounded by construction.
    """
    sentences = _context_sentences(hits, _document_titles())
    if not sentences:
        return REFUSAL_TEXT

    query_vec = np.array(embed([query])[0])
    sentence_vecs = np.array(embed(sentences))
    scores = sentence_vecs @ query_vec

    keep = sorted(np.argsort(scores)[-MAX_ANSWER_SENTENCES:])
    return " ".join(sentences[i] for i in keep)


def answer_query(
    query: str,
    strategy: str = SENTENCE_STRATEGY,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
) -> GroundedAnswer:
    hits = retrieve(query, strategy=strategy, top_k=top_k)
    top_similarity = hits[0].similarity if hits else 0.0

    if not hits or top_similarity < threshold:
        return GroundedAnswer(
            query=query,
            answer=REFUSAL_TEXT,
            refused=True,
            top_similarity=top_similarity,
            sources=[],
            hits=hits,
        )

    floor = max(threshold, top_similarity - CONTEXT_MARGIN)
    used = [h for h in hits if h.similarity >= floor]

    sources: list[str] = []
    for hit in used:
        if hit.doc_id not in sources:
            sources.append(hit.doc_id)

    return GroundedAnswer(
        query=query,
        answer=compose_answer(query, used),
        refused=False,
        top_similarity=top_similarity,
        sources=sources,
        hits=used,
    )


def calibrate(strategy: str = SENTENCE_STRATEGY) -> dict[str, float]:
    """Measure the two similarity clusters and put the threshold in the gap between them."""
    in_scores = [retrieve(q, strategy, top_k=1)[0].similarity for q in CALIBRATION_IN_SCOPE]
    out_scores = [retrieve(q, strategy, top_k=1)[0].similarity for q in CALIBRATION_OUT_OF_SCOPE]

    print("Top-1 cosine similarity, in-scope queries")
    for q, s in zip(CALIBRATION_IN_SCOPE, in_scores):
        print(f"  {s:.4f}  {q}")
    print()
    print("Top-1 cosine similarity, out-of-scope queries")
    for q, s in zip(CALIBRATION_OUT_OF_SCOPE, out_scores):
        print(f"  {s:.4f}  {q}")
    print()

    lowest_in = min(in_scores)
    highest_out = max(out_scores)
    midpoint = round((lowest_in + highest_out) / 2, 4)

    print(f"lowest in-scope    : {lowest_in:.4f}")
    print(f"highest out-of-scope: {highest_out:.4f}")
    print(f"observed gap        : {lowest_in - highest_out:.4f}")
    print(f"threshold (midpoint): {midpoint:.4f}")
    print(f"threshold in use    : {SIMILARITY_THRESHOLD:.4f}")
    if abs(midpoint - SIMILARITY_THRESHOLD) > 1e-4:
        print("  NOTE: SIMILARITY_THRESHOLD is stale, update it to the midpoint above.")
    print()

    return {"lowest_in_scope": lowest_in, "highest_out_of_scope": highest_out, "threshold": midpoint}


def main() -> None:
    print("=" * 70)
    print("TASK 4 - GROUNDED GENERATION")
    print("=" * 70)
    print(f"retrieval collection : {SENTENCE_STRATEGY}")
    print(f"top_k                : {DEFAULT_TOP_K}")
    print()

    print("-" * 70)
    print("THRESHOLD CALIBRATION")
    print("-" * 70)
    calibrate()

    print("-" * 70)
    print(f"IN-SCOPE DEMONSTRATION ({len(DEMO_IN_SCOPE)} queries)")
    print("-" * 70)
    for i, query in enumerate(DEMO_IN_SCOPE, start=1):
        result = answer_query(query)
        print(f"\n[{i}] Q: {query}")
        print(f"    top-1 similarity : {result.top_similarity:.4f}  (threshold {SIMILARITY_THRESHOLD:.4f})")
        print(f"    refused          : {result.refused}")
        print(f"    sources          : {', '.join(result.sources)}")
        print(f"    A: {result.answer}")

    print()
    print("-" * 70)
    print("OUT-OF-SCOPE DEMONSTRATION - must refuse")
    print("-" * 70)
    result = answer_query(DEMO_OUT_OF_SCOPE)
    print(f"\nQ: {DEMO_OUT_OF_SCOPE}")
    print(f"    top-1 similarity : {result.top_similarity:.4f}  (threshold {SIMILARITY_THRESHOLD:.4f})")
    print(f"    refused          : {result.refused}")
    print(f"    A: {result.answer}")


if __name__ == "__main__":
    main()
