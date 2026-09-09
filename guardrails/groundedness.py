"""Task 10 - output-side groundedness check.

The generator in rag/generate.py is extractive, so in normal operation every
sentence it emits is already lifted from retrieved context. This guardrail is the
check that proves it, and more importantly it is the thing that catches an answer
that came from somewhere else - a revised draft, a cached entry that no longer
matches, or a real LLM swapped in behind the env flag.

Each sentence of the answer is scored against the retrieved context by cosine
similarity, and the threshold is set from measured clusters rather than assumed.

What the measurement actually showed matters, because it shaped the design. I
scored three groups of sentences against the same retrieved context: sentences
copied verbatim, sentences that restate a real policy in different words, and
sentences that are simply false. The copied group sits at 1.0. The other two
groups overlap almost entirely - a fabricated claim reached 0.778 while a true
paraphrase sank to 0.393 - so no threshold anywhere can tell a reworded truth
from an on-topic lie.

So this check is deliberately not a truth test. It is a provenance test: "did
this sentence come out of the retrieved documents, or from somewhere else?"
With an extractive generator every legitimate sentence is a verbatim copy and
scores 1.0, which puts the threshold at 0.95 with enormous headroom. Anything
written elsewhere - a fabricated claim, a revised draft, a stale cache entry, or
a real LLM swapped in behind the env flag - falls far below it and is refused.

The honest limitation: a correct paraphrase is also refused. Under MOCK_LLM,
with no model available to judge entailment, that is the safe direction to fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rag.chunking import split_sentences
from rag.index import embed

# Copied sentences score exactly 1.0; every sentence written outside the context
# measured at 0.778 or below. 0.95 sits in that gap with room on both sides.
# See the calibration block in transcripts/task10c_groundedness.txt.
SUPPORT_THRESHOLD = 0.95

REFUSAL_TEXT = (
    "I can't stand behind that answer. Part of it is not supported by the Ola support "
    "policy documents retrieved for this question, so it has been withheld rather than "
    "shown to the customer."
)


@dataclass
class GroundednessVerdict:
    answer: str
    per_sentence: list[tuple[str, float]] = field(default_factory=list)
    threshold: float = SUPPORT_THRESHOLD

    @property
    def unsupported(self) -> list[tuple[str, float]]:
        return [(s, score) for s, score in self.per_sentence if score < self.threshold]

    @property
    def grounded(self) -> bool:
        return not self.unsupported

    @property
    def worst_score(self) -> float:
        return min((score for _, score in self.per_sentence), default=1.0)


def score_sentences(answer: str, context: list[str]) -> list[tuple[str, float]]:
    """Best cosine similarity of each answer sentence against any context sentence."""
    answer_sentences = split_sentences(answer)
    context_sentences: list[str] = []
    for chunk in context:
        context_sentences.extend(split_sentences(chunk))

    if not answer_sentences or not context_sentences:
        return [(s, 0.0) for s in answer_sentences]

    answer_vecs = np.array(embed(answer_sentences))
    context_vecs = np.array(embed(context_sentences))
    similarity = answer_vecs @ context_vecs.T

    return [(s, round(float(similarity[i].max()), 4)) for i, s in enumerate(answer_sentences)]


def check_groundedness(
    answer: str, context: list[str], threshold: float = SUPPORT_THRESHOLD
) -> GroundednessVerdict:
    return GroundednessVerdict(
        answer=answer, per_sentence=score_sentences(answer, context), threshold=threshold
    )


def main() -> None:
    from rag.generate import answer_query

    print("=" * 70)
    print("TASK 10c - OUTPUT-SIDE GROUNDEDNESS GUARDRAIL")
    print("=" * 70)

    question = "A rider was charged twice for the same trip, what is the refund rule?"
    result = answer_query(question)
    context = [h.text for h in result.hits]

    print(f"question: {question}")
    print(f"retrieved {len(context)} chunks from {result.sources}")
    print()

    print("-" * 70)
    print("CASE 1 - the real generated answer (should pass)")
    print("-" * 70)
    clean = check_groundedness(result.answer, context)
    for sentence, score in clean.per_sentence:
        print(f"  {score:.4f}  {sentence[:88]}")
    print(f"  -> grounded={clean.grounded}, worst sentence score {clean.worst_score:.4f}")
    print()

    print("-" * 70)
    print("CASE 2 - DELIBERATE TEST: an ungrounded claim spliced into the answer")
    print("-" * 70)
    fabricated = (
        "Ola refunds every disputed fare automatically within 30 minutes and adds a "
        "5000 rupee apology bonus to the rider's wallet."
    )
    tampered = result.answer + " " + fabricated
    print(f"  injected sentence: {fabricated}")
    print()
    dirty = check_groundedness(tampered, context)
    for sentence, score in dirty.per_sentence:
        flag = "  <-- UNSUPPORTED" if score < dirty.threshold else ""
        print(f"  {score:.4f}  {sentence[:88]}{flag}")
    print()
    print(f"  -> grounded={dirty.grounded}")
    print(f"  -> guardrail fired on {len(dirty.unsupported)} sentence(s)")
    print(f"  -> reply shown to customer: {REFUSAL_TEXT}")
    print()

    print("-" * 70)
    print("THRESHOLD CALIBRATION - three groups scored against the same context")
    print("-" * 70)

    fabricated = [
        "Ola refunds every disputed fare automatically within 30 minutes and adds a "
        "5000 rupee apology bonus to the rider's wallet.",
        "Riders receive a full refund plus double the fare whenever a driver arrives late.",
        "All refunds are approved instantly by the front-line agent with no upper limit.",
        "Ola guarantees refunds within 24 hours for every complaint raised in the app.",
    ]
    paraphrased = [
        "A refund is given if the rider paid for a journey that never actually happened.",
        "The money goes back to the card in about a week, or straight to the wallet if "
        "the rider wants it faster.",
        "An agent can sign off a refund themselves up to one thousand rupees.",
    ]

    copied = [score for _, score in clean.per_sentence]
    fab_scores = [score for _, score in score_sentences(" ".join(fabricated), context)]
    para_scores = [score for _, score in score_sentences(" ".join(paraphrased), context)]

    print(f"  copied verbatim from context   : {min(copied):.4f} - {max(copied):.4f}")
    print(f"  true, but reworded             : {min(para_scores):.4f} - {max(para_scores):.4f}")
    print(f"  fabricated                     : {min(fab_scores):.4f} - {max(fab_scores):.4f}")
    print()
    print("  The reworded and fabricated groups overlap: a false claim reached")
    print(f"  {max(fab_scores):.4f} while a true restatement sank to {min(para_scores):.4f}. No threshold")
    print("  separates them, so this check does not attempt to judge truth.")
    print()
    print("  It tests provenance instead. Copied text scores exactly 1.0 and everything")
    print(f"  written outside the context tops out at {max(max(fab_scores), max(para_scores)):.4f}, so:")
    print(f"    SUPPORT_THRESHOLD = {SUPPORT_THRESHOLD}")
    print(f"    margin below it   : {1.0 - SUPPORT_THRESHOLD:.4f} from copied text")
    print(f"    margin above it   : {SUPPORT_THRESHOLD - max(max(fab_scores), max(para_scores)):.4f} from anything else")
    print()
    print("  Accepted limitation: a correct paraphrase is refused too. With no model")
    print("  available under MOCK_LLM to judge entailment, refusing is the safe direction.")


if __name__ == "__main__":
    main()
