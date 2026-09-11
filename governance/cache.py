"""Task 16 - in-memory cache for grounded generation, keyed on the normalised query.

Grounded generation is the expensive step in a request: it embeds the query,
searches both, and embeds every candidate sentence again to rank them. Two
customers asking the same policy question in slightly different words should not
pay for that twice.

The key is normalised rather than raw, so "What is the refund policy?" and
"what is the refund policy" are one entry. Normalisation stops at punctuation and
whitespace on purpose - stemming or synonym folding would start merging questions
that deserve different answers, and a wrong cache hit is worse than a miss.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from rag.generate import GroundedAnswer, answer_query

_WHITESPACE = re.compile(r"\s+")
_TRAILING_PUNCT = re.compile(r"[?!.,;:\s]+$")


def normalize_query(query: str) -> str:
    """Lowercase, collapse internal whitespace, drop trailing punctuation."""
    return _TRAILING_PUNCT.sub("", _WHITESPACE.sub(" ", query.strip().lower()))


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    underlying_calls: int = 0
    time_saved_ms: float = 0.0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


@dataclass
class GroundedAnswerCache:
    entries: dict[str, GroundedAnswer] = field(default_factory=dict)
    stats: CacheStats = field(default_factory=CacheStats)
    # How long the last real generation took, used to report what a hit saved.
    _last_miss_ms: float = 0.0

    def get(self, query: str) -> GroundedAnswer:
        key = normalize_query(query)

        if key in self.entries:
            self.stats.hits += 1
            self.stats.time_saved_ms += self._last_miss_ms
            return self.entries[key]

        started = time.perf_counter()
        result = answer_query(query)
        elapsed_ms = (time.perf_counter() - started) * 1000

        self.stats.misses += 1
        self.stats.underlying_calls += 1
        self._last_miss_ms = elapsed_ms
        self.entries[key] = result
        return result

    def clear(self) -> None:
        self.entries.clear()
        self.stats = CacheStats()


# One cache for the process. The API and the crew both go through it.
CACHE = GroundedAnswerCache()


def cached_answer_query(query: str) -> GroundedAnswer:
    return CACHE.get(query)


def main() -> None:
    print("=" * 70)
    print("TASK 16 - RESPONSE CACHING FOR GROUNDED GENERATION")
    print("=" * 70)
    print(f"  key            : normalize_query(query)")
    print(f"  normalisation  : lowercase, collapse whitespace, strip trailing punctuation")
    print()

    print("-" * 70)
    print("NORMALISATION - which queries collapse to one key")
    print("-" * 70)
    variants = [
        "What is the standard service credit?",
        "what is the standard service credit",
        "  What   is the  standard service credit?  ",
        "WHAT IS THE STANDARD SERVICE CREDIT!!",
        "How long do service credits last?",
    ]
    for v in variants:
        print(f"  {v!r:<48} -> {normalize_query(v)!r}")
    print()

    # Warm the sentence encoder before timing anything. The first embedding call in
    # a process loads the model from disk, which costs seconds and has nothing to do
    # with the cache. Timing against that would report a speedup of six figures and
    # be worthless as evidence.
    print("-" * 70)
    print("WARM-UP (excluded from the measurements below)")
    print("-" * 70)
    started = time.perf_counter()
    CACHE.get("warm up the sentence encoder before timing anything")
    warmup_ms = (time.perf_counter() - started) * 1000
    print(f"  first call in the process: {warmup_ms:8.2f} ms  <- mostly loading the model")
    print("  the model is now resident, so what follows measures the cache and nothing else")
    print()

    CACHE.clear()
    question = "What is the standard service credit?"

    print("-" * 70)
    print("BEFORE / AFTER ON A REPEATED QUERY")
    print("-" * 70)
    print(f"  query: {question}")
    print()
    print(f"  starting state: {CACHE.stats.hits} hits, {CACHE.stats.misses} misses, "
          f"{CACHE.stats.underlying_calls} calls to answer_query()")
    print()

    started = time.perf_counter()
    first = CACHE.get(question)
    first_ms = (time.perf_counter() - started) * 1000
    print(f"  call 1 (cold)  : {first_ms:8.2f} ms")
    print(f"                   hits={CACHE.stats.hits} misses={CACHE.stats.misses} "
          f"answer_query calls={CACHE.stats.underlying_calls}")
    print(f"                   sources={first.sources}")
    print()

    started = time.perf_counter()
    second = CACHE.get(question)
    second_ms = (time.perf_counter() - started) * 1000
    print(f"  call 2 (same)  : {second_ms:8.2f} ms")
    print(f"                   hits={CACHE.stats.hits} misses={CACHE.stats.misses} "
          f"answer_query calls={CACHE.stats.underlying_calls}")
    print()

    started = time.perf_counter()
    third = CACHE.get("  what is the STANDARD service credit!!  ")
    third_ms = (time.perf_counter() - started) * 1000
    print(f"  call 3 (variant spelling, same normalised key)")
    print(f"                 : {third_ms:8.2f} ms")
    print(f"                   hits={CACHE.stats.hits} misses={CACHE.stats.misses} "
          f"answer_query calls={CACHE.stats.underlying_calls}")
    print()

    print("-" * 70)
    print("EVIDENCE")
    print("-" * 70)
    print(f"  answer_query() was called {CACHE.stats.underlying_calls} time(s) for "
          f"{CACHE.stats.lookups} lookups")
    print(f"  cache hits           : {CACHE.stats.hits}")
    print(f"  hit rate             : {CACHE.stats.hit_rate:.0%}")
    speedup = first_ms / second_ms if second_ms else float("inf")
    print(f"  cold call            : {first_ms:.2f} ms")
    print(f"  warm call            : {second_ms:.2f} ms")
    print(f"  speedup              : {speedup:.0f}x")
    print(f"  identical result     : {second.answer == first.answer}")
    print(f"  variant same result  : {third.answer == first.answer}")
    print()

    print("-" * 70)
    print("A DIFFERENT QUESTION STILL MISSES")
    print("-" * 70)
    other = "How often do we send updates during an outage?"
    started = time.perf_counter()
    result = CACHE.get(other)
    other_ms = (time.perf_counter() - started) * 1000
    print(f"  query: {other}")
    print(f"  {other_ms:8.2f} ms, hits={CACHE.stats.hits} misses={CACHE.stats.misses} "
          f"answer_query calls={CACHE.stats.underlying_calls}")
    print(f"  sources={result.sources}")
    print()
    print("  So the cache is keyed on the question, not returning the last answer to")
    print("  everything, which is the failure mode worth guarding against.")


if __name__ == "__main__":
    main()
