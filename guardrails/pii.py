"""Task 10 - input-side PII masking.

Scope, as agreed in the brief: the phone number is the one field with a fixed
enough format to match reliably without a model behind it. Indian mobile numbers
are ten digits starting 6-9, optionally carrying a +91 country code and
optionally broken up with spaces or hyphens.

Rider names, pickup and drop addresses and payment details are free text with no
single matchable shape, so they are deliberately out of scope here rather than
half-matched. Every example in this repository is fabricated.

The same masker runs on what the model sees and on what the logger writes, so a
raw number cannot reach disk (Task 12).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# (?<!\d) and (?!\d) stop the pattern biting into a longer run of digits.
# The body is 6-9 then nine more digits, optionally split 5+5 the way people type it.
PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?91[\s\-]?)?[6-9]\d{4}[\s\-]?\d{5}(?!\d)"
)

PHONE_MASK = "[PHONE_REDACTED]"


@dataclass
class MaskResult:
    original: str
    masked: str
    findings: list[str]

    @property
    def fired(self) -> bool:
        return bool(self.findings)


def mask_pii(text: str) -> MaskResult:
    """Replace every phone number with a fixed token.

    Full redaction rather than a partial mask: a support transcript is retained
    for 24 months, and the last four digits are still identifying.
    """
    findings = [m.group(0) for m in PHONE_PATTERN.finditer(text)]
    return MaskResult(original=text, masked=PHONE_PATTERN.sub(PHONE_MASK, text), findings=findings)


def contains_pii(text: str) -> bool:
    return PHONE_PATTERN.search(text) is not None


CASES = [
    ("My number is 9876543210, please call me back.", True),
    ("Call me on +91 98765 43210 after 6pm.", True),
    ("Reach me at +91-9123456789 or the app.", True),
    ("Driver rang from 7012345678 and then hung up.", True),
    ("What is the status of ticket OLA-0006?", False),
    ("A refund of 1,000 rupees was approved on 15 September.", False),
    ("P1 tickets get a response in 15 minutes and resolution in 4 hours.", False),
    ("Service credits expire 90 days after issue.", False),
]


def main() -> None:
    print("=" * 70)
    print("TASK 10a - PII MASKING GUARDRAIL")
    print("=" * 70)
    print(f"pattern: {PHONE_PATTERN.pattern}")
    print(f"mask   : {PHONE_MASK}")
    print("scope  : phone numbers only. Names, addresses and payment details have no")
    print("         fixed format to match keylessly and are out of scope by design.")
    print()

    failures = 0
    for text, should_fire in CASES:
        result = mask_pii(text)
        ok = result.fired == should_fire
        failures += not ok
        print(f"[{'ok ' if ok else 'FAIL'}] expected fire={should_fire!s:<5} got={result.fired!s:<5}")
        print(f"       in : {text}")
        print(f"       out: {result.masked}")
        if result.findings:
            print(f"       matched: {result.findings}")
        print()

    print("-" * 70)
    print("DELIBERATE TEST CASE - guardrail firing on a real request")
    print("-" * 70)
    request = "I was double charged on my last trip, my number is +91 98765 43210, ticket OLA-0006."
    result = mask_pii(request)
    print(f"  raw request    : {request}")
    print(f"  what the model sees: {result.masked}")
    print(f"  what the log gets  : {result.masked}")
    print(f"  guardrail fired    : {result.fired}  (matched {result.findings})")
    print(f"  ticket id survives : {'OLA-0006' in result.masked}")
    print()
    print(f"{len(CASES) - failures}/{len(CASES)} pattern cases behaved as expected")


if __name__ == "__main__":
    main()
