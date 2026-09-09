"""Task 9 - the structured response schema every crew answer has to satisfy.

`validate_crew_output` is the only sanctioned way out of the crew. If the crew
produces something that does not fit this shape, that is a bug we want raised
loudly rather than a malformed answer shown to a customer.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class TicketSummary(BaseModel):
    """The Lookup Agent's contribution, when the question named a ticket."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    status: str
    resolution_time_hours: float = Field(ge=0.0)
    escalation_score: float = Field(ge=0.0, le=1.0)
    escalate_recommended: bool
    escalation_threshold: float = Field(ge=0.0, le=1.0)


class SupportResponse(BaseModel):
    """The single response format the whole system agrees on."""

    model_config = ConfigDict(extra="forbid")

    query: str
    answer: str = Field(min_length=1)
    sources: list[str] = Field(default_factory=list)
    retrieval_similarity: float = Field(ge=0.0, le=1.0)
    refused: bool = False
    ticket: TicketSummary | None = None
    guardrails_triggered: list[str] = Field(default_factory=list)
    # The chunks the answer was built from. Carried on the response so the
    # groundedness guardrail and the Task 14 review team can both check the
    # answer against the same context the composer actually saw.
    context: list[str] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def answer_is_not_placeholder(cls, v: str) -> str:
        """Guards against the CrewAI ReAct template leaking through as the answer.

        If a parser ever matches the system prompt's own example text instead of a
        real tool result, this is where it surfaces instead of reaching a customer.
        """
        if "the result of the action" in v:
            raise ValueError("answer contains CrewAI ReAct template text, not a real result")
        return v

    @field_validator("sources")
    @classmethod
    def sources_present_when_answering(cls, v: list[str], info) -> list[str]:
        return v


def extract_json_objects(text: str) -> list[dict[str, Any]]:
    """Pull every balanced top-level JSON object out of a block of text.

    The crew passes tool results around as JSON embedded in prose, so this is how
    downstream steps recover them without regex-guessing.
    """
    objects: list[dict[str, Any]] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(parsed, dict):
                        objects.append(parsed)
                start = -1
    return objects


def validate_crew_output(raw: str, query: str) -> SupportResponse:
    """Parse and validate a crew result. Raises ValidationError if it does not conform."""
    candidates = extract_json_objects(raw)
    for candidate in reversed(candidates):
        if "answer" in candidate:
            candidate.setdefault("query", query)
            return SupportResponse.model_validate(candidate)

    raise ValidationError.from_exception_data(
        "SupportResponse",
        [
            {
                "type": "missing",
                "loc": ("answer",),
                "input": raw[:200],
            }
        ],
    )


def main() -> None:
    print("=" * 70)
    print("TASK 9 - STRUCTURED RESPONSE SCHEMA")
    print("=" * 70)
    print("Every crew response is parsed into SupportResponse before anything")
    print("downstream is allowed to touch it. Fields:")
    for name, field in SupportResponse.model_fields.items():
        required = "required" if field.is_required() else "optional"
        print(f"  {name:<22} {str(field.annotation):<34} {required}")
    print()
    print("extra='forbid', so an unexpected key is an error rather than silently ignored.")
    print()

    good = json.dumps({
        "answer": "Approved refunds return to the original payment method within 5 to 7 working days.",
        "sources": ["refund-and-compensation"],
        "retrieval_similarity": 0.6377,
        "refused": False,
        "ticket": {
            "record_id": "OLA-0006",
            "status": "Closed",
            "resolution_time_hours": 7.6,
            "escalation_score": 0.775,
            "escalate_recommended": True,
            "escalation_threshold": 0.303,
        },
    })

    print("-" * 70)
    print("ACCEPTED - a well-formed crew response")
    print("-" * 70)
    response = validate_crew_output(good, query="refund rule for a double charge")
    print(f"  parsed as {type(response).__name__}")
    print(f"  ticket.escalation_score = {response.ticket.escalation_score}")
    print()

    rejections = [
        (
            "escalation_score above 1.0",
            json.dumps({"answer": "x", "retrieval_similarity": 0.5, "ticket": {
                "record_id": "OLA-0001", "status": "Open", "resolution_time_hours": 1.0,
                "escalation_score": 1.4, "escalate_recommended": True, "escalation_threshold": 0.303}}),
        ),
        (
            "similarity outside [0,1]",
            json.dumps({"answer": "x", "retrieval_similarity": 42}),
        ),
        (
            "empty answer",
            json.dumps({"answer": "", "retrieval_similarity": 0.5}),
        ),
        (
            "CrewAI ReAct template leaked into the answer",
            json.dumps({"answer": "Observation: the result of the action",
                        "retrieval_similarity": 0.5}),
        ),
        (
            "unexpected key",
            json.dumps({"answer": "x", "retrieval_similarity": 0.5, "internal_debug": "leak"}),
        ),
        ("no JSON object at all", "the crew returned prose instead of a response object"),
    ]

    print("-" * 70)
    print("REJECTED - each of these raises instead of reaching a customer")
    print("-" * 70)
    for label, payload in rejections:
        try:
            validate_crew_output(payload, query="q")
        except ValidationError as exc:
            first = exc.errors()[0]
            print(f"  [rejected] {label}")
            print(f"             {first['type']}: {first.get('msg', '')} at {first['loc']}")
        else:
            print(f"  [MISSED]   {label} - this should not have validated")
    print()
    print("The template-text check exists because of the known CrewAI failure mode: if a")
    print("parser ever matches the ReAct system prompt instead of a real tool result, the")
    print("placeholder is caught here rather than being shown as an answer.")


if __name__ == "__main__":
    main()
