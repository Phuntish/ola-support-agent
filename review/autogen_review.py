"""Task 14 - an Autogen review team that sees every draft before the customer does.

Two agents in a RoundRobinGroupChat, bounded with max_turns=2 so each speaks
exactly once:

  Policy-Compliance-Reviewer - checks the draft sentence by sentence against the
                               context the crew actually retrieved
  Final-Editor               - approves or rewrites, and must answer in the
                               ReviewVerdict schema

Three things here are easy to get wrong and all three are load-bearing:

1. `max_turns` is the real constructor parameter, not `max_iterations`. With
   MaxMessageTermination instead you would need (3), not (2), because the
   initiating task message counts as message 1.
2. Giving Final-Editor `output_content_type=ReviewVerdict` is not enough on its
   own. The team must also be constructed with
   `custom_message_types=[StructuredMessage[ReviewVerdict]]` or the run dies with
   "ValueError: Message type ... is not registered".
3. The model client has to advertise `structured_output=True` in model_info, or
   AssistantAgent refuses to accept output_content_type at all.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncGenerator, Mapping, Sequence

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

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelInfo,
    RequestUsage,
    SystemMessage,
)
from pydantic import BaseModel, Field

from guardrails.groundedness import SUPPORT_THRESHOLD, score_sentences

REVIEWER_NAME = "Policy_Compliance_Reviewer"
EDITOR_NAME = "Final_Editor"


class ReviewVerdict(BaseModel):
    """The structured output the Final-Editor has to produce."""

    approved: bool = Field(description="True if the draft goes out unchanged.")
    final_answer: str = Field(description="The answer the customer will actually see.")
    reason: str = Field(description="Why it was approved or what was changed.")


REVIEWER_SYSTEM_MESSAGE = (
    "You are the Policy Compliance Reviewer for Ola support. You are given a draft "
    "answer and the policy context it was supposed to come from. Check every sentence "
    "of the draft against that context and report any sentence the context does not "
    "support. Do not rewrite the draft yourself."
)

EDITOR_SYSTEM_MESSAGE = (
    "You are the Final Editor for Ola support. Read the reviewer's findings. If the "
    "draft is fully supported, approve it unchanged. If the reviewer flagged an "
    "unsupported sentence, remove that sentence and return the remainder. Reply with "
    "the ReviewVerdict schema: approved, final_answer, reason."
)


def build_review_task(question: str, draft: str, context: list[str]) -> str:
    """The task message the team is started with: the draft plus its retrieved context."""
    payload = {"review_payload": True, "question": question, "draft": draft, "context": context}
    return (
        f"A support draft needs review before it is sent.\n\n"
        f"QUESTION: {question}\n\n"
        f"DRAFT ANSWER:\n{draft}\n\n"
        f"RETRIEVED POLICY CONTEXT:\n" + "\n".join(f"- {c}" for c in context) + "\n\n"
        f"MACHINE READABLE COPY:\n{json.dumps(payload)}\n"
    )


class MockReviewClient(ChatCompletionClient):
    """Deterministic model client for the review team.

    Like the crew's mock, this does not pretend to reason. The reviewer's turn runs
    the same provenance check the Task 10 guardrail uses, and the editor's turn acts
    on what the reviewer found. Both are reproducible and hand-checkable.
    """

    def __init__(self) -> None:
        self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self.call_count = 0

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = [],
        tool_choice: Any = "auto",
        json_output: bool | type[BaseModel] | None = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> CreateResult:
        self.call_count += 1

        system_text = " ".join(
            str(m.content) for m in messages if isinstance(m, SystemMessage)
        )
        transcript = "\n".join(str(getattr(m, "content", "")) for m in messages)
        payload = self._payload(transcript)

        if "Final Editor" in system_text:
            content = self._edit(payload, transcript)
        else:
            content = self._review(payload)

        usage = RequestUsage(
            prompt_tokens=len(transcript) // 4, completion_tokens=len(content) // 4
        )
        self._usage = RequestUsage(
            prompt_tokens=self._usage.prompt_tokens + usage.prompt_tokens,
            completion_tokens=self._usage.completion_tokens + usage.completion_tokens,
        )
        return CreateResult(
            finish_reason="stop", content=content, usage=usage, cached=False
        )

    @staticmethod
    def _payload(transcript: str) -> dict[str, Any]:
        from crew.schema import extract_json_objects

        blocks = [b for b in extract_json_objects(transcript) if b.get("review_payload")]
        if not blocks:
            raise ValueError("review task carried no machine-readable payload")
        return blocks[0]

    @staticmethod
    def _unsupported(payload: dict[str, Any]) -> list[tuple[str, float]]:
        scored = score_sentences(payload["draft"], payload["context"])
        return [(s, score) for s, score in scored if score < SUPPORT_THRESHOLD]

    def _review(self, payload: dict[str, Any]) -> str:
        unsupported = self._unsupported(payload)
        if not unsupported:
            return (
                "FINDINGS: none. Every sentence in the draft traces back to the retrieved "
                "policy context. The draft is safe to send unchanged."
            )
        lines = [f"FINDINGS: {len(unsupported)} unsupported sentence(s)."]
        for sentence, score in unsupported:
            lines.append(f'  - provenance {score:.4f} (threshold {SUPPORT_THRESHOLD}): "{sentence}"')
        lines.append("These are not in the retrieved context and must not be sent.")
        return "\n".join(lines)

    def _edit(self, payload: dict[str, Any], transcript: str) -> str:
        from rag.chunking import split_sentences

        unsupported = self._unsupported(payload)
        draft = payload["draft"]

        if not unsupported:
            verdict = ReviewVerdict(
                approved=True,
                final_answer=draft,
                reason=(
                    "The reviewer found no unsupported sentences; every claim traces back to "
                    "the retrieved policy context, so the draft goes out unchanged."
                ),
            )
        else:
            bad = {s for s, _ in unsupported}
            kept = [s for s in split_sentences(draft) if s not in bad]
            removed = "; ".join(f'"{s}"' for s, _ in unsupported)
            verdict = ReviewVerdict(
                approved=False,
                final_answer=" ".join(kept).strip(),
                reason=(
                    f"Removed {len(unsupported)} sentence(s) the retrieved context does not "
                    f"support: {removed}"
                ),
            )
        return verdict.model_dump_json()

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = [],
        tool_choice: Any = "auto",
        json_output: bool | type[BaseModel] | None = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> AsyncGenerator[str | CreateResult, None]:
        result = await self.create(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )
        yield result

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self._usage

    def total_usage(self) -> RequestUsage:
        return self._usage

    def count_tokens(self, messages: Sequence[LLMMessage], **kwargs: Any) -> int:
        return sum(len(str(getattr(m, "content", ""))) for m in messages) // 4

    def remaining_tokens(self, messages: Sequence[LLMMessage], **kwargs: Any) -> int:
        return 8192 - self.count_tokens(messages)

    @property
    def capabilities(self) -> ModelInfo:
        return self.model_info

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            vision=False,
            function_calling=False,
            json_output=True,
            family="mock",
            # Without this, AssistantAgent rejects output_content_type outright.
            structured_output=True,
        )


def build_review_team(client: MockReviewClient | None = None) -> tuple[RoundRobinGroupChat, MockReviewClient]:
    client = client or MockReviewClient()

    reviewer = AssistantAgent(
        name=REVIEWER_NAME,
        model_client=client,
        system_message=REVIEWER_SYSTEM_MESSAGE,
        description="Checks a draft answer against the retrieved policy context.",
    )
    editor = AssistantAgent(
        name=EDITOR_NAME,
        model_client=client,
        system_message=EDITOR_SYSTEM_MESSAGE,
        description="Approves or revises the draft and returns a structured verdict.",
        output_content_type=ReviewVerdict,
    )

    team = RoundRobinGroupChat(
        participants=[reviewer, editor],
        max_turns=2,
        # Required, or the structured message the editor emits is unregistered.
        custom_message_types=[StructuredMessage[ReviewVerdict]],
    )
    return team, client


async def review_draft_async(
    question: str, draft: str, context: list[str]
) -> tuple[ReviewVerdict, list[Any]]:
    team, _ = build_review_team()
    result = await team.run(task=build_review_task(question, draft, context))

    for message in reversed(result.messages):
        if isinstance(getattr(message, "content", None), ReviewVerdict):
            return message.content, result.messages

    raise RuntimeError("the review team produced no ReviewVerdict")


def review_draft(question: str, draft: str, context: list[str]) -> tuple[ReviewVerdict, list[Any]]:
    """Synchronous wrapper, so the rest of the project does not have to be async."""
    return asyncio.run(review_draft_async(question, draft, context))
