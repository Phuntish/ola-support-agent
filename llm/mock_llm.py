"""Deterministic stand-in for a real model, wired in through CrewAI's own extension point.

`MOCK_LLM=1` (the default) means every agent in this project runs on this class.
No API key, no network, same output every time.

Two things here are not obvious, and both were confirmed by dumping the actual
prompts CrewAI sends (see transcripts/task07_crew_tool_use.txt):

1. CrewAI's ReAct system prompt contains the literal line
   "Observation: the result of the action" as part of its format instructions.
   Searching the whole conversation for "Observation:" therefore matches on the
   very first call, before any tool has run, and the agent silently answers with
   template text. This class only ever inspects assistant turns, which is where
   CrewAI appends a real tool result.

2. CrewAI does not populate the `tools` argument for a custom BaseLLM - it
   arrives as None. The tool names and their argument schemas are embedded in the
   system prompt instead, so they are parsed back out of it here. Dispatch then
   happens on the declared argument schema, never on the tool's name: a tool
   called `rag_lookup` contains the substring "lookup" and any name-matching
   dispatcher would hand it a record_id and silently get nothing back.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from crewai.llms.base_llm import BaseLLM

MOCK_ENV_FLAG = "MOCK_LLM"

# Ola ticket ids look like OLA-0001.
RECORD_ID_PATTERN = re.compile(r"\bOLA-\d{4}\b", re.IGNORECASE)

# CrewAI wraps the real task in boilerplate ("Current Task: ...", then the
# expected-output criteria and a pep talk). Only the first part is the question,
# and sending the rest to the retriever would wreck the embedding.
CURRENT_TASK_PATTERN = re.compile(
    r"Current Task:\s*(.*?)(?:\n\nThis is the expected criteria|\Z)", re.DOTALL
)

FINAL_ANSWER_PREAMBLE = "Thought: I now know the final answer\nFinal Answer: "


def mock_enabled() -> bool:
    """Mock mode is the default; a real LLM has to be asked for explicitly."""
    return os.environ.get(MOCK_ENV_FLAG, "1") not in ("0", "false", "False", "")


def parse_tool_schemas(system_prompt: str) -> dict[str, dict[str, Any]]:
    """Recover {tool_name: json-schema} from CrewAI's ReAct system prompt.

    The prompt lays each tool out as:

        Tool Name: rag_lookup
        Tool Arguments: { ...json schema... }
        Tool Description: ...
    """
    schemas: dict[str, dict[str, Any]] = {}
    for block in system_prompt.split("Tool Name:")[1:]:
        name = block.splitlines()[0].strip()
        if "Tool Arguments:" not in block:
            continue
        after_args = block.split("Tool Arguments:", 1)[1]
        raw_schema = after_args.split("Tool Description:", 1)[0].strip()
        try:
            schemas[name] = json.loads(raw_schema)
        except json.JSONDecodeError:
            schemas[name] = {}
    return schemas


def build_tool_input(schema: dict[str, Any], task_text: str) -> dict[str, Any]:
    """Fill a tool's arguments from its declared schema, not from its name."""
    properties = list((schema.get("properties") or {}).keys())

    if "record_id" in properties:
        match = RECORD_ID_PATTERN.search(task_text)
        return {"record_id": match.group(0).upper() if match else ""}

    if "query" in properties:
        return {"query": task_text.strip()}

    # Unknown schema: hand the whole task text to whatever the first string field is.
    return {properties[0]: task_text.strip()} if properties else {}


def extract_current_task(task_text: str) -> str:
    """Strip CrewAI's boilerplate back down to the question the user actually asked."""
    match = CURRENT_TASK_PATTERN.search(task_text)
    return (match.group(1) if match else task_text).strip()


def _as_message_list(messages: str | list[Any]) -> list[dict[str, str]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    normalised: list[dict[str, str]] = []
    for m in messages:
        if isinstance(m, dict):
            normalised.append({"role": str(m.get("role", "")), "content": str(m.get("content", ""))})
        else:
            normalised.append({"role": getattr(m, "role", ""), "content": str(getattr(m, "content", m))})
    return normalised


def latest_observation(messages: list[dict[str, str]]) -> str | None:
    """Return the most recent real tool result, ignoring the system template.

    This is pitfall (1) in the module docstring. Only assistant turns are read,
    because that is the only place CrewAI writes a genuine Observation.
    """
    for message in reversed(messages):
        if message["role"] != "assistant":
            continue
        if "\nObservation:" not in message["content"]:
            continue
        return message["content"].split("\nObservation:", 1)[1].strip()
    return None


class MockLLM(BaseLLM):
    """A CrewAI-compatible LLM that follows the ReAct protocol deterministically."""

    def __init__(self, model: str = "mock-ola-support", temperature: float | None = 0.0):
        super().__init__(model=model, temperature=temperature)
        self.call_count = 0
        self.tool_calls: list[dict[str, Any]] = []

    def call(
        self,
        messages: str | list[Any],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
        **kwargs: Any,
    ) -> str:
        self.call_count += 1
        conversation = _as_message_list(messages)

        system_prompt = "\n".join(m["content"] for m in conversation if m["role"] == "system")
        task_text = "\n".join(m["content"] for m in conversation if m["role"] == "user")

        # A tool has already run: turn its result into the final answer.
        observation = latest_observation(conversation)
        if observation is not None:
            return FINAL_ANSWER_PREAMBLE + observation

        schemas = parse_tool_schemas(system_prompt)
        if schemas:
            focus = extract_current_task(task_text)
            tool_name = self._select_tool(schemas, focus)
            tool_input = build_tool_input(schemas[tool_name], focus)
            self.tool_calls.append({"tool": tool_name, "input": tool_input})
            return (
                f"Thought: I need to use {tool_name} to answer this.\n"
                f"Action: {tool_name}\n"
                f"Action Input: {json.dumps(tool_input)}"
            )

        # No tools on this agent, so it is the composer: write the answer directly.
        return FINAL_ANSWER_PREAMBLE + self._compose(task_text)

    def _select_tool(self, schemas: dict[str, dict[str, Any]], task_text: str) -> str:
        """Pick a tool by what its arguments are, not by what it is called."""
        wants_record = RECORD_ID_PATTERN.search(task_text) is not None

        for name, schema in schemas.items():
            properties = (schema.get("properties") or {}).keys()
            if wants_record and "record_id" in properties:
                return name
            if not wants_record and "query" in properties:
                return name
        return next(iter(schemas))

    def _compose(self, task_text: str) -> str:
        """Assemble the composer's answer out of the material it was handed.

        Nothing is invented. CrewAI passes the upstream agents' outputs in as
        context, those outputs are JSON, and this merges them into one response
        object. Every sentence below is either copied from a tool result or is a
        fixed template filled with tool-supplied numbers.
        """
        from crew.schema import extract_json_objects

        blobs = extract_json_objects(task_text)
        retrieval = next((b for b in blobs if "top_similarity" in b), None)
        ticket = next((b for b in blobs if "escalation_score" in b), None)

        parts: list[str] = []
        payload: dict[str, Any] = {
            "answer": "",
            "sources": [],
            "retrieval_similarity": 0.0,
            "refused": False,
            "ticket": None,
            "context": [],
        }

        if retrieval is not None:
            payload["sources"] = retrieval.get("sources", [])
            payload["retrieval_similarity"] = retrieval.get("top_similarity", 0.0)
            payload["refused"] = bool(retrieval.get("refused", False))
            payload["context"] = retrieval.get("context", [])
            parts.append(str(retrieval.get("answer", "")).strip())

        if ticket is not None and "error" not in ticket:
            payload["ticket"] = {
                "record_id": ticket["record_id"],
                "status": ticket["status"],
                "resolution_time_hours": ticket["resolution_time_hours"],
                "escalation_score": ticket["escalation_score"],
                "escalate_recommended": ticket["escalate_recommended"],
                "escalation_threshold": ticket["escalation_threshold"],
            }
            verdict = "above" if ticket["escalate_recommended"] else "below"
            parts.append(
                f"Ticket {ticket['record_id']} is currently {ticket['status']} in the "
                f"{ticket['category']} queue, opened {ticket['days_since_created']} days ago with "
                f"{ticket['resolution_time_hours']} hours of handling time logged. Its escalation "
                f"score is {ticket['escalation_score']}, {verdict} the {ticket['escalation_threshold']} "
                f"threshold, so escalation is "
                f"{'recommended' if ticket['escalate_recommended'] else 'not recommended'}."
            )
        elif ticket is not None:
            parts.append(str(ticket.get("message", "That ticket could not be found.")))

        payload["answer"] = "\n\n".join(p for p in parts if p) or "I don't know."
        return json.dumps(payload, indent=2)

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 8192


def get_llm() -> BaseLLM:
    """The single place the project decides which model the agents run on."""
    if mock_enabled():
        return MockLLM()
    raise RuntimeError(
        "MOCK_LLM is disabled but no real LLM is configured. "
        "This project is designed to run fully offline; set MOCK_LLM=1."
    )
