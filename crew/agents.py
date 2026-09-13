"""Task 7 - the three agents.

Retrieval answers policy questions from the knowledge base, Lookup reads one
ticket, and Composer writes the customer-facing reply from what those two found.
Composer holds no tools at all, which is what forces it to work from their output
rather than going and fetching its own facts.
"""

from __future__ import annotations

import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
# CrewAI 1.9 asks an interactive yes/no about execution tracing on first run,
# which would hang an unattended run_all.py. Turn it off before importing.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
# CrewAI keys its "first execution" marker on the project DIRECTORY NAME
# (appdirs.user_data_dir(get_project_directory_name())), so a fresh clone
# under any other folder name is treated as a first run and prompts
# "Would you like to view your execution traces? [y/N] (20s timeout)".
# CREWAI_TESTING is the only switch that suppresses it. In crewai 1.9.3 it is
# read in exactly one functional place - _is_test_environment() in
# events/listeners/tracing/utils.py - so it gates those prompts and nothing else.
os.environ.setdefault("CREWAI_TESTING", "true")

from crewai import Agent

from llm.mock_llm import get_llm

RETRIEVAL_AGENT = "Policy Retrieval Specialist"
LOOKUP_AGENT = "Ticket Lookup Specialist"
COMPOSER_AGENT = "Support Response Composer"


def build_retrieval_agent(llm, tools: list) -> Agent:
    return Agent(
        role=RETRIEVAL_AGENT,
        goal="Find the Ola support policy that answers the customer's question, and never invent one.",
        backstory=(
            "You have read every policy document the support organisation publishes. "
            "When the knowledge base does not cover something you say so plainly "
            "instead of filling the gap yourself."
        ),
        tools=tools,
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def build_lookup_agent(llm, tools: list) -> Agent:
    return Agent(
        role=LOOKUP_AGENT,
        goal="Read the requested support ticket and report its status and escalation score.",
        backstory=(
            "You are the only agent trusted with the ticket system. You report what "
            "the record says and the score the system computed, without editorialising."
        ),
        tools=tools,
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def build_composer_agent(llm) -> Agent:
    return Agent(
        role=COMPOSER_AGENT,
        goal="Turn the policy answer and the ticket details into one clear reply for the customer.",
        backstory=(
            "You write the message the customer actually reads. You have no tools "
            "and no database access, so everything you write has to come from what "
            "the other two agents handed you."
        ),
        tools=[],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def build_agents(llm=None, retrieval_tools: list | None = None, lookup_tools: list | None = None):
    """Build all three agents. Tools are injected, never imported here - see Task 15."""
    llm = llm or get_llm()
    return {
        RETRIEVAL_AGENT: build_retrieval_agent(llm, retrieval_tools or []),
        LOOKUP_AGENT: build_lookup_agent(llm, lookup_tools or []),
        COMPOSER_AGENT: build_composer_agent(llm),
    }
