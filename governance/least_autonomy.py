"""Task 15, application layer - least autonomy, enforced rather than documented.

The ticket tool reads customer records. Exactly one agent is allowed to hold it,
and that rule lives in a registry that raises when it is broken, instead of in a
comment that everybody trusts.

Why a registry rather than just "we only wired it to the Lookup Agent": the tool
list is an ordinary Python list, and the day somebody adds a fourth agent and
copies an existing agent's construction, nothing would stop the ticket tool
travelling with it. `tools_for()` is the only way to obtain a tool object in this
project, and it refuses any pairing that is not declared below.
"""

from __future__ import annotations

from crew.agents import COMPOSER_AGENT, LOOKUP_AGENT, RETRIEVAL_AGENT

RAG_TOOL = "rag_lookup"
TICKET_TOOL = "check_support_ticket_status"

# The whole policy, in one place. An agent that is not listed gets nothing.
TOOL_OWNERS: dict[str, str] = {
    RAG_TOOL: RETRIEVAL_AGENT,
    TICKET_TOOL: LOOKUP_AGENT,
}

AGENT_TOOLS: dict[str, list[str]] = {
    RETRIEVAL_AGENT: [RAG_TOOL],
    LOOKUP_AGENT: [TICKET_TOOL],
    COMPOSER_AGENT: [],
}

RISK_LEVEL = "Medium"


class ToolAccessError(RuntimeError):
    """Raised when an agent asks for a tool it is not the declared owner of."""


def _build(tool_name: str):
    if tool_name == RAG_TOOL:
        from tools.rag_tool import rag_lookup

        return rag_lookup
    if tool_name == TICKET_TOOL:
        from tools.ticket_status import ticket_status_tool

        return ticket_status_tool()
    raise ToolAccessError(f"unknown tool {tool_name!r}")


def grant(tool_name: str, agent_role: str):
    """Hand one tool to one agent, or refuse."""
    owner = TOOL_OWNERS.get(tool_name)
    if owner is None:
        raise ToolAccessError(f"{tool_name!r} is not a registered tool")
    if owner != agent_role:
        raise ToolAccessError(
            f"least-autonomy violation: {tool_name!r} may only be held by {owner!r}, "
            f"but {agent_role!r} asked for it"
        )
    return _build(tool_name)


def tools_for(agent_role: str) -> list:
    """Every tool this agent is entitled to. The only supported way to get tools."""
    if agent_role not in AGENT_TOOLS:
        raise ToolAccessError(f"{agent_role!r} is not a registered agent")
    return [grant(name, agent_role) for name in AGENT_TOOLS[agent_role]]


RISK_JUSTIFICATION = """\
This system is classified Medium risk on the Low / Medium / High scheme, and it sits
there for the reason the scheme names customer support tickets explicitly. It is not
Low: Low covers summarisation and transcription, where the worst outcome is a poor
rendering of text the user already has, whereas this agent states policy as fact to
riders and drivers who will act on it, and a wrong refund window or a wrong SLA
becomes a commitment Ola has to honour or visibly break. It is not High either: it
touches no medical data, makes no hiring decision, and moves no money. It reads a
ticket record and quotes published policy. The ticket tool exposes a status, a
handling time and a derived score - operational data about a support case, not
financial or health data about a person, and the one PII field in range is masked
before any component sees it. Medium is also the level the controls are built for:
a human is still in the loop for anything the policy does not already cover, the
answer is grounded in retrieved documents rather than model recall, a second agent
team reviews every draft, and the tool that reads customer records is held by one
agent under a registry that raises rather than warns."""


def main() -> None:
    from crew.crew import build_crew

    print("=" * 70)
    print("TASK 15 - APPLICATION LAYER: LEAST AUTONOMY")
    print("=" * 70)
    print("Declared ownership")
    for tool, owner in TOOL_OWNERS.items():
        print(f"  {tool:<32} -> {owner}")
    print()
    print("Per-agent entitlement")
    for agent, tools in AGENT_TOOLS.items():
        print(f"  {agent:<32} {tools or '[] (no tools at all)'}")
    print()

    print("-" * 70)
    print("ALLOWED - the Lookup Agent asks for the ticket tool")
    print("-" * 70)
    tool = grant(TICKET_TOOL, LOOKUP_AGENT)
    print(f"  granted: {tool.name}")
    print()

    print("-" * 70)
    print("BLOCKED - every other agent asking for the same tool")
    print("-" * 70)
    for agent in (RETRIEVAL_AGENT, COMPOSER_AGENT):
        try:
            grant(TICKET_TOOL, agent)
        except ToolAccessError as exc:
            print(f"  [refused] {agent}")
            print(f"            ToolAccessError: {exc}")
    print()

    print("-" * 70)
    print("BLOCKED - an unregistered agent asking for anything")
    print("-" * 70)
    try:
        tools_for("Some New Agent Somebody Added")
    except ToolAccessError as exc:
        print(f"  [refused] ToolAccessError: {exc}")
    print()

    print("-" * 70)
    print("THE LIVE CREW, INSPECTED AFTER CONSTRUCTION")
    print("-" * 70)
    crew, _ = build_crew("What is the status of ticket OLA-0006?")
    for agent in crew.agents:
        names = sorted(t.name for t in agent.tools)
        holds = TICKET_TOOL in names
        print(f"  {agent.role:<34} tools={names or '[]'}")
        print(f"  {'':<34} holds ticket tool: {holds}")
    holders = [a.role for a in crew.agents if any(t.name == TICKET_TOOL for t in a.tools)]
    print()
    print(f"  agents holding {TICKET_TOOL}: {holders}")
    print(f"  exactly one holder: {len(holders) == 1}")
    print()

    print("=" * 70)
    print(f"RISK CLASSIFICATION: {RISK_LEVEL}")
    print("=" * 70)
    print(RISK_JUSTIFICATION)


if __name__ == "__main__":
    main()
