"""The retrieval tool the Retrieval Agent carries.

The name `rag_lookup` is deliberate. It contains the substring "lookup", so any
dispatcher that routes tool calls by name-matching will confuse it with
`check_support_ticket_status` and hand it a record_id it cannot use. The mock LLM
dispatches on the declared argument schema instead - see llm/mock_llm.py.
"""

from __future__ import annotations

import json

from crewai.tools import tool

from governance.cache import cached_answer_query


def rag_lookup_payload(query: str) -> dict:
    """Retrieve and ground an answer, returned as a plain dict for reuse off-crew.

    Goes through the Task 16 cache, so a repeated question skips the embedding
    and ranking work entirely.
    """
    result = cached_answer_query(query)
    return {
        "answer": result.answer,
        "refused": result.refused,
        "top_similarity": result.top_similarity,
        "sources": result.sources,
        "context": [h.text for h in result.hits],
    }


@tool("rag_lookup")
def rag_lookup(query: str) -> str:
    """Search the Ola support policy knowledge base and return the grounded policy text.

    Use this for any question about support policy: SLAs, refunds, escalation,
    service credits, outages, data retention and so on. Returns JSON with the
    answer, the source documents and the retrieval similarity.
    """
    return json.dumps(rag_lookup_payload(query), indent=2)
