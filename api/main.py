"""Task 11 - the FastAPI surface: two HTTP endpoints and one WebSocket chat endpoint.

Run it with:

    .venv/bin/uvicorn api.main:app --reload

Everything still runs under MOCK_LLM with no API key. The WebSocket endpoint is
the interesting one: a client that vanishes mid-conversation raises
WebSocketDisconnect inside the receive loop, and that has to be caught per
connection so the server keeps serving everybody else.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("MOCK_LLM", "1")

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from api.logging_mw import JsonLinesLoggingMiddleware, log_ws_event, new_trace_id
from crew.crew import run_guarded, run_reviewed
from crew.memory import ask as ask_with_memory
from crew.schema import SupportResponse
from rag.chunking import Document
from rag.index import index_document

app = FastAPI(
    title="Ola Domain Support Agent",
    description="Grounded support-policy answers and ticket lookups, running fully offline.",
    version="1.0.0",
)
app.add_middleware(JsonLinesLoggingMiddleware)

# Every live WebSocket, so we can show the server still has other clients after
# one of them drops.
ACTIVE_CONNECTIONS: dict[str, WebSocket] = {}


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(
        default=None, description="Supply one to keep conversation memory across calls."
    )
    review: bool = Field(
        default=False,
        description="Run the Autogen review team over the draft before returning it.",
    )


class AskResponse(BaseModel):
    trace_id: str
    response: SupportResponse
    llm_calls: int
    tools_used: list[str] = Field(default_factory=list)
    review_approved: bool | None = Field(
        default=None, description="Set when the Task 14 review stage ran."
    )
    review_reason: str | None = None


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)


class AddDocumentResponse(BaseModel):
    doc_id: str
    chunks_indexed: dict[str, int]
    message: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "MOCK_LLM", "active_websockets": str(len(ACTIVE_CONNECTIONS))}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a support question through the guarded crew path."""
    if request.session_id:
        # Memory-backed path; returns the answer text only.
        answer = ask_with_memory(request.question, request.session_id)
        return AskResponse(
            trace_id=new_trace_id(),
            response=SupportResponse(
                query=request.question, answer=answer, retrieval_similarity=0.0
            ),
            llm_calls=0,
        )

    if request.review:
        run, verdict = run_reviewed(request.question)
        return AskResponse(
            trace_id=new_trace_id(),
            response=run.response,
            llm_calls=run.llm_calls,
            tools_used=run.tools_used,
            review_approved=None if verdict is None else verdict.approved,
            review_reason=None if verdict is None else verdict.reason,
        )

    run = run_guarded(request.question)
    return AskResponse(
        trace_id=new_trace_id(),
        response=run.response,
        llm_calls=run.llm_calls,
        tools_used=run.tools_used,
    )


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document(request: AddDocumentRequest) -> AddDocumentResponse:
    """Add or replace a knowledge-base document in both Chroma collections."""
    try:
        counts = index_document(
            Document(doc_id=request.doc_id, title=request.title, text=request.text)
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a 500
        raise HTTPException(status_code=500, detail=f"indexing failed: {exc}") from exc

    return AddDocumentResponse(
        doc_id=request.doc_id,
        chunks_indexed=counts,
        message=f"Indexed {request.doc_id} into both collections.",
    )


@app.websocket("/ws/chat")
async def chat(websocket: WebSocket) -> None:
    """Multi-turn chat. One client dropping must not disturb any other client."""
    await websocket.accept()
    trace_id = new_trace_id()
    ACTIVE_CONNECTIONS[trace_id] = websocket
    log_ws_event(trace_id, "ws_connect", extra={"active": len(ACTIVE_CONNECTIONS)})

    turns = 0
    try:
        while True:
            question = await websocket.receive_text()
            turns += 1
            started = time.perf_counter()

            run = run_guarded(question)
            duration_ms = (time.perf_counter() - started) * 1000

            log_ws_event(trace_id, "ws_message", raw_text=question, duration_ms=duration_ms,
                         extra={"turn": turns})
            await websocket.send_json(
                {
                    "trace_id": trace_id,
                    "turn": turns,
                    "answer": run.response.answer,
                    "sources": run.response.sources,
                    "refused": run.response.refused,
                    "guardrails_triggered": run.response.guardrails_triggered,
                }
            )
    except WebSocketDisconnect:
        # The client went away. Log it, clean up this connection only, and return
        # normally so the server carries on serving everyone else.
        log_ws_event(trace_id, "ws_disconnect", extra={"turns_completed": turns})
    finally:
        ACTIVE_CONNECTIONS.pop(trace_id, None)
        log_ws_event(trace_id, "ws_closed", extra={"active": len(ACTIVE_CONNECTIONS)})
