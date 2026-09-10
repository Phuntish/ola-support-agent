"""Task 12 - one JSON-Lines log entry per request.

Written as raw ASGI middleware rather than BaseHTTPMiddleware. BaseHTTPMiddleware
builds a fresh Request downstream, so a body read in the middleware is consumed
before the endpoint ever sees it; wrapping `receive` avoids that entirely.

The request text is masked with the same masker the Task 10 guardrail uses, and
it is masked *before* the line is written, so a raw phone number never touches
disk. The audit at the bottom of transcripts/task12_logging.txt greps the log
file for the number that was sent and shows nothing.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from guardrails.pii import mask_pii

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_PATH = LOG_DIR / "requests.jsonl"

# Body fields that can carry customer text, in the order we prefer them.
TEXT_FIELDS = ("question", "message", "text", "content")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def write_log_line(entry: dict[str, Any], log_path: Path = LOG_PATH) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, separators=(",", ":")) + "\n")


def extract_request_text(body: bytes) -> str:
    """Pull the customer-supplied text out of a JSON body, if there is any."""
    if not body:
        return ""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    for field in TEXT_FIELDS:
        if isinstance(payload.get(field), str):
            return payload[field]
    return ""


def build_entry(
    *,
    trace_id: str,
    channel: str,
    method: str,
    path: str,
    status: int | None,
    duration_ms: float,
    raw_text: str,
    event: str = "request",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    masked = mask_pii(raw_text)
    entry: dict[str, Any] = {
        "ts": _now_iso(),
        "trace_id": trace_id,
        "event": event,
        "channel": channel,
        "method": method,
        "path": path,
        "status": status,
        "duration_ms": round(duration_ms, 2),
        # Masked before it is written. This field is the only place customer text
        # reaches disk, and it goes through the Task 10 masker on the way.
        "request_text": masked.masked,
        "pii_masked": masked.fired,
        "pii_hits": len(masked.findings),
    }
    if extra:
        entry.update(extra)
    return entry


class JsonLinesLoggingMiddleware:
    """Log every HTTP request as one JSON line, and stamp the trace id on the response."""

    def __init__(self, app, log_path: Path = LOG_PATH):
        self.app = app
        self.log_path = log_path

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        trace_id = new_trace_id()
        scope["trace_id"] = trace_id
        started = time.perf_counter()

        body_parts: list[bytes] = []

        async def receive_wrapper():
            message = await receive()
            if message["type"] == "http.request":
                body_parts.append(message.get("body", b""))
            return message

        status_holder: dict[str, int] = {}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                headers = list(message.get("headers", []))
                headers.append((b"x-trace-id", trace_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            write_log_line(
                build_entry(
                    trace_id=trace_id,
                    channel="http",
                    method=scope.get("method", ""),
                    path=scope.get("path", ""),
                    status=status_holder.get("status"),
                    duration_ms=duration_ms,
                    raw_text=extract_request_text(b"".join(body_parts)),
                    extra={"bytes_in": sum(len(p) for p in body_parts)},
                ),
                self.log_path,
            )


def log_ws_event(
    trace_id: str,
    event: str,
    raw_text: str = "",
    duration_ms: float = 0.0,
    extra: dict[str, Any] | None = None,
) -> None:
    """WebSocket traffic does not pass through the HTTP middleware, so log it here."""
    write_log_line(
        build_entry(
            trace_id=trace_id,
            channel="websocket",
            method="WS",
            path="/ws/chat",
            status=None,
            duration_ms=duration_ms,
            raw_text=raw_text,
            event=event,
            extra=extra,
        )
    )
