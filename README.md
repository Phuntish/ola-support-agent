# Ola Domain Support Agent

**Completed the Ola (Business Operations / Customer Support) track.**

A support agent that answers Ola support-policy questions from a hand-written knowledge
base, looks up individual support tickets from a generated dataset, remembers a
conversation, is guarded against misuse, and has its answers reviewed by a second agent
team before they reach the user.

> Build status: **complete** - all 16 tasks across Parts 1-4. `run_all.py` regenerates every
> transcript in this repository and reports 24/24 steps passing.

## Everything runs with zero API keys and zero network

Every graded path runs under `MOCK_LLM=1`. There is no API key anywhere in this repository
and no runtime call leaves the machine:

- **Embeddings** come from `sentence-transformers/all-MiniLM-L6-v2` running locally on CPU.
- **The vector index** is ChromaDB with a local persistent path (`.chroma/`).
- **Language-model calls** run through a deterministic mock, never a hosted model.
- **Telemetry is off.** `run_all.py` sets `CREWAI_DISABLE_TELEMETRY=true` and
  `OTEL_SDK_DISABLED=true` in the child environment, and `rag/index.py` sets
  `ANONYMIZED_TELEMETRY=False` before importing ChromaDB and passes
  `Settings(anonymized_telemetry=False)` to the client. **Confirmed: `CREWAI_DISABLE_TELEMETRY=true`
  is set before any CrewAI import.**

The one-time exception is setup: the first run downloads the ~90 MB MiniLM model from
Hugging Face into the local cache. After that the model loads from disk and the project is
fully offline.

## Setup

This project is pinned to **Python 3.11**. That is not arbitrary: it was developed on an
Intel (x86_64) Mac, where PyTorch publishes no macOS wheels after 2.2.2, and 2.2.2 supports
Python 3.8-3.12. CrewAI additionally requires `<3.14`.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Running it

```bash
.venv/bin/python run_all.py              # regenerate every transcript in transcripts/
.venv/bin/python run_all.py task04       # regenerate one step by transcript prefix
```

Each step runs as its own subprocess and its stdout is written verbatim to
`transcripts/<name>.txt` with a header recording the module, exit code and wall clock. Every
transcript in this repository was produced by actually running the code.

Individual modules can also be run directly:

```bash
.venv/bin/python dataset.py
.venv/bin/python -m rag.generate
```

## Part 1 - Dataset and RAG core

### Task 1: dataset design (`dataset.py`)

Deterministic seeded generator producing 60 tickets. The reproducible design choices:

| Choice | Value |
| --- | --- |
| `SEED` | `20260908` |
| `N_TICKETS` | `60` |
| Category weights | Billing `0.30`, Technical Issue `0.22`, Account Access `0.16`, Product Defect `0.12`, General Inquiry `0.20` |
| Status weights | Open `0.18`, In Progress `0.20`, Escalated `0.10`, Resolved `0.34`, Closed `0.18` |
| `resolution_time_hours` range | `0.25` - `72.0` hours |
| `days_since_created` range | `0` - `30` (windowed per status) |

**Why 0.25-72 hours:** the bottom of the range is a single-touch FAQ reply and the top is the
three-business-day ceiling of the lowest severity tier in the SLA policy, so the range spans
every SLA tier the knowledge base defines without implying week-long backlogs.

Per-category hours are drawn from a triangular distribution whose mode reflects the real work
involved — Account Access resets are near-scripted (mode 3 h), Product Defects need reproduce/
file/verify (mode 24 h). Escalated tickets have their clock multiplied by 1.5 and capped at 72 h.

`escalated` is not an independent coin flip. It follows the ticket's own history: every
`Escalated` status ticket is escalated, aged live tickets (`Open`/`In Progress`, 14+ days old)
escalate with probability 0.35, and a thin 6% tail of finished tickets went through escalation
on the way. The first seed tried landed inside the required band, so no reseeding was needed.

**Measured output** (`transcripts/task01_dataset.txt`):

| | |
| --- | --- |
| Records | 60 (requirement: ≥40) |
| Per category | Billing 20, Technical Issue 14, Account Access 12, General Inquiry 10, Product Defect 4 (requirement: each ≥3) |
| Per status | Open 11, In Progress 17, Escalated 6, Resolved 15, Closed 11 (requirement: each ≥1) |
| Escalated share | **16.7%** (requirement: 10-30%) |

`dataset.py` includes a `validate()` function that asserts every one of these thresholds, and
the transcript ends in `VALIDATION PASSED`.

### Task 2: knowledge base (`kb/`)

Twelve original policy documents, 4-5 sentences each, one per required topic.
`rag/kb_report.py` verifies the count, the sentence band and topic coverage
(`transcripts/task02_knowledge_base.txt`).

### Task 3: two chunking strategies, two collections (`rag/chunking.py`, `rag/index.py`)

| Strategy | Parameters | Chunks | Chroma collection |
| --- | --- | --- | --- |
| Fixed-size with overlap | 400 chars, 80 char overlap, snapped to word boundaries | 25 | `ola_kb_fixed` |
| Sentence-based | 2 sentences per chunk, 1 sentence overlap | 37 | `ola_kb_sentence` |

Both collections are written with `collection.upsert()` and configured with
`{"hnsw:space": "cosine"}`. Embeddings are unit-normalised, so Chroma's reported distance is
exactly `1 - cosine similarity`.

Both strategies prepend the document title to the text that gets embedded. Several policies
share vocabulary ("ticket", "customer", "agent"), and the title is the cheapest way to keep a
chunk anchored to its own topic. Both collections get identical treatment so the Task 5
comparison stays fair.

### Task 4: grounded generation and the calibrated threshold (`rag/generate.py`)

Under `MOCK_LLM` there is no model that can be trusted to stay inside its context, so
generation is **extractive**: the composer only ever re-emits sentences that came back from the
retriever, ranked against the query and returned in their original order. Groundedness is
therefore a property of the code rather than something we hope a model respects.

**The refusal threshold was measured, not guessed.** Top-1 cosine similarity on the
sentence-based collection:

| In-scope query | Top-1 |
| --- | --- |
| How long are closed support tickets kept before deletion? | 0.7835 |
| When does a ticket escalate from Tier 1 to Tier 2? | 0.7369 |
| What is the first response time for a P1 ticket? | 0.6984 |
| How long does a refund take to reach the original payment method? | 0.6766 |
| How much is a standard service credit worth? | 0.5779 |
| Are billing queries handled on a public holiday? | 0.4507 |

| Out-of-scope query | Top-1 |
| --- | --- |
| Who won the cricket world cup in 2011? | 0.1613 |
| Write me a Python function that reverses a linked list. | 0.1254 |
| What is the boiling point of water at sea level? | 0.0476 |

- Lowest in-scope: **0.4507**
- Highest out-of-scope: **0.1613**
- Observed gap: **0.2894**
- **Chosen threshold: 0.3060**, the midpoint of that gap.

A tutorial default of 0.5, 0.6 or 0.7 would have refused three of the six in-scope queries
outright. `calibrate()` re-measures these numbers on every run and prints a warning if the
constant in the code has drifted from the measured midpoint.

The out-of-scope demonstration deliberately uses a *near-miss* — "What is the current share
price of Ola Electric?" — which mentions Ola and still scores only **0.2731**, below the
threshold, and correctly triggers the refusal. Five in-scope queries all answer from the
knowledge base (`transcripts/task04_grounded_generation.txt`).

A secondary `CONTEXT_MARGIN` of 0.15 keeps chunks more than 0.15 below the best hit from
contributing sentences; `top_k` always returns k chunks and the weakest is often an unrelated
policy that merely shares vocabulary.

### Task 5: chunking comparison (`rag/eval_chunks.py`)

Five queries, ground truth assigned by reading the knowledge base; two of the five need more
than one document, which is what makes recall meaningful. Retrieved chunks are mapped back to
parent documents and deduplicated before scoring. Macro-averages:

| Strategy | k | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| fixed | 3 | 0.500 | 0.800 | 0.600 |
| **sentence** | **3** | **0.800** | **0.800** | **0.767** |
| fixed | 5 | 0.433 | 0.900 | 0.580 |
| sentence | 5 | 0.467 | 0.800 | 0.580 |

**Recommendation.** At k=3, the setting generation actually runs at, the two strategies tie on
recall (0.800) but not on precision: sentence-based scores 0.800 against fixed-size's 0.500,
giving F1 0.767 versus 0.600. The per-query arithmetic shows why — fixed-size windows produce
only about two chunks per document on a knowledge base of four-sentence policies, so a top-3
request always has to reach into a second document and precision is pinned at exactly 0.500 on
all five queries, whereas sentence groups give roughly three chunks per document and can satisfy
top-3 from the correct policy alone, hitting precision 1.000 on three of the five. **The
sentence-based collection is the one deployed**: recall is no worse, and because the generator
feeds retrieved sentences straight into the answer, every spurious document is a chance for an
off-topic sentence to appear in a customer-facing reply. Raising fixed-size to k=5 buys recall
0.900 but costs precision 0.433, which is the wrong trade here.

## Part 2 - CrewAI orchestration

### Task 6: ticket lookup and escalation score (`tools/ticket_status.py`)

```
escalation_score = 0.55 * escalated + 0.45 * min(days_since_created / 30, 1)
```

Range `[0, 1]`. The flag is weighted higher because an explicit escalation is a human
decision, but the age term is weighted enough that a ticket can cross the line on staleness
alone — which is the case a bare boolean OR misses entirely.

**Threshold: 0.3030, derived from the dataset rather than typed in.** The 80th percentile of
`days_since_created` across the 60 generated tickets is 20.2 days, and an *unflagged* ticket
at that age scores `0.45 × 20.2/30 = 0.3030`. `ESCALATION_THRESHOLD` is computed at import
time from `SUPPORT_TICKETS`, so it moves if the dataset does.

What that selects (`transcripts/task06_ticket_tool.txt`):

| | |
| --- | --- |
| Score range across dataset | 0.0000 – 0.9250 |
| Lowest score of a flagged ticket | 0.5950 |
| Highest score of an unflagged ticket | 0.4350 |
| At or above threshold | 19/60 = 31.7% |
| — carrying `escalated=True` | 10 |
| — **stale but never escalated** | **9** |

Those 9 are the point of the design. `OLA-0005` is Resolved, never escalated, 26 days old, and
scores 0.390 — flagged on age alone.

### Task 7: the crew (`crew/agents.py`, `crew/crew.py`)

Three agents run sequentially via `Crew.kickoff()`:

| Agent | Tool |
| --- | --- |
| Policy Retrieval Specialist | `rag_lookup` |
| Ticket Lookup Specialist | `check_support_ticket_status` |
| Support Response Composer | none — works only from the other two |

The lookup task is only added when the question actually names a record id, so a policy-only
question doesn't hand the Lookup Agent an id it doesn't have. Transcript shows the RAG tool
alone on one query and both tools on another.

**On the two documented CrewAI pitfalls.** Both were resolved by dumping the actual prompts
CrewAI sends to a custom `BaseLLM` rather than working from assumption:

1. The ReAct system prompt genuinely does contain the literal line `Observation: the result of
   the action` as part of its format instructions. `latest_observation()` in
   `llm/mock_llm.py` therefore reads **only assistant turns**, which is the sole place CrewAI
   writes a real tool result.
2. CrewAI does **not** populate the `tools` argument for a custom `BaseLLM` — it arrives as
   `None`, and the tool schemas are embedded in the system prompt text instead. So
   `parse_tool_schemas()` recovers them from that prompt, and `build_tool_input()` dispatches
   on the declared argument schema (`record_id` vs `query`). The retrieval tool is named
   `rag_lookup` specifically so that any name-matching dispatcher would break.

A third, undocumented one: CrewAI 1.9 asks an **interactive** yes/no about execution tracing on
first run, which would hang an unattended `run_all.py`. `CREWAI_TRACING_ENABLED=false` is set
before the import.

### Task 8: session memory (`crew/memory.py`)

`InMemoryChatMessageHistory` + `RunnableWithMessageHistory`, one history per session id.
The `LangChainDeprecationWarning` is expected and left in the transcript.

Memory does real work here. A follow-up like *"how long does that take?"* has no retrievable
subject, so the chain rewrites it against the previous turn before the crew sees it:

- `task08a_memory_multi_turn.txt` — turn 2 reports `resolved using history: True` and the
  query sent to the crew is the merged question. History ends holding 6 messages.
- `task08b_memory_fresh_session.txt` — the *same* follow-up asked cold reports
  `history turns available: 0`, `resolved using history: False`, and drifts off-topic into
  service credits. Messages visible from the other session: 0.

### Task 9: structured output (`crew/schema.py`)

`SupportResponse` with `extra="forbid"`. Every crew result goes through
`validate_crew_output()` — there is no other way out of the crew. Six rejection cases are
demonstrated, including a validator that catches CrewAI's ReAct template text leaking into an
answer.

### Task 10: guardrails (`guardrails/`)

Each fires on a deliberate test; `run_guarded()` in `crew/crew.py` is the live path, ordered
`detect_injection → mask_pii → crew → check_groundedness`.

- **PII masking** — Indian mobile format, `+91` optional, spaces/hyphens tolerated. 8/8 cases,
  including negatives that must *not* match: `OLA-0006`, `1,000 rupees`, `15 minutes`.
  Full redaction rather than partial: tickets are retained 24 months and the last four digits
  still identify someone.
- **Prompt injection** — 6 labelled patterns, 10/10 cases. Includes the near-miss
  *"Ignore the driver's comment, I just want to know the SLA"*, correctly allowed through.
  End-to-end, a blocked request records **0 LLM calls** — the crew never starts.
- **Groundedness** — sentence-level cosine against retrieved context, threshold **0.95**.

The groundedness threshold needed a design change once measured, and the numbers are worth
stating. Scoring three groups against the same context:

| Group | Range |
| --- | --- |
| Copied verbatim from context | 1.0000 – 1.0000 |
| True, but reworded | 0.3933 – 0.7467 |
| Fabricated | 0.6164 – 0.7776 |

The last two **overlap** — a false claim reached 0.7776 while a true restatement sank to
0.3933 — so no threshold anywhere separates truth from falsehood. This check therefore does
not attempt to judge truth. It is a **provenance** test: copied text scores exactly 1.0,
anything written elsewhere tops out at 0.7776, and 0.95 sits in that gap with 0.05 margin
below and 0.17 above. Since the generator is extractive, every legitimate sentence is a
verbatim copy. The accepted limitation is that a correct paraphrase is refused too; with no
model available under `MOCK_LLM` to judge entailment, refusing is the safe direction to fail.

## Part 3 - Evaluation, observability, FastAPI

### Task 11: the API (`api/main.py`)

```bash
.venv/bin/uvicorn api.main:app --reload
```

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | liveness + active WebSocket count |
| `POST` | `/ask` | guarded crew answer; `AskRequest` → `AskResponse` |
| `POST` | `/add-document` | index a new policy into **both** collections |
| `WS` | `/ws/chat` | multi-turn chat |

All HTTP request/response bodies are Pydantic models, and `/ask` returns the same
`SupportResponse` the crew validates against — the schema is shared, not re-declared.
An empty question returns `422` from Pydantic before any agent runs.

**The disconnect test** (`transcripts/task11_api_endpoints.txt`) uses two concurrent clients.
Client A connects, completes two turns, then vanishes with no close handshake. That raises
`WebSocketDisconnect` inside A's receive loop, which is caught per connection. Client B — open
throughout — then completes two more turns on the same running server, including an injection
attempt that comes back `refused=True`. `/health` afterwards reports `active_websockets: 0`,
so both connections were cleaned up rather than leaked.

`POST /add-document` is demonstrated end to end: a new lost-property policy is indexed
(1 fixed chunk, 2 sentence chunks) and is immediately retrievable by a following `/ask`. The
demo then rebuilds both collections from `kb/` so the document doesn't linger into the
evaluation run.

### Task 12: structured logging (`api/logging_mw.py`)

Raw ASGI middleware, not `BaseHTTPMiddleware` — the latter builds a fresh `Request`
downstream, so a body read in the middleware is consumed before the endpoint sees it.
Wrapping `receive` avoids that. WebSocket traffic doesn't pass through HTTP middleware, so
`log_ws_event()` logs connect / message / disconnect / close separately.

One JSON line per request, to `logs/requests.jsonl`:

```json
{"ts":"...","trace_id":"c74e0d92a1b34f10","event":"request","channel":"http","method":"POST",
 "path":"/ask","status":200,"duration_ms":2417.83,
 "request_text":"I was charged twice on my trip, call me on [PHONE_REDACTED]. What is the refund rule?",
 "pii_masked":true,"pii_hits":1,"bytes_in":118}
```

The trace ID is also returned on the response as `x-trace-id`. **The masker runs before the
line is serialised**, so the number is gone before anything touches disk — it is the same
masker the Task 10 guardrail uses, not a second implementation that could drift.

The audit at the end of `transcripts/task12_logging.txt` greps the whole log file for the
number that was actually sent, in three formats:

| Format searched | Occurrences in log |
| --- | --- |
| `+91 98765 43210` (as sent) | **0** |
| `919876543210` (digits only) | **0** |
| `9876543210` (national) | **0** |
| `[PHONE_REDACTED]` | 2 |

### Task 13: LLM-as-judge evaluation (`evaluation/`)

15 queries: one per required KB topic (12), plus 3 that should not be answered — two out of
scope and one prompt injection. Each query runs through the full guarded path, evidence is
measured from the answer, and the judge prompt is rendered around that evidence.

**Averages across all 15** (`transcripts/task13b_judge_eval.txt`):

| Metric | Average |
| --- | --- |
| Accuracy | **1.0000** |
| Grounding | **1.0000** |
| Completeness | **0.9667** |
| Safety | **1.0000** |
| Overall mean | 0.9917 |

Split by scope: in-scope (n=12) completeness 0.958; out-of-scope (n=3) all four at 1.000.
Only E05 loses points, missing one of two required facts.

**On the mock judge, plainly.** With no model available it cannot reason about an answer.
What it does is apply the rubric in `JUDGE_SYSTEM_PROMPT` to evidence that was genuinely
measured: which document the answer cited, what share of its sentences trace back to
retrieved context, which required facts are present, and whether anything unsafe got out.
Every score can be re-derived by hand from the transcript.

Three of the four metrics saturate at 1.000, which is what this architecture *should*
produce — extractive generation cannot emit an ungrounded sentence, and all three
out-of-scope queries are correctly declined. Because that is indistinguishable from a judge
that rubber-stamps everything, the transcript ends with a **negative control** running the
same judge over deliberately broken output:

| Injected fault | Judge response |
| --- | --- |
| Ungrounded sentence spliced in | grounding **0.75** |
| Answer attributed to the wrong document | accuracy **0.50** |
| Out-of-scope question answered | accuracy **0.00**, completeness **0.00**, safety **0.00** |
| Phone number left unmasked | safety **0.00**, grounding 0.75 |

So the high scores reflect correct answers, not a blind judge.

## Part 4 - Resilience and governance

### Task 14: Autogen review stage (`review/autogen_review.py`)

A `RoundRobinGroupChat` of two agents with `max_turns=2`, so each speaks exactly once:

| Agent | Role |
| --- | --- |
| `Policy_Compliance_Reviewer` | scores every draft sentence against the retrieved context |
| `Final_Editor` | approves or rewrites, `output_content_type=ReviewVerdict` |

`ReviewVerdict` is `approved: bool`, `final_answer: str`, `reason: str`. The team takes the
Composer's draft **plus the context the crew actually retrieved**, which is why
`SupportResponse` carries `context` — the reviewer checks against what the composer saw, not
a fresh retrieval.

Three constructor details are load-bearing, and all three are in the code with comments:

1. `max_turns=2` is the real parameter, not `max_iterations`. With `MaxMessageTermination`
   it would need `(3)`, since the initiating task message counts as message 1.
2. `output_content_type=ReviewVerdict` on the agent is **not sufficient**. The team must
   also be built with `custom_message_types=[StructuredMessage[ReviewVerdict]]` or the run
   fails with `ValueError: Message type ... is not registered`.
3. The model client must advertise `structured_output=True` in `model_info`, or
   `AssistantAgent` rejects `output_content_type` outright.

Both cases are demonstrated on the same question:

| Transcript | approved | draft changed |
| --- | --- | --- |
| `task14a_review_approved.txt` | **True** | **False** — byte-identical passthrough |
| `task14b_review_revised.txt` | **False** | **True** — injected claim removed |

The injected claim for the revise test is written in the policy's own register
("credits 5000 rupees to the rider's wallet… wait exceeds ten minutes") so it can't be caught
by vocabulary alone. It scores **0.4915** provenance against a 0.95 threshold, the reviewer
reports it by name, and the editor removes that sentence while keeping the rest intact.

Exposed as `run_reviewed()` and on the API as `POST /ask {"review": true}`, which returns
`review_approved` and `review_reason`. It's opt-in per request rather than always-on, since
it costs two extra model turns.

### Task 15: four-layer governance (`governance/`)

**Application layer — least autonomy.** Tool ownership lives in a registry that raises,
not in a comment. `tools_for()` is the only supported way to obtain a tool object, and
`crew/crew.py` no longer imports tools directly:

```
rag_lookup                       -> Policy Retrieval Specialist
check_support_ticket_status      -> Ticket Lookup Specialist
Support Response Composer        -> [] (no tools at all)
```

Demonstrated refusals (`transcripts/task15a_least_autonomy.txt`):

```
ToolAccessError: least-autonomy violation: 'check_support_ticket_status' may only be
held by 'Ticket Lookup Specialist', but 'Policy Retrieval Specialist' asked for it
```

…the same for the Composer, and an unregistered agent gets nothing at all. The live crew is
then inspected after construction: `agents holding check_support_ticket_status:
['Ticket Lookup Specialist']`, `exactly one holder: True`.

A plain "we only wired it to the Lookup Agent" would not survive someone adding a fourth
agent by copying an existing one. The registry is what makes that a failure instead of a
silent privilege escalation.

**Risk classification: Medium.** Not Low — Low covers summarisation and transcription, where
the worst outcome is a poor rendering of text the user already has, whereas this agent states
policy as fact to riders and drivers who act on it, and a wrong refund window becomes a
commitment Ola must honour or visibly break. Not High — no medical data, no hiring decision,
no money movement; the ticket tool exposes a status, a handling time and a derived score,
which is operational data about a case rather than financial or health data about a person,
and the one PII field in range is masked before any component sees it. Medium is also the
level the controls are built for: grounded answers rather than model recall, a second agent
team reviewing every draft, and record access held by one agent under a registry that raises.

**Runtime layer — budget cap.** Checked *before* the crew starts, because rejecting after the
fact has already spent what the cap protects.

```
escalation of cost: (question_tokens + 180 context) x 5 calls
cap:                1500 tokens ($0.000375 at an assumed $0.00025/1K)
```

A 9,258-character request projects to **12,470 tokens ($0.003118)** and is rejected with
`guardrails_triggered: ['budget:request_too_large']` and **0 model calls made**. The error
names the projection, the cap, the question's own size, and what to do about it. The
transcript includes a sweep showing the cap bites between 400 and 800 characters.

The dollar rate is a stated assumption, not a measurement — nothing is billed under
`MOCK_LLM`. The token counts are measured from the text that would be sent.

### Task 16: response caching (`governance/cache.py`)

In-memory, keyed on `normalize_query()` — lowercase, collapse whitespace, strip trailing
punctuation. Normalisation deliberately stops there; stemming or synonym folding would start
merging questions that deserve different answers, and a wrong cache hit is worse than a miss.

Wired into `tools/rag_tool.py`, so the crew's retrieval path goes through it.

**Measured** (`transcripts/task16_response_cache.txt`):

| | |
| --- | --- |
| Cold call | 58.06 ms |
| Warm call | 0.03 ms |
| Speedup | ~1700x |
| `answer_query()` calls | **1** for 3 lookups |
| Hit rate | 67% |

The transcript runs a **warm-up call first and excludes it from the timings**. The first
embedding call in any process loads the model from disk and costs ~4.8 seconds; timing
against that would have reported a six-figure speedup that measures model loading rather than
the cache. A third lookup with different capitalisation, spacing and punctuation hits the same
key, and a genuinely different question still misses — so the cache is keyed on the question
rather than returning the last answer to everything.

## Task → file → transcript map

| Task | Implementation | Transcript |
| --- | --- | --- |
| 1 - Dataset design | `dataset.py` | `transcripts/task01_dataset.txt` |
| 2 - Knowledge base | `kb/*.md`, `rag/kb_report.py` | `transcripts/task02_knowledge_base.txt` |
| 3 - Chunking | `rag/chunking.py` | `transcripts/task03a_chunking.txt` |
| 3 - Embedding and indexing | `rag/index.py` | `transcripts/task03b_indexing.txt` |
| 4 - Grounded generation | `rag/generate.py` | `transcripts/task04_grounded_generation.txt` |
| 5 - Chunking comparison | `rag/eval_chunks.py` | `transcripts/task05_chunking_eval.txt` |
| 6 - Ticket tool + escalation score | `tools/ticket_status.py` | `transcripts/task06_ticket_tool.txt` |
| 7 - CrewAI crew with tools | `crew/agents.py`, `crew/crew.py`, `llm/mock_llm.py`, `tools/rag_tool.py` | `transcripts/task07_crew_tool_use.txt` |
| 8 - Memory carried across turns | `crew/memory.py` | `transcripts/task08a_memory_multi_turn.txt` |
| 8 - Fresh session, state absent | `crew/memory.py` | `transcripts/task08b_memory_fresh_session.txt` |
| 9 - Structured output schema | `crew/schema.py` | `transcripts/task09_schema_validation.txt` |
| 10 - PII masking | `guardrails/pii.py` | `transcripts/task10a_guardrail_pii.txt` |
| 10 - Prompt injection | `guardrails/injection.py` | `transcripts/task10b_guardrail_injection.txt` |
| 10 - Groundedness | `guardrails/groundedness.py` | `transcripts/task10c_guardrail_groundedness.txt` |
| 10 - All three, live path | `guardrails/pipeline.py`, `crew/crew.py` | `transcripts/task10d_guardrails_end_to_end.txt` |
| 11 - FastAPI endpoints + WebSocket | `api/main.py`, `api/demo.py` | `transcripts/task11_api_endpoints.txt` |
| 12 - JSON-Lines logging + PII audit | `api/logging_mw.py` | `transcripts/task12_logging.txt` |
| 13 - Evaluation test set | `evaluation/testset.py` | `transcripts/task13a_testset.txt` |
| 13 - LLM-as-judge scoring | `evaluation/judge.py`, `llm/mock_llm.py` | `transcripts/task13b_judge_eval.txt` |
| 14 - Review approves unchanged | `review/autogen_review.py`, `review/demo.py` | `transcripts/task14a_review_approved.txt` |
| 14 - Review revises the draft | `review/autogen_review.py`, `review/demo.py` | `transcripts/task14b_review_revised.txt` |
| 15 - Least autonomy + risk level | `governance/least_autonomy.py` | `transcripts/task15a_least_autonomy.txt` |
| 15 - Runtime budget cap | `governance/budget.py` | `transcripts/task15b_budget_cap.txt` |
| 16 - Response caching | `governance/cache.py` | `transcripts/task16_response_cache.txt` |
