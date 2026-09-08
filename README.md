# Ola Domain Support Agent

**Completed the Ola (Business Operations / Customer Support) track.**

A support agent that answers Ola support-policy questions from a hand-written knowledge
base, looks up individual support tickets from a generated dataset, remembers a
conversation, is guarded against misuse, and has its answers reviewed by a second agent
team before they reach the user.

> Build status: **Part 1 complete** (dataset, knowledge base, chunking, indexing, grounded
> generation, chunking evaluation). Parts 2-4 are in progress and their sections below are
> marked accordingly.

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

_In progress._

## Part 3 - Evaluation, observability, FastAPI

_In progress._

## Part 4 - Resilience and governance

_In progress._

## Task → file → transcript map

| Task | Implementation | Transcript |
| --- | --- | --- |
| 1 - Dataset design | `dataset.py` | `transcripts/task01_dataset.txt` |
| 2 - Knowledge base | `kb/*.md`, `rag/kb_report.py` | `transcripts/task02_knowledge_base.txt` |
| 3 - Chunking | `rag/chunking.py` | `transcripts/task03a_chunking.txt` |
| 3 - Embedding and indexing | `rag/index.py` | `transcripts/task03b_indexing.txt` |
| 4 - Grounded generation | `rag/generate.py` | `transcripts/task04_grounded_generation.txt` |
| 5 - Chunking comparison | `rag/eval_chunks.py` | `transcripts/task05_chunking_eval.txt` |
| 6-16 | _in progress_ | |
