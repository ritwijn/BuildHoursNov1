# Architecture

**Product:** Groww Mutual Fund FAQ Assistant  
**Version:** 1.0  
**Date:** 1 November 2026  
**Source of truth:** `PRD.md`  
**Status:** Prototype architecture (RAG demo)

This document describes how the system is built so each RAG stage is inspectable, configurable, and constrained to facts-only answers with a single citation.

---

## 1. Design goals

| Goal | Architectural implication |
|------|---------------------------|
| Facts only, one citation, no advice | Guardrails sit **before** retrieval (intent) and **after** retrieval (prompt + post-check). Generation is grounded in retrieved chunks only. |
| Demo-able RAG | Offline ingest and online query write **stage artifacts** to disk (and Chroma). |
| Config-driven models | Embedding model, Chroma paths, `top_k`, chunking, and LLM provider/model come from config / `.env`. |
| Small corpus, fail closed | Unknown schemes / missing facts → “not in sources” or closest official page; no parametric invention. |
| No PII | Input sanitizer rejects PAN / Aadhaar / account / OTP / email / phone **before** logging or LLM. |
| No performance math | Returns/comparison intent is classified and routed to a factsheet link, not computed. |

Non-goals from the PRD stay out of this design: auth, multi-tenant hosting, live NAV feeds, all-AMC coverage.

---

## 2. System context

```
┌──────────────┐     public HTTP      ┌─────────────────────────┐
│ Groww public │ ───────────────────► │ Ingest pipeline         │
│ pages + extra│                      │ (scrape → load → chunk  │
│              │                      │  → embed)               │
│ official URLs│                      └───────────┬─────────────┘
└──────────────┘                                  │
                                                  ▼
┌──────────────┐     question         ┌─────────────────────────┐
│ User (tiny   │ ───────────────────► │ Query pipeline          │
│ UI)          │ ◄── answer + 1 URL   │ (guard → retrieve → LLM)│
└──────────────┘     + last-updated   └───────────┬─────────────┘
                                                  │
                     ┌────────────────────────────┼────────────────┐
                     ▼                            ▼                ▼
              data/raw, chunks              ChromaDB         Groq / optional
              retrieval_logs                (vectors)        LLM providers
```

**Actors**

- **Retail / support user:** asks a question in the UI.
- **Demo reviewer:** inspects `data/` artifacts and optional pipeline view.
- **Operator:** sets `.env` / config and re-runs ingest.

---

## 3. Two pipelines

The architecture is split so ingest can be re-run without serving, and serving never mutates the corpus except via an explicit ingest job.

### 3.1 Offline ingest (build the corpus)

```
sources.yaml / config
        │
        ▼
   ┌─────────┐    data/raw/{id}.html|.pdf + .txt + .meta.json
   │ Scraper │ ──────────────────────────────────►
   └────┬────┘
        ▼
   ┌─────────┐    Documents (sections + metadata), read from data/raw/
   │ Document│  clean, validate; skip sources with fetch errors
   │ loader  │
   └────┬────┘
        ▼
   ┌─────────┐    data/chunks/chunks.jsonl
   │ Chunker │ ──────────────────────────────────►
   └────┬────┘
        ▼
   ┌─────────┐    embeddings (in-memory then Chroma)
   │ Embedder│  sentence-transformers/all-MiniLM-L6-v2
   └────┬────┘
        ▼
   ┌─────────┐    data/chroma/  + ingest_manifest.json
   │ Chroma  │ ──────────────────────────────────►
   └─────────┘
```

**Done when:** the five required Groww URLs (plus extra official URLs) exist as raw files, chunks, and a Chroma collection; `last_ingest_at` is recorded.

### 3.2 Online query (answer a question)

```
UI question
    │
    ▼
┌──────────────┐  reject / strip
│ PII filter   │ ──────────────────► user-facing error (do not log PII)
└──────┬───────┘
       ▼
┌──────────────┐
│ Intent router│  advice | returns | factual | out-of-scope
└──────┬───────┘
       │
       ├─ advice ──► refusal template + SEBI/AMFI investor-education URL
       ├─ returns ─► no compute; factsheet / official page link
       │
       ▼ factual
┌──────────────┐
│ Query embed  │  same MiniLM model as ingest
└──────┬───────┘
       ▼
┌──────────────┐    data/retrieval_logs/{query_id}.json
│ Chroma query │ ──────────────────────────────────────►
│ top_k        │
└──────┬───────┘
       ▼
┌──────────────┐
│ Prompt build │  system rules + retrieved chunks only
└──────┬───────┘
       ▼
┌──────────────┐
│ LLM          │  Groq (model from .env); pluggable providers
└──────┬───────┘
       ▼
┌──────────────┐
│ Post-check   │  ≤3 sentences, exactly one URL from retrieved set,
│              │  last-updated stamp, no invented citation
└──────────────┘
       ▼
   UI answer
```

---

## 4. Logical components

| Component | Responsibility | Reads | Writes |
|-----------|----------------|-------|--------|
| **Config loader** | Merge `config` + `.env`; expose embedding, Chroma, chunking, `top_k`, LLM. | config files, env | — |
| **Source registry** | Canonical list of 15–25 official public URLs (id, title, url, type, scheme, optional PDF `pages`): the 5 required Groww pages from PRD §5.1 plus HDFC MF KIM/SID/factsheets, Groww help pages, and SEBI/AMFI investor education. Nothing outside this list is fetched or cited. | `config/sources.yaml` (or equivalent) | — |
| **Scraper** | Fetch allowlisted public pages; extract text into sections; store HTML/text + metadata. | URLs | `data/raw/` |
| **Document loader** | Read `data/raw/` back into `Document` objects (sections + source metadata); normalize text; drop empty/duplicate sections; fail if a required source has no content. Chunking always starts from disk, so it can re-run without re-fetching. | `data/raw/`, source registry | in-memory `Document` list |
| **Chunker** | Split documents with a documented strategy (heading/section-aware preferred). | `Document` list | `data/chunks/` |
| **Embedder** | Encode chunk text with MiniLM. | chunks | vectors (handed to Chroma) |
| **Vector store** | Persist/query embeddings + metadata (`chunk_id`, `source_url`, section). | config path | `data/chroma/` |
| **PII filter** | Regex/heuristics; never persist matched values. | user text | — |
| **Intent router** | Classify: factual / advice / returns / PII / other. Can be rules-first (keywords) with optional LLM assist. | query | optional log (sanitized) |
| **Retriever** | Embed query; `collection.query(top_k)`; attach scores and URLs. | Chroma | `data/retrieval_logs/` |
| **Prompt builder** | System + user + context blocks; refusal/returns templates. | retrieved chunks | optional prompt dump |
| **LLM client** | Provider interface: Groq required; Gemini / OpenAI / Anthropic / Mistral optional. | `.env` | — |
| **Answer assembler** | Enforce length, single citation from retrieved URLs, last-updated. | LLM output, ingest timestamp | response DTO |
| **UI** | Welcome, 3 examples, facts-only note, Q&A, disclaimer; optional pipeline panel. | response DTO, artifacts | — |

---

## 5. Data architecture

Every stage leaves a file (or Chroma collection) a reviewer can open.

### 5.1 Directory layout

```
.
├── PRD.md
├── architecture.md
├── README.md
├── .env.example
├── config/
│   ├── settings.yaml          # chunking, chroma, top_k, embedding model name
│   └── sources.yaml           # 15–25 official URLs (5 required Groww pages + extras)
├── data/
│   ├── raw/
│   │   ├── {source_id}.html   # or .txt
│   │   └── {source_id}.meta.json
│   ├── chunks/
│   │   └── chunks.jsonl
│   ├── chroma/                # Chroma persist directory
│   ├── ingest_manifest.json   # last_ingest_at, source counts, model ids
│   └── retrieval_logs/
│       └── {query_id}.json
├── src/                       # or app + notebooks
│   ├── ingest/
│   ├── retrieve/
│   ├── generate/
│   └── ui/
└── Docs/
    ├── ProblemStatement.txt
    ├── sources.csv            # submission source list
    └── sample_qa.md
```

Implementation may rename folders; the **contract** is: raw, chunks, chroma, retrieval logs, and a manifest exist and are documented in the README.

### 5.2 Artifact schemas

**Raw metadata** (`{source_id}.meta.json`)

```json
{
  "source_id": "hdfc-large-cap-direct-growth",
  "title": "HDFC Large Cap Fund Direct Growth",
  "url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
  "fetched_at": "2026-11-01T06:00:00Z",
  "http_status": 200,
  "content_type": "text/html"
}
```

**Document** (Document loader output, in memory; built from `{source_id}.txt` + `.meta.json` + `sources.yaml`)

```json
{
  "source_id": "hdfc-large-cap-direct-growth",
  "title": "HDFC Large Cap Fund Direct Growth",
  "url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
  "type": "scheme",
  "scheme": "HDFC Large Cap Fund",
  "fetched_at": "2026-11-01T06:00:00Z",
  "sections": [
    { "heading": "Key facts", "text": "Expense ratio: ...\nExit load: ..." }
  ]
}
```

**Chunk** (one JSONL line)

```json
{
  "chunk_id": "hdfc-large-cap-direct-growth::003",
  "source_id": "hdfc-large-cap-direct-growth",
  "source_url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
  "section": "Exit load",
  "text": "...",
  "chunk_index": 3
}
```

**Chroma metadata per vector:** `chunk_id`, `source_id`, `source_url`, `section` (if any). Document text is the chunk `text`. Collection name and persist path are config.

**Ingest manifest**

```json
{
  "last_ingest_at": "2026-11-01T06:05:00Z",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "chroma_collection": "mf_faq",
  "source_count": 18,
  "chunk_count": 142
}
```

`last_ingest_at` is what the UI prints as **Last updated from sources**.

**Retrieval log**

```json
{
  "query_id": "...",
  "asked_at": "...",
  "query": "What is the expense ratio of HDFC Large Cap Fund Direct Growth?",
  "intent": "factual",
  "top_k": 5,
  "hits": [
    {
      "chunk_id": "...",
      "source_url": "...",
      "score": 0.81,
      "text": "..."
    }
  ]
}
```

Do **not** write retrieval logs if the PII filter fired.

### 5.3 Chunking strategy (decision at implement time)

Corpus shape from the PRD: scheme pages mix **labeled fact blocks** (expense ratio, SIP, riskometer) and prose; the AMC page is mixed overview.

**Preferred:** heading / section-aware split (HTML headings, definition lists, or “label: value” blocks) so one chunk ≈ one fact family.  
**Fallback:** recursive character split with config `chunk_size` and `chunk_overlap`.

Document the chosen strategy in README (PRD Phase 2). Keep parameters in `config/settings.yaml`.

---

## 6. Retrieval and generation

### 6.1 Embedding contract

- **Ingest and query must use the same model:** `sentence-transformers/all-MiniLM-L6-v2` (overridable in config; changing it requires **full re-ingest**).
- Query text is embedded as-is after PII filter (no extra instruction prefix unless config says otherwise).

### 6.2 Retrieval

- Backend: **ChromaDB**, same collection as ingest.
- `top_k` from config.
- Return documents + distances/scores + `source_url`.
- If all scores are below an optional `min_score` (config), treat as **missing fact**: still cite the best retrieved URL or the configured fallback source (`fallback_source_id`, default the HDFC AMC overview); do not invent numbers.

### 6.3 Prompt contract

The LLM sees:

1. **System:** facts-only; ≤ 3 sentences; answer only from context; one citation URL copied from context metadata; refuse advice; do not compute/compare returns; if context lacks the fact, say so.
2. **Context:** retrieved chunks as `{source_url, section, text}`.
3. **User:** the sanitized question.
4. **Output shape** (enforced in prompt and post-check):

```
<answer ≤ 3 sentences>
Source: <url>
Last updated from sources: <last_ingest_at date>
```

Citation URL **must** be one of the retrieved `source_url` values (or the configured refusal / returns source on those template routes). Post-check replaces hallucinated URLs.

### 6.4 Intent routing (W1)

| Intent | Detection (v1: rules) | Path |
|--------|----------------------|------|
| PII | PAN / Aadhaar / phone / email / OTP / account-like tokens | Stop; no store; generic “do not share personal details” |
| Advice | buy, sell, switch, better for me, should I, recommend, portfolio | Refusal + one investor-education link from the registry (`refusal_url_source_id`, default SEBI Investor Education) |
| Returns | return, CAGR, outperformed, best performing, compare performance | No math; link the official HDFC MF factsheet (`returns_source_id`), else the scheme page; ≤ 3 sentences |
| Factual | expense ratio, exit load, SIP, lock-in, riskometer, benchmark, statement download, etc. | RAG |
| Other / unknown scheme | no matching scheme in corpus | Fail closed: not in this prototype’s sources |

Rules-first keeps the demo deterministic. An LLM classifier is optional and must still honor the same routes.

---

## 7. LLM provider layer

```
                    ┌─────────────┐
                    │ LLM Client  │  generate(messages) → text
                    └──────┬──────┘
           ┌───────────────┼────────────────┐
           ▼               ▼                ▼
        Groq*        Gemini / OpenAI   Anthropic / Mistral
        (required)   (optional)        (optional)
```

- **Required for the exercise:** Groq. Model id from `.env` (PRD intent: `qwen/qwen3.8-27b` — confirm against Groq’s catalog at implementation).
- **API keys** only in `.env`, never in git.
- Switching provider is a config enum (`LLM_PROVIDER`) plus the matching key; application code talks to one interface.

---

## 8. Configuration

| Key | Where | Default |
|-----|--------|---------|
| `EMBEDDING_MODEL` | config | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHROMA_PERSIST_DIR` | config | `data/chroma` |
| `CHROMA_COLLECTION` | config | `mf_faq` |
| `CHUNK_STRATEGY` | config | `section` or `recursive` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | config | chosen at implementation |
| `TOP_K` | config | e.g. `5` |
| `MIN_SCORE` | config | optional |
| `LLM_PROVIDER` | `.env` / config | `groq` |
| `LLM_MODEL` | `.env` | Groq model id |
| `GROQ_API_KEY` | `.env` | — |
| Optional `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY` | `.env` | — |

Changing embedding model or chunking **invalidates** Chroma; operator re-runs ingest.

---

## 9. UI architecture

Minimal surface (PRD §7):

| Element | Notes |
|---------|--------|
| Welcome line | What the assistant is |
| 3 example questions | Pre-fill factual queries from PRD §13 |
| Persistent note | **Facts-only. No investment advice.** |
| Disclaimer | Canonical snippet from PRD §7 |
| Input + answer | Body, **Source:** URL, **Last updated from sources:** date |
| Pipeline panel (preferred) | Show last retrieval hits / link to artifacts |

**Stack:** smallest working prototype (Streamlit, Gradio, or notebook). UI is a thin client over the query pipeline; it does not scrape or embed.

---

## 10. Security and compliance (architecture-level)

| Rule | Mechanism |
|------|-----------|
| Public sources only | Source registry allowlist; citations must come from chunk metadata, never free-typed by the model without post-check. |
| No third-party blogs | Registry includes only Groww public pages, HDFC MF official documents, and SEBI/AMFI investor education. |
| No PII persist | Filter before logs, Chroma, and LLM. |
| No performance claims | Intent route + prompt; no calculator module in the system. |
| No advice | Intent route + refusal template. |

This is a local/demo prototype: no production auth, rate limits, or multi-user isolation unless added later.

---

## 11. Runtime views

### 11.1 Ingest sequence

```
Operator → Config → Scraper → disk/raw
                → Document loader (reads disk/raw) → Documents
                → Chunker → disk/chunks
                → Embedder → Chroma persist
                → Manifest (last_ingest_at)
```

### 11.2 Query sequence (factual)

```
User → UI → PII filter → Intent (factual)
    → Embed query → Chroma top_k → retrieval log
    → Prompt (chunks only) → LLM
    → Post-check (length, URL ∈ retrieved, last-updated)
    → UI
```

### 11.3 Query sequence (advice)

```
User → UI → PII filter → Intent (advice)
    → Refusal copy + refusal URL from registry
    → UI  (LLM optional; not required)
```

---

## 12. Failure modes

| Failure | Behaviour |
|---------|-----------|
| Fetch 4xx/5xx / JS-empty page | Record error in raw meta; skip or retry; ingest continues for other URLs; README known limit. |
| Empty Chroma | UI: ingest required; do not call LLM with empty context pretending facts exist. |
| LLM timeout / bad key | User-visible error; no fake fact. |
| Hallucinated URL | Post-check substitutes top retrieved `source_url`. |
| Hallucinated numbers | Mitigated by “only from context”; demo reviewer checks sample Q&A against pages. |
| Question about other AMC | Fail closed. |

---

## 13. Phased build (maps to PRD §11)

| Phase | Architecture slice |
|-------|--------------------|
| 0 | Repo, config, `.env.example`, disclaimer |
| 1 | Scraper + `data/raw`; Document loader reads it back |
| 2 | Chunker (consumes Documents) + `data/chunks` |
| 3 | Embedder + Chroma persist |
| 4 | Retriever + `data/retrieval_logs` |
| 5 | Intent + prompt + LLM client + post-check |
| 6 | UI |
| 7 | Source list, sample Q&A, README |

Each phase should leave artifacts that the next phase consumes; do not skip disk writes in favor of in-memory-only pipelines.

---

## 14. Technology choices (locked by PRD)

| Concern | Choice |
|---------|--------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB (persistent) |
| Primary LLM | Groq; model from `.env` |
| Optional LLMs | Gemini, OpenAI, Anthropic, Mistral |
| Language | Python (assumed for MiniLM + Chroma + Streamlit/Gradio) |

---

## 15. Open implementation notes

- Confirm Groq model id against current catalog.
- Choose fetch method that still uses **public HTML** (plain HTTP vs lightweight render) if Groww pages are JS-heavy.
- Sources: 5 required Groww pages + 19 verified extras (HDFC MF KIM/SID per scheme, June 2026 factsheet pages for the 4 schemes, Flexi Cap fund facts / scheme summary, Groww help statement guides, Groww expense-ratio explainer, AMFI Investor Corner, SEBI investor education). Each was checked to return 200 with extractable text for the project's own user agent; `www.hdfcfund.com` HTML pages return 403 and are excluded.
- UI framework: pick one in Phase 6; architecture does not depend on it.
