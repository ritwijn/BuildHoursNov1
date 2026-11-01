# Product Requirements Document (PRD)

**Product:** Groww Mutual Fund FAQ Assistant  
**Version:** 1.0  
**Date:** 1 November 2026  
**Source of truth:** `Docs/ProblemStatement.txt`  
**Status:** Draft for prototype / Build Hours

---

## 1. Overview

A small, facts-only FAQ assistant that answers questions about selected HDFC Mutual Fund schemes using **official public pages only**. Every answer includes **one source link**. The assistant does **not** give investment advice.

This is a **RAG demo**. Each pipeline stage must persist its output so the flow can be inspected: load → chunk → embed → retrieve → generate.

---

## 2. Problem statement

Retail investors and support/content teams repeatedly ask the same scheme facts: expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark, and how to download statements.

They need answers that are:

- Factual and short
- Tied to an official public page
- Explicitly **not** advice (no buy/sell/portfolio recommendations)
- Transparent about when source data was last ingested

---

## 3. Goals and non-goals

### 3.1 Goals

| ID | Goal |
|----|------|
| G1 | Answer factual MF scheme/AMC questions from a small, curated public corpus. |
| G2 | Show **exactly one** citation URL in every answer. |
| G3 | Refuse opinionated / advice questions politely, with a relevant educational link. |
| G4 | Keep answers ≤ 3 sentences and stamp **Last updated from sources**. |
| G5 | Persist artifacts for each RAG stage so the demo can be walked through. |
| G6 | Make models, vector DB, and related knobs changeable via config / `.env`. |

### 3.2 Non-goals

- Portfolio construction, “should I buy/sell?”, or personalized allocation.
- Computing or comparing returns / performance rankings.
- Scraping or citing the Groww app backend, screenshots, or third-party blogs.
- Collecting or storing PII (PAN, Aadhaar, account numbers, OTPs, emails, phones).
- Coverage of all AMCs or all HDFC schemes (prototype is AMC page + 4 schemes).
- Production-grade auth, multi-tenant hosting, or live market data feeds.

---

## 4. Users

| Persona | Need |
|---------|------|
| Retail user comparing schemes | Fast, cited facts (SIP min, exit load, riskometer, etc.). |
| Support / content | Repeatable answers to the same MF FAQs without inventing advice. |
| Demo reviewer / evaluator | Visible RAG stages, source list, sample Q&A, disclaimer. |

---

## 5. Scope

### 5.1 In scope — corpus (must scrape and store)

| # | Page | URL |
|---|------|-----|
| 1 | AMC overview — HDFC Mutual Fund | https://groww.in/mutual-funds/amc/hdfc-mutual-funds |
| 2 | Large Cap — HDFC Large Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| 3 | Flexi Cap — HDFC Flexi Cap Direct Plan Growth | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| 4 | ELSS (tax saver) — HDFC ELSS Tax Saver Fund Direct Plan Growth | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth |
| 5 | Mid Cap — HDFC Mid Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth |

Additional **official public** URLs (Groww scheme/AMC pages, SEBI/AMFI educational pages, official factsheets/statement-help pages) may be added to reach **15–25 sources** for the submitted source list. Third-party blogs are out of scope as citations.

### 5.2 In scope — query types

Factual only, for example:

- Expense ratio
- Exit load
- Minimum SIP
- ELSS lock-in
- Riskometer / benchmark
- How to download capital-gains / account statements (process from public help pages)

### 5.3 Out of scope — query types

- “Should I buy / sell / switch?”
- “Which fund is better?”
- Return comparison, predicted performance, or “best fund”
- Anything requiring PAN, folio, login, or personal holdings

---

## 6. Product requirements

### 6.1 FAQ assistant (working prototype)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | User can ask a free-text question about in-scope schemes/AMC. | P0 |
| FR-2 | Assistant identifies the **exact fact** asked and answers only that. | P0 |
| FR-3 | Answers are **≤ 3 sentences**. | P0 |
| FR-4 | Every answer includes **one** clear citation link. | P0 |
| FR-5 | Every answer includes **Last updated from sources: &lt;timestamp/date&gt;**. | P0 |
| FR-6 | Opinion/advice questions get a polite **facts-only refusal** plus a relevant educational link (not a scheme pick). | P0 |
| FR-7 | Performance / returns questions: **do not compute or compare**; link to the official factsheet (or equivalent official page). | P0 |
| FR-8 | UI shows a welcome line, **3 example questions**, and the note: **“Facts-only. No investment advice.”** | P0 |
| FR-9 | UI (or nearby copy) includes a short **disclaimer snippet**. | P0 |
| FR-10 | Reject or strip PII in input; do not persist PAN, Aadhaar, account numbers, OTPs, emails, or phone numbers. | P0 |

### 6.2 RAG pipeline (must be explicit and inspectable)

Each stage **writes artifacts** (files and/or DB collections) that can be shown in the demo.

| Phase | What happens | Artifact to persist |
|-------|----------------|---------------------|
| 1. Data loading | Fetch the in-scope public pages; extract text. | Raw HTML/text per URL + metadata (URL, fetched_at). |
| 2. Data chunking | Chunk with a strategy suited to this corpus (scheme pages are structured fact blocks + prose; AMC overview is mixed). Strategy is chosen at implementation time and documented. | Chunk JSON/JSONL: `chunk_id`, `source_url`, `text`, optional heading/section. |
| 3. Data embedding | Embed chunks with **`sentence-transformers/all-MiniLM-L6-v2`**. | Embedding vectors + mapping to `chunk_id`. |
| 4. Vector store | Persist in **ChromaDB**. | Chroma collection; config-driven path/name. |
| 5. Retrieval | Query embedding uses the **same** embedding model as ingestion. Retrieve top-k chunks from the **same** Chroma collection. | Query, retrieved chunks, scores, source URLs. |
| 6. Generation | LLM answers **only from retrieved context**; cite one URL; refuse advice. | Prompt + context + model response (optional debug panel). |

### 6.3 Configuration

All of the following must be **config / environment driven** (not hardcoded in application logic):

| Setting | Default / notes |
|---------|-----------------|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB (persist directory, collection name) |
| LLM provider | Groq (primary for this exercise) |
| LLM model | From `.env` (intended: `qwen/qwen3.8-27b` — confirm exact Groq model id at implementation) |
| Alternate providers (optional / pluggable) | Gemini, OpenAI, Anthropic, Mistral — keys in `.env` |
| Chunk size / overlap / strategy name | Config |
| Retrieval `top_k` | Config |
| Source “last updated” timestamp | Derived from last successful ingest |

---

## 7. UX requirements

Minimal UI:

1. Welcome line (what the assistant is).
2. Three tappable/copyable example questions (factual).
3. Persistent note: **Facts-only. No investment advice.**
4. Question input + answer area.
5. Answer layout:
   - Body (≤ 3 sentences)
   - **Source:** &lt;one URL&gt;
   - **Last updated from sources:** &lt;date&gt;
6. Optional (strongly preferred for the RAG demo): a collapsible **pipeline view** showing retrieved chunks / stage artifacts.

Disclaimer snippet (canonical copy for UI + README):

> This assistant shares publicly available scheme facts only. It is not investment advice, not a recommendation to buy or sell, and not a substitute for the scheme information document, factsheet, or a SEBI-registered adviser.

---

## 8. Behaviour rules (W1–W3)

These map to the skills being tested.

### W1 — Identify the fact; answer vs refuse

- Extract: scheme (if any), field (expense ratio, SIP, lock-in, …), vs advice intent.
- If the fact is in retrieved context → answer.
- If the fact is missing → say so in ≤ 3 sentences and still cite the closest official page.
- If the user wants a decision → refuse.

### W2 — Prompting

- Instruction style: facts only, short, one citation, last-updated line.
- Polite refusal template, e.g.: *I can only share publicly listed scheme facts, not buy/sell or portfolio advice.* Then one educational official link (AMFI/SEBI/Groww help).
- Citation wording: one URL, clearly labelled.

### W3 — RAG only

- No answering from parametric “general knowledge” when the corpus should contain the fact.
- Citations must be URLs from the retrieved set (AMC/scheme/official educational pages), not invented links.

---

## 9. Technical architecture (prototype)

```
Public pages  →  Loader  →  Chunker  →  Embedder (MiniLM)
                                      →  ChromaDB
User query    →  Embed  →  Retrieve k  →  LLM (Groq / configurable)
                                      →  Answer + 1 URL + last-updated
```

**Suggested repo layout (implementation may refine):**

- `config` — models, chunking, Chroma paths
- `.env` — API keys, LLM model id
- `data/raw` — loaded pages
- `data/chunks` — chunk artifacts
- `data/chroma` — vector store persist
- `data/retrieval_logs` — optional query traces
- `src/` or notebooks — pipeline + UI
- `Docs/` — PRD, source list, sample Q&A

---

## 10. Constraints and compliance

| Constraint | Rule |
|------------|------|
| Public sources only | Cite Groww public pages and official regulator/AMC educational/factsheet pages. No app-backend screenshots. No blogs. |
| No PII | Do not accept/store PAN, Aadhaar, account numbers, OTPs, emails, phones. |
| No performance claims | Do not compute or compare returns. Point to official factsheet. |
| Clarity | ≤ 3 sentences + last-updated line + one source. |
| No advice | Refuse portfolio / buy-sell / “which is better”. |

---

## 11. Phased delivery

Implement in this order. Each phase is demoable on its own.

| Phase | Name | Done when |
|-------|------|-----------|
| 0 | Config & project skeleton | Config + `.env` example; README stub; disclaimer copy. |
| 1 | Data loading | All 5 required URLs fetched; raw text + metadata stored. |
| 2 | Data chunking | Chunks on disk; strategy documented (why it fits this data). |
| 3 | Embedding + Chroma | MiniLM embeddings in Chroma; collection inspectable. |
| 4 | Retrieval | Query → top-k chunks + scores + source URLs logged. |
| 5 | LLM generation | Groq (model from `.env`); answer/refuse rules; one citation. |
| 6 | Tiny UI | Welcome, 3 examples, facts-only note, answer + source + last updated. |
| 7 | Deliverables pack | Source list, sample Q&A, README, disclaimer, prototype link or ≤3-min demo. |

---

## 12. Deliverables (submission)

| Deliverable | Spec |
|-------------|------|
| Working prototype | App/notebook link, **or** ≤ 3-minute demo video if hosting is not possible. |
| Source list | CSV or Markdown of **15–25 URLs** actually used. |
| README | Setup steps, scope (AMC + schemes), known limits. |
| Sample Q&A | **5–10** queries with assistant answers **and** links. |
| Disclaimer snippet | Facts-only, no advice (same as UI). |
| Stage artifacts | Explicit stored outputs for load / chunk / embed / retrieve (RAG demo). |

---

## 13. Sample questions (UI examples + Q&A file)

Use at least three of these on the UI:

1. What is the expense ratio of HDFC Large Cap Fund Direct Growth?
2. What is the ELSS lock-in for HDFC ELSS Tax Saver Fund Direct Plan Growth?
3. What is the minimum SIP for HDFC Mid Cap Fund Direct Growth?
4. What is the exit load of HDFC Flexi Cap Direct Plan Growth?
5. What is the riskometer / benchmark of HDFC Large Cap Fund Direct Growth?
6. How do I download a capital-gains statement?

**Refusal samples** (for the Q&A file, not as “try these” if they encourage advice-seeking in the main UI):

- Should I buy HDFC Flexi Cap?
- Which of these funds is better for me?

---

## 14. Success criteria

The prototype is successful if:

1. Factual questions on in-scope schemes return a correct fact **supported by the cited page**.
2. Every answer has **one** working public source URL.
3. Advice questions are refused without a recommendation.
4. Returns questions do not invent numbers; they point to a factsheet/official page.
5. UI shows welcome, 3 examples, and **Facts-only. No investment advice.**
6. A reviewer can inspect **each RAG stage’s stored data**.
7. Embedding model, Chroma, and LLM model can be changed from config / `.env` without code archaeology.
8. Submission pack matches Section 12.

---

## 15. Known risks / limits (to document in README)

- Groww pages may be JS-rendered; scraping may need a fetch strategy that still stays on public HTML.
- Facts can go stale; “last updated from sources” is ingest time, not live NAV time.
- Small corpus: questions about other AMCs/schemes should fail closed (no guess).
- LLM may still over-answer; system prompt + retrieval-only context must constrain this.
- Groq model id must match the provider’s current catalog.

---

## 16. Open decisions (implementation)

| Topic | Guidance |
|-------|----------|
| Chunking strategy | Choose based on extracted HTML (prefer heading/section-aware chunks for scheme fact tables; fallback recursive character split). Document the choice. |
| UI stack | Smallest working prototype (Streamlit / Gradio / notebook) is acceptable. |
| Extra 10–20 URLs | Official statement-download help, SEBI/AMFI investor education, scheme factsheet PDFs if publicly linked. |
| Multi-provider LLM | Groq required for the exercise; other providers behind config if time allows. |

---

## 17. Appendix — mapping from problem statement

| Problem statement item | PRD section |
|------------------------|-------------|
| Facts-only FAQ + one source link + no advice | §§2, 6.1, 7, 8 |
| Who this helps | §4 |
| 5 Groww URLs | §5.1 |
| Query examples + refuse advice | §§5.2–5.3, 13 |
| Public sources / no PII / no performance / ≤3 sentences | §10 |
| Deliverables | §12 |
| W1–W3 skills | §8 |
| RAG stages + Chroma + MiniLM + Groq | §§6.2–6.3, 9, 11 |
| Config / `.env` | §6.3 |
