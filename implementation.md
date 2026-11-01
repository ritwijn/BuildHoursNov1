# Implementation Plan

**Product:** Groww Mutual Fund FAQ Assistant  
**From:** `PRD.md`, `architecture.md`  
**Stack:** Python 3.11+, MiniLM, ChromaDB, Groq, Streamlit

Build in this order. Each step writes files you can inspect before the next step.

---

## 1. Creating directory

```
.
├── config/
│   ├── settings.yaml          # models, chunking, Chroma, top_k
│   ├── sources.yaml           # 15–25 official URLs (5 required Groww pages + extras)
│   └── intents.yaml           # advice / returns / factual keywords
├── data/
│   ├── raw/                   # scraped pages
│   ├── chunks/chunks.jsonl
│   ├── chroma/                # vector DB
│   ├── ingest_manifest.json
│   └── retrieval_logs/
├── src/
│   ├── config.py
│   ├── ingest/                # scrape, chunk, embed, store
│   ├── retrieve/
│   ├── generate/              # PII, intent, LLM, post-check
│   └── ui/app.py
├── tests/
├── .env.example
├── requirements.txt
└── Docs/
```

**Config (not hardcoded):** embedding model `sentence-transformers/all-MiniLM-L6-v2`, Chroma path/collection, chunk size/overlap, `top_k`, LLM from `.env` (`LLM_PROVIDER=groq`, `LLM_MODEL=…`, `GROQ_API_KEY=`).

**Done when:** folders exist, `settings.yaml` + `.env.example` exist, no API keys in `.py` files.

---

## 2. Scraping document

Fetch only URLs in `sources.yaml` (allowlist): the five required Groww pages (`required: true`) plus 15–20 optional official pages (HDFC MF KIM/SID/factsheets, Groww help, SEBI/AMFI education). A failed optional source is recorded and skipped.

**Flow:** HTTP fetch (`httpx`) → extract text → save files. If extracted text is too short, optional Playwright fallback (public HTML only).

**PDFs** (KIM, SID, factsheets): save `data/raw/{id}.pdf`, extract one section per page with `pypdf`; an optional `pages:` list in `sources.yaml` keeps only some pages (the 140-page factsheet keeps just the 4 schemes' pages).

**Extract (HTML):** drop nav/script/footer; split on headings; turn tables/`dl` into `Label: Value` lines (expense ratio, exit load, min SIP, lock-in, riskometer, benchmark). If `__NEXT_DATA__` JSON exists on Groww pages, pull those facts into a “Key facts” section. Groww help pages: take the answer for the page's `questionId` from the same JSON.

**Write per source:**

| File | Content |
|------|---------|
| `data/raw/{id}.html` or `.pdf` | Document fetched |
| `data/raw/{id}.txt` | Readable sections |
| `data/raw/{id}.meta.json` | url, fetched_at, http_status, error |

On failure: save meta with `error`, continue. Fail the run only if a **required** URL failed.

**Done when:** all five required pages have text that actually contains expense ratio, exit load, SIP, riskometer, benchmark.

---

## 3. Loading documents

Read the scraped files from `data/raw/` back into memory as documents for chunking. Chunking reads **only from disk**, never from the scraper's in-memory output, so it can be re-run without re-fetching.

**Flow:** list `data/raw/*.meta.json` → skip sources whose meta has `error` → read matching `{id}.txt` → parse `## heading` blocks into sections → attach metadata from `sources.yaml` + meta.

**Document shape:**

```
Document
  source_id, title, url, type (amc | scheme | kim | sid | factsheet | fund_facts | scheme_summary | help | explainer | regulator)
  scheme (if any), fetched_at
  sections: [ { heading, text } ]
```

**Clean:** normalize whitespace and unicode (`₹`, `%`, non-breaking spaces); drop empty or duplicate sections; drop sections shorter than `min_chunk_chars`.

**Validate:** every source in `sources.yaml` has a loaded document or a recorded error; a required source with no sections fails the run. Print a table: source → sections → chars.

```
load_documents(raw_dir, sources) -> list[Document]
```

**Done when:** `python -m src.cli load` lists all required sources with non-zero sections, and the Large Cap document has a section containing the expense ratio.

---

## 4. Chunking

**Strategy:** one chunk per section (heading-aware). Prefix every chunk with the scheme title so four similar “Exit load” blocks do not collide in search.

If a section is longer than `chunk_size`, split on line boundaries and keep `chunk_overlap`. Fallback strategy: recursive character split (`config`).

**Fact chunks** (`fact_chunks: true`): a section that is a `Label: Value` list (≥ 3 lines, ≥ 80% matching) is also indexed one fact per chunk, so “Expense ratio: 1.02%” is not outranked by long SID pages that only mention expense-ratio limits.

**Write:** `data/chunks/chunks.jsonl`

```
chunk_id, source_id, source_url, section, text, chunk_index
```

`chunk_id = {source_id}::{index}` (stable across re-runs).

**Done when:** grepping `expense ratio` finds a chunk that names each of the four schemes.

---

## 5. Embedding

Same model for ingest and query: `sentence-transformers/all-MiniLM-L6-v2`, normalized vectors, batch from config.

```
embed_texts(list[str]) -> list[list[float]]
```

Load the model once and reuse. Changing the model name requires a full re-ingest.

---

## 6. Storing data in vector DB

**Store:** ChromaDB `PersistentClient` at `data/chroma`, collection from config, cosine space.

On ingest: **rebuild** the collection (no stale chunks). Store `id`, embedding, document text, metadata (`chunk_id`, `source_id`, `source_url`, `section`).

**Write:** `data/ingest_manifest.json` with `last_ingest_at`, embedding model, collection, chunk count. This date is **Last updated from sources** in the UI.

If collection embedding model ≠ config → error: “re-run ingest”. Empty collection → do not call the LLM.

**Done when:** `collection.count() ==` chunk count; second ingest does not duplicate.

---

## 7. Retrieval logic

**Query path:**

```
question
  → PII filter (reject; no log)
  → intent (advice | returns | out_of_scope | factual)
  → if template route: return fixed answer + one official URL (no LLM)
  → embed query with the same MiniLM model
  → Chroma top_k
  → log data/retrieval_logs/{query_id}.json
  → if best score < min_score: “I couldn’t find that in my sources” + closest URL
  → LLM (Groq) with retrieved chunks only
  → post-check → answer
```

**Retrieve:** embed `question + query_suffix` (default ` (mutual funds)`, anchors short questions to the domain); `score = 1 - cosine_distance`. Scheme boost: if the question names a known scheme, over-fetch 3× and move chunks from every source tagged with that scheme (Groww page, KIM, SID, fund facts) to the front.

**LLM:** Groq client; model from `.env`. Prompt: facts only, ≤ 3 sentences, one `Source:` URL copied from a retrieved chunk, no advice, no return math. Strip `<think>` if the model emits it.

**Post-check:** citation must be in retrieved URLs (else use top hit); trim to 3 sentences; if body has advice phrases → refusal; numbers with `%` / `₹` must appear in retrieved text; append last-updated from the manifest.

**Done when:** sample factual questions hit the right scheme page; advice/returns/PII never invent facts.

---

## 8. UI

Streamlit: `streamlit run src/ui/app.py`. UI only calls `answer_question()` — it does not scrape or embed.

| Element | What |
|---------|------|
| Welcome | HDFC MF facts on Groww (expense ratio, SIP, lock-in, …) |
| Note | **Facts-only. No investment advice.** |
| 3 examples | Expense ratio / ELSS lock-in / min SIP |
| Answer | Body, **Source:** link, **Last updated from sources:** |
| Pipeline expander | Intent, retrieved chunks + scores, post-check flags |
| Footer | Disclaimer from PRD |

Cache embedder + Chroma. If ingest not run, show “run ingest first”. Do not echo PII-rejected input.

---

## 9. Edge cases and test cases for guardrails

Guardrails run **before** retrieval/LLM (PII, intent) and **after** LLM (post-check). Template routes must **not** call the LLM.

### 9.1 Edge cases

| Case | Expected |
|------|----------|
| PAN / Aadhaar / phone / email / OTP / folio in the question | Reject. No retrieval log, no LLM. Generic “don’t share personal details” message. |
| `₹500`, `1%`, `0.98%`, `3 years` | **Not** PII. Treat as a normal question. |
| “Should I buy HDFC Flexi Cap?” / “Which fund is better?” | Advice refusal + SEBI investor-education URL. No recommendation. |
| “What were 5-year returns?” / “Which performed better?” | No calculated numbers. Link the official HDFC MF factsheet. |
| “Expense ratio of SBI Bluechip?” | Out of scope (corpus is HDFC AMC + listed schemes). |
| “How do I download a capital-gains statement?” | Factual; no scheme required. |
| Fact missing / low retrieval score | “I couldn’t find that in my sources.” + closest official URL. |
| LLM invents a URL | Replace with top retrieved URL. |
| LLM returns 5 sentences | Keep first 3. |
| LLM invents an expense ratio not in chunks | Replace with not-found (grounding check). |
| LLM says “you should buy” | Replace with advice refusal. |
| Empty question / empty Chroma / bad Groq key | Clear error; no fake fact. |
| Other AMC / unknown scheme | Fail closed. |

### 9.2 Test cases (`pytest`)

**PII**

| Input | Result |
|-------|--------|
| `My PAN is ABCDE1234F, min SIP?` | Reject; `retrieval_logs` unchanged |
| `Aadhaar 2345 6789 0123` | Reject |
| `OTP 123456 for my folio` | Reject |
| `email me at a@b.com` | Reject |
| `What is min SIP? It is ₹500.` | Pass filter |
| `Exit load 1% and expense 0.98%` | Pass filter |

**Intent / templates** (mock LLM — assert it was **not** called)

| Input | Intent | LLM? |
|-------|--------|------|
| Expense ratio of HDFC Large Cap Direct Growth | factual | yes |
| ELSS lock-in for HDFC ELSS Tax Saver | factual | yes |
| Should I buy HDFC Flexi Cap? | advice | no |
| Which of these funds is better for me? | advice | no |
| 5-year returns of HDFC Mid Cap | returns | no |
| Expense ratio of SBI Bluechip | out_of_scope | no |

**Post-check**

| Input to post-check | Result |
|---------------------|--------|
| `Source: https://evil.example` | Replaced with top hit URL |
| Two URLs in the answer | One URL left |
| Five sentences | Three sentences |
| `0.98%` in body, not in chunks | Not-found message |
| `you should invest` | Advice refusal |
| Missing last-updated | Appended from manifest |

**Retrieval smoke (after ingest)**

| Question | Top hit should be |
|----------|-------------------|
| Exit load of HDFC Flexi Cap | Flexi Cap Groww page |
| Expense ratio of HDFC Large Cap | Large Cap Groww page |
| Min SIP of HDFC Mid Cap | Mid Cap Groww page |

Run `pytest -q` after chunking and after guardrails. Network/Groq tests are optional and marked, not default.
