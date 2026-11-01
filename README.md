# Groww Mutual Fund FAQ Assistant (facts-only RAG prototype)

A small assistant that answers **factual** questions about HDFC Mutual Fund schemes listed on Groww: expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark, fund manager, and how to download statements. Every answer is at most 3 sentences, cites **one** official source link, and ends with *Last updated from sources*. It refuses advice, never computes returns, and rejects personal details.

> **Disclaimer:** This assistant shares publicly available scheme facts only. It is not investment advice, not a recommendation to buy or sell, and not a substitute for the scheme information document, factsheet, or a SEBI-registered adviser.

**Facts-only. No investment advice.**

---

## Scope

| | |
|---|---|
| Product | **Groww** (investment platform) |
| AMC | **HDFC Mutual Fund** |
| Schemes (4) | HDFC Large Cap Fund · HDFC Flexi Cap Fund · HDFC ELSS Tax Saver · HDFC Mid Cap Fund (Direct Growth) |
| Sources | **24 official public URLs**, listed in [`Docs/sources.csv`](Docs/sources.csv) |
| Out of scope | Other AMCs, other HDFC schemes, and **Groww Mutual Fund's own schemes** (Groww is also an AMC; questions like "Groww ELSS Tax Saver Fund" are refused as out of scope) |

**Sources (24):**

| Group | Count | What |
|---|---|---|
| Groww scheme/AMC pages | 5 | The required pages: HDFC AMC overview + the 4 scheme pages |
| HDFC MF documents (PDF) | 11 | KIM + SID for each scheme (21 Nov 2025), June 2026 factsheet (only the 4 schemes' pages), Flexi Cap fund facts and scheme summary |
| Groww help / explainer | 5 | Capital-gains / ELSS / CAS statement guides, redemption & lock-in, expense ratio explainer |
| SEBI / AMFI | 3 | Investor education (used as the refusal link) |

No third-party blogs, no app back-end screenshots.

### What it answers

| Question type | Behaviour |
|---|---|
| Factual ("Expense ratio of HDFC Large Cap?") | Retrieves chunks, LLM answers only from them, one citation |
| How-to ("How do I download a capital-gains statement?") | Answered from Groww help steps |
| Advice ("Should I buy…?", "Which is better for me?") | Polite refusal + SEBI investor-education link (no LLM call) |
| Returns ("5-year returns?") | No numbers; links the official HDFC MF factsheet (no LLM call) |
| Other AMC / scheme | Out-of-scope message (no LLM call) |
| Contains PAN / Aadhaar / phone / email / OTP / account no. | Rejected before anything is logged, embedded or sent to the LLM |
| Fact not in the sources | "I couldn't find that in my sources." + closest official page |

See [`Docs/sample_qa.md`](Docs/sample_qa.md) for 10 sample questions with the assistant's actual answers and links.

---

## Setup

Requirements: Python 3.11+ (developed on 3.14), macOS/Linux, a [Groq](https://console.groq.com) API key.

```bash
git clone git@github.com:ritwijn/BuildHoursNov1.git
cd BuildHoursNov1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # includes torch via sentence-transformers (large)

cp .env.example .env                     # then set GROQ_API_KEY=... in .env
```

`.env` holds the API key and LLM model (`LLM_MODEL=qwen/qwen3.8-27b`). It is git-ignored.

## Run

```bash
python -m src.cli ingest                 # scrape → load → chunk → embed → store (~1 min)
streamlit run src/ui/app.py              # UI at http://localhost:8501
```

The repo includes a snapshot of the scraped pages, chunks and embeddings (see below), so you can skip scraping with `python -m src.cli ingest --no-scrape`, which rebuilds only the Chroma store.

### CLI (one command per RAG stage)

| Command | Stage | Writes |
|---|---|---|
| `python -m src.cli scrape` | 1. Data loading (fetch + extract) | `data/raw/{id}.html\|.pdf`, `.txt`, `.meta.json` |
| `python -m src.cli load` | Load scraped files into documents | (prints a per-source report) |
| `python -m src.cli chunk` | 2. Chunking | `data/chunks/chunks.jsonl` |
| `python -m src.cli embed` | 3. Embedding (MiniLM) | `data/embeddings/embeddings.npy`, `index.json` |
| `python -m src.cli store` | 4. Vector store (ChromaDB) | `data/chroma/`, `data/ingest_manifest.json` |
| `python -m src.cli ingest [--no-scrape]` | All of the above | |
| `python -m src.cli retrieve "<q>"` | 5. Retrieval only | `data/retrieval_logs/{query_id}.json` |
| `python -m src.cli ask -v "<q>"` | 6. Full guarded answer | adds the answer + raw LLM output to the log |
| `python -m src.cli sample-qa` | Regenerate `Docs/sample_qa.md` | |
| `python -m src.cli export-sources` | Regenerate `Docs/sources.csv` | |
| `python -m src.cli status` | Show config and artifact state | |

### Tests

```bash
pytest -q
```

86 tests: PII patterns, intent routing, post-check, chunking, embeddings, Chroma store, the answer service (with a fake LLM), and Streamlit UI smoke tests.

---

## How it works

```
Ingest:  sources.yaml → scrape (HTML + PDF) → load & clean → chunk → MiniLM embed → ChromaDB
Query:   question → PII filter → intent router ─┬─ advice / returns / out-of-scope → fixed answer + official link
                                                 └─ factual → embed → Chroma top-k → Groq LLM → post-check → answer
```

### Stored artifacts (committed snapshot)

| Stage | Path | In git |
|---|---|---|
| Raw pages | `data/raw/` (HTML, PDF, extracted `.txt`, `.meta.json`) | yes |
| Chunks | `data/chunks/chunks.jsonl` (2,045 chunks) | yes |
| Embeddings | `data/embeddings/embeddings.npy` + `index.json` | yes |
| Vector store | `data/chroma/` | no (rebuilt by `ingest --no-scrape`) |
| Manifest | `data/ingest_manifest.json` (last ingest time, model, counts) | yes |
| Retrieval logs | `data/retrieval_logs/` | only the 6 from `Docs/sample_qa.md` |

### Chunking strategy

- **Section-aware:** one chunk per section of a page (Groww "Key facts", "Exit load history", AMC sections; one section per PDF page). Long sections split on line boundaries (`chunk_size` 800, `chunk_overlap` 100) so `Label: Value` lines are never cut.
- **Title prefix:** every chunk starts with `{source title} — {section}`, so four near-identical "Exit load" blocks from different schemes stay distinguishable.
- **Fact chunks:** a section that is a `Label: Value` list (e.g. Key facts) is also indexed one fact per chunk, so "Expense ratio: 1.02%" is not outranked by long SID pages that only mention expense-ratio *limits*.
- Fallback `recursive` strategy available in config.

### Retrieval and guardrails

- Same embedding model for ingest and query (`sentence-transformers/all-MiniLM-L6-v2`, normalized, cosine). Changing it is detected and requires re-ingest.
- The question is embedded with a ` (mutual funds)` suffix to anchor short questions to the domain.
- If the question names a scheme, chunks from every source about that scheme (Groww page, KIM, SID, fund facts) are moved to the front.
- Below `min_score` (0.35) the LLM is not called.
- **Post-check** on every LLM reply: the citation must be a retrieved URL (else replaced), extra URLs removed, ≤ 3 sentences, advice phrasing → refusal, any `%` / `₹` figure not present in the retrieved text → "couldn't find".

### Configuration

Everything tunable is in `config/settings.yaml` (overridable from `.env`):

| Setting | Default |
|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Chroma path / collection | `data/chroma` / `mf_faq` |
| Chunking | `section`, 800 / 100, fact chunks on |
| Retrieval | `top_k` 5, `min_score` 0.35, scheme boost on |
| LLM | `LLM_PROVIDER=groq`, `LLM_MODEL` from `.env` (Gemini / OpenAI / Anthropic / Mistral are reserved config slots, not implemented) |
| Answer | ≤ 3 sentences; refusal link = SEBI investor education; returns link = HDFC MF factsheet |
| Sources | `config/sources.yaml` (PDF sources can list `pages:` to ingest) |
| Intent keywords | `config/intents.yaml` |

---

## Known limits

- **Facts can be stale.** "Last updated from sources" is the ingest time, not a live NAV/TER feed. Scheme documents are dated (KIM/SID 21 Nov 2025, factsheet June 2026); re-run `ingest` to refresh.
- **Small, fixed corpus.** Questions about other AMCs, other HDFC schemes, or Groww Mutual Fund's schemes fail closed. Facts that are on none of the 24 pages get "I couldn't find that in my sources."
- **Rules-first intent routing.** Keyword lists can misroute unusual phrasings (e.g. "Should I check the exit load?" is treated as advice). Questions with no keyword still go through retrieval and the score threshold.
- **Retrieval is phrasing-sensitive.** MiniLM is small; very short or unusual questions can retrieve weaker chunks. The post-check and "couldn't find" fallback keep this from producing invented facts, but some answerable questions may be declined.
- **LLM can still over-answer.** Mitigated by context-only prompting, temperature 0, citation replacement, and the number-grounding check, not eliminated.
- **Scraping.** `www.hdfcfund.com` web pages return 403 to the scraper and are not used (their PDFs on `files.hdfcfund.com` are). CAMS / KFintech statement pages need JavaScript and are not used. The optional Playwright fallback is not installed by default.
- **Formatting quirks** come straight from source data, e.g. ELSS lock-in reads "3 year(s)".
- **Groq model id** (`qwen/qwen3.8-27b`) must exist in Groq's catalog; change `LLM_MODEL` in `.env` if it is retired.
- **Local prototype.** No auth or rate limiting. `streamlit run` listens on all interfaces by default; add `--server.address localhost` to keep it local.
