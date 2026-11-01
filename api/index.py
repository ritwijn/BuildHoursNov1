"""FastAPI app: the HTTP API over answer_question(), plus the static UI when run locally.

Local:   uvicorn api.index:app --reload        → http://localhost:8000
Vercel:  this file is the Python function; Vercel serves public/ itself and routes /api/* here.

On Vercel the deployment filesystem is read-only, so the bundled Chroma store is copied
to /tmp on cold start and retrieval logs go to /tmp (ephemeral).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.config import get_settings, load_sources  # noqa: E402
from src.generate.service import answer_question  # noqa: E402
from src.ingest.store import IngestRequired, read_manifest  # noqa: E402

ON_VERCEL = bool(os.environ.get("VERCEL"))
PUBLIC = ROOT / "public"
EXAMPLES = [
    "What is the expense ratio of HDFC Large Cap Fund Direct Growth?",
    "What is the ELSS lock-in for HDFC ELSS Tax Saver Fund Direct Plan Growth?",
    "What is the minimum SIP for HDFC Mid Cap Fund Direct Growth?",
]


def _prepare_runtime() -> None:
    """Point Chroma and logs at writable /tmp copies when the bundle is read-only."""
    if not ON_VERCEL:
        return
    settings = get_settings()
    bundled = settings.resolve_path(settings.chroma.persist_dir)
    writable = Path("/tmp/chroma")
    if not writable.exists() and bundled.exists():
        shutil.copytree(bundled, writable)
    settings.chroma.persist_dir = str(writable)
    settings.paths.retrieval_log_dir = "/tmp/retrieval_logs"


_prepare_runtime()
app = FastAPI(title="Groww MF Facts Assistant", docs_url="/api/docs", openapi_url="/api/openapi.json")


class AskRequest(BaseModel):
    question: str = Field(..., max_length=500)


@app.get("/api/status")
def status() -> dict:
    settings = get_settings()
    manifest = read_manifest(settings)
    return {
        "ready": manifest is not None,
        "sources": manifest and manifest["source_count"],
        "chunks": manifest and manifest["chunk_count"],
        "last_ingest_at": manifest and manifest["last_ingest_at"],
        "embedding_model": settings.embedding.model,
        "vector_store": f"chroma/{settings.chroma.collection}",
        "llm": f"{settings.llm.provider}/{settings.llm.model}",
        "top_k": settings.retrieval.top_k,
        "examples": EXAMPLES,
    }


@app.post("/api/ask")
def ask(request: AskRequest) -> dict:
    try:
        answer = answer_question(request.question)
    except IngestRequired as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    titles = {source.url: source.title for source in load_sources()}
    return {
        "status": answer.status,
        "body": answer.body,
        "source_url": answer.source_url,
        "source_title": titles.get(answer.source_url),
        "last_updated": answer.last_updated,
        "intent": answer.intent,
        "scheme_id": answer.scheme_id,
        "query_id": answer.query_id,
        # The raw LLM error text can contain provider details; keep it server-side.
        "flags": [flag for flag in answer.flags if not flag.startswith("llm_error:")],
        "hits": [
            {
                "rank": rank,
                "score": round(hit.score, 3),
                "section": hit.section,
                "chunk_id": hit.chunk_id,
                "source_url": hit.source_url,
                "preview": hit.text.split("\n", 1)[-1][:240],
            }
            for rank, hit in enumerate(answer.hits, start=1)
        ],
        "prompt": answer.prompt,
    }


if not ON_VERCEL and PUBLIC.exists():
    # Locally, serve the same static UI that Vercel serves from public/.
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(PUBLIC / "index.html")

    app.mount("/", StaticFiles(directory=PUBLIC), name="public")
