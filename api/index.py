"""FastAPI app: the HTTP API over answer_question(), plus the static UI when run locally.

Local:   uvicorn api.index:app --reload        → http://localhost:8000
Vercel:  this file is the Python function; Vercel serves public/ itself and routes /api/* here.

The Chroma store isn't committed: it is built from the committed embeddings at startup
(/tmp on Vercel, whose bundle is read-only; data/chroma/ locally). Retrieval logs on
Vercel go to /tmp (ephemeral).
"""

from __future__ import annotations

import os
import sys
from urllib.parse import parse_qsl, urlencode
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
from src.ingest.embedder import EmbeddingsStale  # noqa: E402
from src.ingest.store import IngestRequired, ensure_store, read_manifest  # noqa: E402

ON_VERCEL = bool(os.environ.get("VERCEL"))
PUBLIC = ROOT / "public"
EXAMPLES = [
    "What is the expense ratio of HDFC Large Cap Fund Direct Growth?",
    "What is the ELSS lock-in for HDFC ELSS Tax Saver Fund Direct Plan Growth?",
    "What is the minimum SIP for HDFC Mid Cap Fund Direct Growth?",
]


def _prepare_runtime() -> None:
    """Build the Chroma store from the committed embeddings if it's missing or stale.

    On Vercel the bundle is read-only, so the store and retrieval logs live in /tmp
    (rebuilt once per cold start, ~1 s). Locally it lives in data/chroma/.
    """
    settings = get_settings()
    if ON_VERCEL:
        settings.chroma.persist_dir = "/tmp/chroma"
        settings.paths.retrieval_log_dir = "/tmp/retrieval_logs"
    try:
        ensure_store(settings)
    except (FileNotFoundError, EmbeddingsStale) as exc:
        # Requests will get a 503 "corpus not ready" from open_collection().
        print(f"vector store not built: {exc}")


_prepare_runtime()
app = FastAPI(title="Groww MF Facts Assistant", docs_url="/api/docs", openapi_url="/api/openapi.json")


class RewrittenPathMiddleware:
    """Restore the original /api/<path> on Vercel.

    vercel.json rewrites /api/:path* to /api/index?__path=:path*, and the function may see
    /api/index rather than the URL the browser asked for. When __path is present, route on
    /api/<__path> and drop the parameter; otherwise leave the request untouched (local runs).
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            query = parse_qsl(scope.get("query_string", b"").decode(), keep_blank_values=True)
            rewritten = [value for key, value in query if key == "__path"]
            if rewritten:
                path = "/api/" + rewritten[0].lstrip("/")
                rest = urlencode([(key, value) for key, value in query if key != "__path"])
                scope = {**scope, "path": path, "raw_path": path.encode(), "query_string": rest.encode()}
        await self.app(scope, receive, send)


app.add_middleware(RewrittenPathMiddleware)


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


@app.api_route("/api/{rest:path}", methods=["GET", "POST"], include_in_schema=False)
def unknown_api_route(rest: str) -> None:
    # Say which path arrived, so routing problems (e.g. a platform rewrite) are visible.
    raise HTTPException(status_code=404, detail=f"No API route for /api/{rest}")


if not ON_VERCEL and PUBLIC.exists():
    # Locally, serve the same static UI that Vercel serves from public/.
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(PUBLIC / "index.html")

    app.mount("/", StaticFiles(directory=PUBLIC), name="public")
