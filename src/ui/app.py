"""Streamlit UI: a thin client over answer_question(). It never scrapes, chunks or embeds.

Run from the project root:  streamlit run src/ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.generate.service import answer_question  # noqa: E402
from src.ingest.embedder import get_model  # noqa: E402
from src.ingest.store import IngestRequired, open_collection, read_manifest  # noqa: E402
from src.models import Answer  # noqa: E402

WELCOME = (
    "Ask about publicly listed facts for HDFC Mutual Fund schemes on Groww: expense ratio, "
    "exit load, minimum SIP, lock-in, riskometer, benchmark and fund manager."
)
FACTS_ONLY = "**Facts-only. No investment advice.**"
DISCLAIMER = (
    "This assistant shares publicly available scheme facts only. It is not investment advice, "
    "not a recommendation to buy or sell, and not a substitute for the scheme information document, "
    "factsheet, or a SEBI-registered adviser."
)
EXAMPLES = [
    "What is the expense ratio of HDFC Large Cap Fund Direct Growth?",
    "What is the ELSS lock-in for HDFC ELSS Tax Saver Fund Direct Plan Growth?",
    "What is the minimum SIP for HDFC Mid Cap Fund Direct Growth?",
]
PII_PLACEHOLDER = "[message removed — contained personal details]"
STATUS_LABELS = {
    "answered": "Answered from sources",
    "not_found": "Not in sources",
    "refused": "Advice refused",
    "returns": "Returns not computed",
    "out_of_scope": "Out of scope",
    "pii": "Personal details rejected",
    "error": "Service error",
}


@st.cache_resource(show_spinner="Loading embedding model…")
def warm_up(model_name: str):
    """Load the embedder once per server process; later questions reuse it."""
    return get_model(model_name)


def corpus_status() -> tuple[dict | None, str | None]:
    """Return (manifest, problem). problem is None when the store is ready."""
    settings = get_settings()
    manifest = read_manifest(settings)
    try:
        open_collection(settings)
    except IngestRequired as exc:
        return manifest, str(exc)
    if manifest is None:
        return None, "No ingest manifest found."
    return manifest, None


def render_answer(answer: Answer) -> None:
    if answer.status == "error":
        st.error(answer.body)
    elif answer.status == "pii":
        st.warning(answer.body)
    else:
        st.markdown(answer.body)
    if answer.source_url:
        st.markdown(f"**Source:** [{answer.source_url}]({answer.source_url})")
    if answer.last_updated:
        st.markdown(f"**Last updated from sources:** {answer.last_updated}")
    render_pipeline(answer)


def render_pipeline(answer: Answer) -> None:
    with st.expander("Pipeline view", expanded=False):
        st.markdown(
            f"**Route:** {STATUS_LABELS.get(answer.status, answer.status)} · "
            f"**Intent:** `{answer.intent}` · **Scheme:** `{answer.scheme_id or '—'}`"
        )
        if answer.flags:
            st.markdown("**Checks:** " + ", ".join(f"`{flag}`" for flag in answer.flags))
        if answer.hits:
            st.markdown("**Retrieved chunks**")
            st.dataframe(
                [
                    {
                        "rank": rank,
                        "score": round(hit.score, 3),
                        "section": hit.section,
                        "chunk_id": hit.chunk_id,
                        "preview": hit.text.split("\n", 1)[-1][:200],
                    }
                    for rank, hit in enumerate(answer.hits, start=1)
                ],
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("No retrieval: this route answers from a fixed template without the LLM.")
        if answer.prompt:
            st.markdown("**Prompt sent to the LLM**")
            st.code("\n\n".join(f"[{m['role']}]\n{m['content']}" for m in answer.prompt), language=None)
        st.markdown("**Artifacts**")
        lines = ["- Raw pages: `data/raw/`", "- Chunks: `data/chunks/chunks.jsonl`",
                 "- Vectors: `data/embeddings/`, `data/chroma/`"]
        if answer.query_id:
            lines.append(f"- This query's log: `data/retrieval_logs/{answer.query_id}.json`")
        st.markdown("\n".join(lines))


def render_sidebar(manifest: dict | None) -> None:
    settings = get_settings()
    with st.sidebar:
        st.subheader("Corpus")
        if manifest:
            st.markdown(
                f"- Sources: **{manifest['source_count']}**\n"
                f"- Chunks: **{manifest['chunk_count']}**\n"
                f"- Last ingest: {manifest['last_ingest_at']}\n"
                f"- Embedding: `{manifest['embedding_model']}`\n"
                f"- Vector store: `{manifest['chroma_collection']}` ({manifest['chroma_distance']})"
            )
        else:
            st.markdown("Not built yet.")
        st.subheader("Model")
        st.markdown(f"- LLM: `{settings.llm.provider}` / `{settings.llm.model}`\n- top_k: {settings.retrieval.top_k}")
        st.divider()
        st.caption(DISCLAIMER)


def main() -> None:
    st.set_page_config(page_title="HDFC MF Facts Assistant", page_icon="📄", layout="centered")
    settings = get_settings()
    manifest, problem = corpus_status()
    render_sidebar(manifest)

    st.title("HDFC Mutual Fund Facts Assistant")
    st.write(WELCOME)
    st.info(FACTS_ONLY)

    if problem:
        st.error(f"The source corpus isn't ready: {problem}\n\nRun `python -m src.cli ingest` and reload.")
        st.caption(DISCLAIMER)
        return

    warm_up(settings.embedding.model)
    st.session_state.setdefault("history", [])

    st.caption("Try an example:")
    pending = None
    for column, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        if column.button(example, width="stretch"):
            pending = example

    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            render_answer(item["answer"])

    typed = st.chat_input("Ask a factual question about an HDFC scheme")
    question = pending or typed
    if question:
        with st.spinner("Looking it up…"):
            try:
                answer = answer_question(question)
            except IngestRequired as exc:
                st.error(str(exc))
                return
        shown = PII_PLACEHOLDER if answer.status == "pii" else question
        st.session_state.history.append({"question": shown, "answer": answer})
        with st.chat_message("user"):
            st.write(shown)
        with st.chat_message("assistant"):
            render_answer(answer)

    st.divider()
    st.caption(DISCLAIMER)


main()
