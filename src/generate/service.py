"""answer_question(): the single entry point for the CLI and UI.

question → PII filter → intent → template route (no LLM) or
retrieve → min_score gate → LLM → post-check → Answer
"""

from __future__ import annotations

from datetime import datetime

from src.config import Settings, Source, get_settings, load_sources
from src.generate import prompts
from src.generate.intent import classify
from src.generate.llm import LLMClient, LLMError, get_llm_client
from src.generate.pii import PII_MESSAGE, detect_pii
from src.generate.postcheck import postcheck
from src.ingest.store import IngestRequired, read_manifest
from src.models import Answer
from src.retrieve.retriever import retrieve, write_log


def answer_question(
    text: str,
    *,
    settings: Settings | None = None,
    sources: list[Source] | None = None,
    llm: LLMClient | None = None,
) -> Answer:
    settings = settings or get_settings()
    sources = sources if sources is not None else load_sources()
    question = (text or "").strip()
    if not question:
        return Answer(status="empty", body=prompts.EMPTY)

    # Nothing about this question is logged, embedded or sent anywhere past this point.
    pii_kinds = detect_pii(question)
    if pii_kinds:
        return Answer(status="pii", body=PII_MESSAGE, intent="pii", flags=[f"pii:{kind}" for kind in pii_kinds])

    manifest = read_manifest(settings)
    if manifest is None:
        raise IngestRequired("No ingest manifest; run `python -m src.cli ingest`")
    last_updated = _format_date(manifest.get(settings.answer.last_updated_field) or manifest["last_ingest_at"])
    by_id = {source.id: source for source in sources}
    refusal_url = _source_url(by_id, settings.answer.refusal_url_source_id)
    fallback_url = _source_url(by_id, settings.answer.fallback_source_id)

    intent = classify(question, sources)
    common = {"intent": intent.intent, "scheme_id": intent.scheme_id, "last_updated": last_updated,
              "flags": [f"matched:{term}" for term in intent.matched]}

    if intent.intent == "advice":
        return Answer(status="refused", body=prompts.REFUSAL, source_url=refusal_url, **common)
    if intent.intent == "returns":
        if settings.answer.returns_source_id:
            url = _source_url(by_id, settings.answer.returns_source_id)
        else:
            url = by_id[intent.scheme_id].url if intent.scheme_id in by_id else fallback_url
        return Answer(status="returns", body=prompts.RETURNS, source_url=url, **common)
    if intent.intent == "out_of_scope":
        return Answer(status="out_of_scope", body=prompts.OUT_OF_SCOPE, source_url=fallback_url, **common)

    result = retrieve(question, intent=intent.intent, scheme_id=intent.scheme_id, settings=settings)
    common["query_id"] = result.query_id
    common["hits"] = result.hits
    if result.below_threshold:
        # Weak matches are noise: cite the named scheme's page, else the fallback page.
        url = by_id[intent.scheme_id].url if intent.scheme_id in by_id else fallback_url
        answer = Answer(status="not_found", body=prompts.NOT_FOUND, source_url=url, **common)
        answer.flags.append(f"below_min_score:{result.best_score:.3f}")
        return _logged(answer, settings)

    messages = prompts.build_messages(question, result.hits)
    try:
        client = llm or get_llm_client(settings)
        raw = client.generate(messages, temperature=settings.llm.temperature, max_tokens=settings.llm.max_tokens)
    except LLMError as exc:
        answer = Answer(status="error", body=prompts.LLM_UNAVAILABLE, prompt=messages, **common)
        answer.last_updated = None
        answer.flags.append(f"llm_error:{exc}")
        return _logged(answer, settings)

    checked = postcheck(raw, result.hits, max_sentences=settings.answer.max_sentences, refusal_url=refusal_url)
    answer = Answer(status=checked.status, body=checked.body, source_url=checked.source_url, prompt=messages, **common)
    answer.flags.extend(checked.flags)
    answer.flags.append(f"llm:{getattr(client, 'name', 'llm')}:{settings.llm.model}")
    return _logged(answer, settings, raw=raw)


def _logged(answer: Answer, settings: Settings, raw: str | None = None) -> Answer:
    """Add the outcome to the retrieval log written by retrieve()."""
    data = {"answer": {"status": answer.status, "body": answer.body, "source_url": answer.source_url,
                       "last_updated": answer.last_updated, "flags": answer.flags}}
    if raw is not None:
        data["llm_raw"] = raw
    if settings.answer.dump_prompts and answer.prompt:
        data["prompt"] = answer.prompt
    write_log(answer.query_id, data, settings)
    return answer


def _source_url(by_id: dict[str, Source], source_id: str) -> str:
    if source_id not in by_id:
        raise ValueError(f"answer config names source '{source_id}', which is not in sources.yaml")
    return by_id[source_id].url


def _format_date(timestamp: str) -> str:
    moment = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    return f"{moment.day} {moment.strftime('%b %Y')}"
