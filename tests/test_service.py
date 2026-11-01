import json

import pytest

from src.config import ChromaConfig, PathsConfig, RetrievalConfig, Settings
from src.generate.llm import LLMError
from src.generate.service import answer_question
from src.ingest.store import IngestRequired
from src.models import Hit
from src.retrieve.retriever import RetrievalResult
from tests.test_guardrails import SOURCES

HITS = [
    Hit(chunk_id="large::000", source_id="large", source_url="https://groww.in/large", section="Key facts",
        text="HDFC Large Cap Fund Direct Growth — Key facts\nExpense ratio: 1.02%", score=0.78),
]


class FakeLLM:
    name = "fake"

    def __init__(self, reply="The expense ratio is 1.02%.\nSource: https://groww.in/large", error=None):
        self.reply, self.error, self.calls = reply, error, []

    def generate(self, messages, *, temperature, max_tokens):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.reply


@pytest.fixture
def settings(tmp_path):
    settings = Settings(
        chroma=ChromaConfig(persist_dir=str(tmp_path / "chroma")),
        retrieval=RetrievalConfig(min_score=0.35),
        paths=PathsConfig(manifest_file=str(tmp_path / "manifest.json"), retrieval_log_dir=str(tmp_path / "logs")),
    )
    manifest = {"last_ingest_at": "2026-11-01T07:37:49Z", "sources_fetched_at": "2026-10-31T06:56:52Z"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return settings


@pytest.fixture
def fake_retrieve(monkeypatch, settings):
    calls = []

    def retrieve(question, *, intent, scheme_id, settings, best=0.78):
        from src.retrieve.retriever import write_log

        calls.append(question)
        write_log("q1", {"query": question, "hits": [hit.model_dump() for hit in HITS]}, settings)
        return RetrievalResult(query_id="q1", hits=HITS, best_score=best, below_threshold=best < 0.35)

    monkeypatch.setattr("src.generate.service.retrieve", retrieve)
    return calls


def ask(text, settings, llm):
    return answer_question(text, settings=settings, sources=SOURCES, llm=llm)


def test_factual_answer_goes_through_llm_and_is_logged(settings, fake_retrieve, tmp_path):
    llm = FakeLLM()
    answer = ask("What is the expense ratio of HDFC Large Cap Fund Direct Growth?", settings, llm)
    assert answer.status == "answered"
    assert answer.render() == (
        "The expense ratio is 1.02%.\nSource: https://groww.in/large\nLast updated from sources: 1 Nov 2026"
    )
    assert len(llm.calls) == 1
    assert "Expense ratio: 1.02%" in llm.calls[0][1]["content"]
    log = json.loads((tmp_path / "logs" / "q1.json").read_text(encoding="utf-8"))
    assert log["answer"]["status"] == "answered" and log["llm_raw"].startswith("The expense ratio")


@pytest.mark.parametrize(
    "text, status, url",
    [
        ("Should I buy HDFC Flexi Cap?", "refused", "https://investor.sebi.gov.in/ie"),
        ("Which of these funds is better for me?", "refused", "https://investor.sebi.gov.in/ie"),
        ("What were the 5-year returns of HDFC Mid Cap?", "returns", "https://files.hdfcfund.com/factsheet.pdf"),
        ("What is the expense ratio of SBI Bluechip?", "out_of_scope", "https://groww.in/amc"),
    ],
)
def test_template_routes_skip_retrieval_and_llm(settings, fake_retrieve, text, status, url):
    llm = FakeLLM()
    answer = ask(text, settings, llm)
    assert (answer.status, answer.source_url) == (status, url)
    assert answer.last_updated == "1 Nov 2026"
    assert llm.calls == [] and fake_retrieve == []


def test_pii_rejected_without_log_retrieval_or_llm(settings, fake_retrieve, tmp_path):
    llm = FakeLLM()
    answer = ask("My PAN is ABCDE1234F, what is my SIP?", settings, llm)
    assert answer.status == "pii" and answer.source_url is None
    assert "ABCDE1234F" not in answer.model_dump_json()
    assert llm.calls == [] and fake_retrieve == []
    assert not (tmp_path / "logs").exists()


def test_below_min_score_returns_not_found_without_llm(settings, monkeypatch):
    monkeypatch.setattr(
        "src.generate.service.retrieve",
        lambda *a, **k: RetrievalResult(query_id="q2", hits=HITS, best_score=0.2, below_threshold=True),
    )
    llm = FakeLLM()
    answer = ask("How do I download a capital-gains statement?", settings, llm)
    assert (answer.status, answer.source_url) == ("not_found", "https://groww.in/amc")
    assert llm.calls == []
    named = ask("What is the dividend history of HDFC Mid Cap?", settings, llm)
    assert (named.status, named.source_url) == ("not_found", "https://groww.in/mid")


def test_llm_failure_gives_error_not_a_fact(settings, fake_retrieve):
    answer = ask("Expense ratio of HDFC Large Cap?", settings, FakeLLM(error=LLMError("bad key")))
    assert answer.status == "error" and "1.02" not in answer.body
    assert answer.source_url is None and answer.last_updated is None


def test_empty_question_and_missing_ingest(settings, tmp_path):
    assert ask("   ", settings, FakeLLM()).status == "empty"
    (tmp_path / "manifest.json").unlink()
    with pytest.raises(IngestRequired):
        ask("Expense ratio of HDFC Large Cap?", settings, FakeLLM())


def test_returns_falls_back_to_scheme_page_without_factsheet(settings, fake_retrieve):
    settings.answer.returns_source_id = None
    answer = ask("What were the 5-year returns of HDFC Mid Cap?", settings, FakeLLM())
    assert answer.source_url == "https://groww.in/mid"


def test_last_updated_field_is_configurable(settings, fake_retrieve):
    settings.answer.last_updated_field = "sources_fetched_at"
    assert ask("Should I buy HDFC Flexi Cap?", settings, FakeLLM()).last_updated == "31 Oct 2026"


def test_llm_failure_reports_a_safe_reason_code(settings, fake_retrieve):
    answer = ask("Expense ratio of HDFC Large Cap?", settings, FakeLLM(error=LLMError("401 key=secret", reason="auth_failed")))
    assert "llm_reason:auth_failed" in answer.flags


def test_missing_key_is_reported_as_not_configured(settings, fake_retrieve, monkeypatch):
    settings.groq_api_key = None
    answer = answer_question("Expense ratio of HDFC Large Cap?", settings=settings, sources=SOURCES)
    assert answer.status == "error" and "llm_reason:not_configured" in answer.flags


@pytest.mark.parametrize(
    "question, reply, url",
    [
        ("Who is the trustee of the scheme?", "I couldn't find that in my sources.", "https://groww.in/amc"),
        ("What is the dividend history of HDFC Mid Cap?", "I couldn't find that in my sources.", "https://groww.in/mid"),
        ("What is the AUM of HDFC Mid Cap?", "The AUM is ₹99,999 Cr.\nSource: https://groww.in/large", "https://groww.in/mid"),
    ],
)
def test_model_not_found_cites_scheme_or_fallback_not_the_unrelated_top_hit(settings, fake_retrieve, question, reply, url):
    answer = ask(question, settings, FakeLLM(reply=reply))
    assert answer.status == "not_found"
    assert answer.source_url == url
