"""API + static UI tests. Template routes only, so no LLM call; skipped if ingest hasn't run."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.ingest.store import read_manifest

pytestmark = pytest.mark.skipif(read_manifest() is None, reason="run `python -m src.cli ingest` first")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    from api.index import app

    return TestClient(app)


def test_index_page_has_required_elements(client):
    page = client.get("/")
    assert page.status_code == 200
    html = page.text
    assert "Facts-only. No investment advice." in html
    assert "not a substitute for the scheme information document" in html
    assert 'id="examples"' in html and 'id="ask-form"' in html


def test_status_reports_corpus_and_three_examples(client):
    info = client.get("/api/status").json()
    assert info["ready"] is True and info["chunks"] > 0
    assert len(info["examples"]) == 3
    assert all("HDFC" in example for example in info["examples"])


def test_advice_question_returns_refusal_with_education_link(client):
    answer = client.post("/api/ask", json={"question": "Should I buy HDFC Flexi Cap?"}).json()
    assert answer["status"] == "refused"
    assert "not buy/sell or portfolio advice" in answer["body"]
    assert answer["source_url"] == "https://investor.sebi.gov.in/iematerial.html"
    assert answer["last_updated"]
    assert answer["hits"] == [] and answer["prompt"] is None


def test_returns_question_links_factsheet(client):
    answer = client.post("/api/ask", json={"question": "What were the 5-year returns of HDFC Mid Cap?"}).json()
    assert answer["status"] == "returns"
    assert "Factsheet" in answer["source_url"]


def test_pii_is_rejected_and_not_echoed(client):
    response = client.post("/api/ask", json={"question": "My PAN is ABCDE1234F, what is my SIP?"})
    assert response.json()["status"] == "pii"
    assert "ABCDE1234F" not in response.text


def test_overlong_question_is_rejected(client):
    assert client.post("/api/ask", json={"question": "x" * 501}).status_code == 422


def test_missing_corpus_returns_503(client, monkeypatch):
    from src.ingest.store import IngestRequired

    def raise_ingest(_question):
        raise IngestRequired("No ingest manifest")

    monkeypatch.setattr("api.index.answer_question", raise_ingest)
    assert client.post("/api/ask", json={"question": "Expense ratio of HDFC Large Cap?"}).status_code == 503
