"""The deployed API ships only requirements.txt. Import it with ingest-only packages blocked
(as on Vercel) and a fresh store dir, and check it builds Chroma and answers."""

import subprocess
import sys
from pathlib import Path

import pytest

from src.ingest.store import read_manifest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = r'''
import importlib.abc, os, sys
BLOCK = ("bs4", "lxml", "pypdf", "sentence_transformers", "torch", "transformers", "kubernetes")
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in BLOCK:
            raise ImportError(f"{name} is not installed in the deployed API")
sys.meta_path.insert(0, Blocker())
sys.path.insert(0, os.getcwd())
from src.config import get_settings
settings = get_settings()
settings.chroma.persist_dir = sys.argv[1]
settings.paths.retrieval_log_dir = sys.argv[2]
from fastapi.testclient import TestClient
import api.index as app_module
client = TestClient(app_module.app)
answer = client.post("/api/ask", json={"question": "Should I buy HDFC Flexi Cap?"}).json()
assert answer["status"] == "refused", answer
from src.ingest.store import open_collection
print("vectors", open_collection(settings).count())
'''

pytestmark = pytest.mark.skipif(read_manifest() is None, reason="run `python -m src.cli ingest` first")


def test_api_starts_and_builds_store_without_ingest_only_packages(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path / "chroma"), str(tmp_path / "logs")],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "vectors " in result.stdout and not result.stdout.strip().endswith("vectors 0")
