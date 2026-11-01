"""Fetch allowlisted public pages and store raw HTML, text, and metadata."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from src.config import Source, Settings, allowed_source_urls, get_settings, load_sources
from src.ingest.extract import extract_pdf_sections, extract_sections, parse_sections_file, sections_to_text
from src.models import RawMeta, Section


class FetchError(Exception):
    pass


def load_all(
    sources: list[Source] | None = None,
    settings: Settings | None = None,
) -> int:
    settings = settings or get_settings()
    sources = sources or load_sources()
    raw_dir = settings.resolve_path(settings.paths.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    allowlist = allowed_source_urls()
    results: list[RawMeta] = []
    required_failed = False

    with httpx.Client(
        follow_redirects=True,
        headers={"User-Agent": settings.loader.user_agent, "Accept": "text/html,application/xhtml+xml"},
        timeout=settings.loader.timeout_s,
    ) as client:
        for index, source in enumerate(sources):
            if index:
                time.sleep(settings.loader.request_delay_s)
            meta = _load_one(client, source, settings, raw_dir, allowlist)
            results.append(meta)
            status = "ok" if meta.error is None else "FAIL"
            print(
                f"{status:4}  {source.id:40}  "
                f"status={meta.http_status}  chars={meta.text_chars}  "
                f"sections={meta.section_count}  method={meta.fetch_method}  "
                f"{meta.error or ''}"
            )
            if meta.error and source.required:
                required_failed = True

    return 1 if required_failed else 0


def read_sections(source_id: str, settings: Settings | None = None) -> list[Section]:
    settings = settings or get_settings()
    path = settings.resolve_path(settings.paths.raw_dir) / f"{source_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"No extracted text for {source_id}: {path}")
    return parse_sections_file(path.read_text(encoding="utf-8"))


def _load_one(
    client: httpx.Client,
    source: Source,
    settings: Settings,
    raw_dir: Path,
    allowlist: set[str],
) -> RawMeta:
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if source.url not in allowlist:
        meta = RawMeta(
            source_id=source.id,
            title=source.title,
            url=source.url,
            fetched_at=fetched_at,
            error="URL is not in sources.yaml allowlist",
        )
        _write_meta(raw_dir, meta)
        return meta

    try:
        response, method = _fetch(client, source.url, settings)
    except Exception as exc:
        meta = RawMeta(
            source_id=source.id,
            title=source.title,
            url=source.url,
            fetched_at=fetched_at,
            fetch_method="http",
            error=str(exc),
        )
        _write_meta(raw_dir, meta)
        return meta

    content_type = response.headers.get("content-type", "")
    is_pdf = "pdf" in content_type.lower() or response.content[:5] == b"%PDF-"
    method_used = method
    if is_pdf:
        (raw_dir / f"{source.id}.pdf").write_bytes(response.content)
        try:
            sections = extract_pdf_sections(response.content, source.pages)
        except Exception as exc:
            sections = []
            print(f"pdf extraction failed for {source.id}: {exc}")
        text = sections_to_text(sections)
    else:
        html = response.text
        (raw_dir / f"{source.id}.html").write_text(html, encoding="utf-8")
        sections = extract_sections(html)
        text = sections_to_text(sections)

    if not is_pdf and len(text) < settings.loader.min_text_chars and settings.loader.use_browser_fallback:
        rendered = _fetch_browser(source.url, settings)
        if rendered:
            html = rendered
            (raw_dir / f"{source.id}.html").write_text(html, encoding="utf-8")
            sections = extract_sections(html)
            text = sections_to_text(sections)
            method_used = "browser"

    (raw_dir / f"{source.id}.txt").write_text(text, encoding="utf-8")
    error = None
    if not text.strip():
        error = "empty extraction"
    meta = RawMeta(
        source_id=source.id,
        title=source.title,
        url=source.url,
        fetched_at=fetched_at,
        http_status=response.status_code,
        content_type=content_type,
        fetch_method=method_used,
        text_chars=len(text),
        section_count=len(sections),
        error=error,
    )
    _write_meta(raw_dir, meta)
    return meta


def _fetch(client: httpx.Client, url: str, _settings: Settings) -> tuple[httpx.Response, str]:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = client.get(url)
            if response.status_code >= 500:
                last_error = FetchError(f"HTTP {response.status_code}")
                time.sleep(1.0)
                continue
            if response.status_code >= 400:
                raise FetchError(f"HTTP {response.status_code}")
            return response, "http"
        except httpx.RequestError as exc:
            last_error = FetchError(str(exc))
            time.sleep(1.0)
    raise last_error or FetchError("fetch failed")


def _fetch_browser(url: str, settings: Settings) -> str | None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed; skipping browser fallback")
        return None
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(user_agent=settings.loader.user_agent)
            page.goto(url, wait_until="networkidle", timeout=int(settings.loader.timeout_s * 1000))
            html = page.content()
            browser.close()
            return html
    except Exception as exc:
        print(f"browser fallback failed: {exc}")
        return None


def _write_meta(raw_dir: Path, meta: RawMeta) -> None:
    path = raw_dir / f"{meta.source_id}.meta.json"
    path.write_text(json.dumps(meta.model_dump(), indent=2) + "\n", encoding="utf-8")
