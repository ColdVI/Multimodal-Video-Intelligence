"""T10: Playwright UI smoke — drives the redesigned Gradio UI against a live stack
and captures the 7 screenshots required by UI_REDESIGN_TALIMATI.md §7.

Needs RUN_FAZ8_INTEGRATION=1 (see faz8_support.readiness) plus the UI container
reachable at UI_URL (default http://localhost:7860, matching docker-compose.faz7.yml).
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from faz8_support import readiness


playwright_sync_api = pytest.importorskip("playwright.sync_api")
sync_playwright = playwright_sync_api.sync_playwright

UI_URL = os.getenv("UI_URL", "http://localhost:7860").rstrip("/")
ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "ui_redesign"


@pytest.fixture(scope="module")
def page():
    readiness("system")
    try:
        httpx.get(UI_URL, timeout=5).raise_for_status()
    except Exception as exc:
        pytest.skip(f"SKIPPED: UI not reachable at {UI_URL} ({exc})")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        browser_page = browser.new_page(viewport={"width": 1440, "height": 1000})
        browser_page.goto(UI_URL, wait_until="load", timeout=30000)
        browser_page.wait_for_timeout(2000)
        yield browser_page
        browser.close()


def test_t10_home_empty_state(page):
    page.screenshot(path=str(ARTIFACT_DIR / "home_empty.png"), full_page=True)
    assert page.locator("#mvi-results .empty-state__title").inner_text() == "Henüz sorgu yapılmadı"
    assert page.locator("#mvi-status-badge .status-badge").count() == 1


def test_t10_search_results(page):
    page.locator("#mvi-search-button").click()
    page.wait_for_selector("#mvi-results .result-card", timeout=30000)
    page.wait_for_timeout(500)
    page.screenshot(path=str(ARTIFACT_DIR / "search_results.png"), full_page=True)
    assert page.locator("#mvi-results .result-card").count() == 10
    assert page.locator("#mvi-results .result-card--primary").count() == 1
    # diagnostics §2.7: 11 canonical fields + embedding mode, all rendered
    assert page.locator(".diagnostics-item").count() >= 11


def test_t10_result_expanded(page):
    page.locator("#mvi-detail-selector input").click()
    page.wait_for_timeout(300)
    options = page.locator("ul.options li")
    if options.count() > 1:
        options.nth(1).click()
    else:
        page.keyboard.press("Escape")
    page.wait_for_timeout(500)
    page.screenshot(path=str(ARTIFACT_DIR / "result_expanded.png"), full_page=True)
    assert page.locator("#mvi-detail .diagnostics-item").count() > 0


def test_t10_advanced_settings(page):
    page.locator("text=Advanced Search Settings").click()
    page.wait_for_timeout(400)
    page.screenshot(path=str(ARTIFACT_DIR / "advanced_settings.png"), full_page=True)
    assert page.locator(".pattern-not-implemented").count() == 1


def test_t10_comparison(page):
    page.locator('button[role="tab"]', has_text="Karşılaştır").click()
    page.wait_for_timeout(400)
    page.locator('button:has-text("Karşılaştır")').last.click()
    page.wait_for_selector(".comparison-card", timeout=30000)
    page.wait_for_timeout(300)
    page.screenshot(path=str(ARTIFACT_DIR / "comparison.png"), full_page=True)
    assert page.locator(".comparison-card").count() >= 1
    page.locator('button[role="tab"]', has_text="Ara").click()
    page.wait_for_timeout(300)


def test_t10_synthetic_warning(page):
    badge = page.locator("#mvi-status-badge")
    badge.screenshot(path=str(ARTIFACT_DIR / "synthetic_warning.png"))
    assert "SENTETİK" in badge.inner_text()


def test_t10_no_media_state(page):
    media = page.locator(".media-slot").first
    media.screenshot(path=str(ARTIFACT_DIR / "no_media_state.png"))
    assert "medya önizlemesi bu ortamda servis edilmiyor" in media.inner_text()
