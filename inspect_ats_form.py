"""
inspect_ats_form.py — Verify file-upload and cover-letter selectors
against a real, live ATS application page.

This exists to answer one question: do the selectors used in
agents/application_agent.py._upload_documents() actually match real
DOM on Greenhouse / Ashby / Lever forms, or were they guessed?

Usage:
    source .venv/bin/activate
    python inspect_ats_form.py "https://jobs.ashbyhq.com/openai/<job-id>/application"
    python inspect_ats_form.py "https://job-boards.greenhouse.io/<company>/jobs/<id>"

What it does:
    1. Opens the page in a real (headed, so you can see it) browser
    2. Waits for the page to fully hydrate
    3. Lists every input/textarea/select on the page with its
       name/id/type/placeholder/aria-label
    4. Flags which ones our current selectors would actually match
    5. Takes a full-page screenshot for manual cross-check
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

# ── The exact selectors currently used in application_agent.py ─────────────
# Keep these in sync manually — the point is to test what's ACTUALLY there.

RESUME_SELECTORS = [
    "input[type='file'][name*='resume' i]",
    "input[type='file'][id*='resume' i]",
    "input[type='file'][name*='cv' i]",
]
COVER_FILE_SELECTORS = [
    "input[type='file'][name*='cover' i]",
    "input[type='file'][id*='cover' i]",
]
COVER_TEXTAREA_SELECTORS = [
    "textarea[name*='cover' i]",
    "textarea[id*='cover' i]",
]
GENERIC_FILE_INPUT = "input[type='file']"


async def inspect(url: str) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # headed — watch it load
        page = await browser.new_page()

        print(f"\nNavigating to: {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(2)  # let any lazy-mounted React fields settle

        # ── 1. Dump every relevant form element ─────────────────────────
        print("\n" + "=" * 70)
        print("ALL FORM ELEMENTS ON PAGE")
        print("=" * 70)

        elements = await page.locator("input, textarea, select").all()
        if not elements:
            print("No form elements found — page may still be loading, or")
            print("the form may be behind a click (e.g. 'Apply' button).")

        for el in elements:
            tag = await el.evaluate("e => e.tagName.toLowerCase()")
            el_type = await el.get_attribute("type") or ("textarea" if tag == "textarea" else "")
            name = await el.get_attribute("name")
            el_id = await el.get_attribute("id")
            placeholder = await el.get_attribute("placeholder")
            aria_label = await el.get_attribute("aria-label")
            print(
                f"  <{tag}> type={el_type!r:12} name={name!r:25} "
                f"id={el_id!r:25} placeholder={placeholder!r:20} aria-label={aria_label!r}"
            )

        # ── 2. Check our actual selectors against this page ─────────────
        print("\n" + "=" * 70)
        print("SELECTOR MATCH CHECK (what application_agent.py would find)")
        print("=" * 70)

        async def check(label: str, selectors: list[str]) -> None:
            for sel in selectors:
                count = await page.locator(sel).count()
                status = "MATCH" if count > 0 else "-"
                print(f"  [{status}] {label}: `{sel}` -> {count} element(s)")

        await check("Resume file input", RESUME_SELECTORS)
        await check("Cover letter file input", COVER_FILE_SELECTORS)
        await check("Cover letter textarea", COVER_TEXTAREA_SELECTORS)

        generic_count = await page.locator(GENERIC_FILE_INPUT).count()
        print(f"  [info] Generic `input[type='file']` on page: {generic_count} element(s)")

        # ── 3. Screenshot for manual visual cross-check ──────────────────
        out_dir = Path("data/inspection")
        out_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = out_dir / "form_inspection.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"\nFull-page screenshot saved to: {screenshot_path}")
        print("(Open it and visually compare against the element list above.)")

        await browser.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_ats_form.py <application_url>")
        print("\nExample URLs to try (replace with a currently live posting):")
        print("  Ashby:      https://jobs.ashbyhq.com/<company>/<job-id>/application")
        print("  Greenhouse: https://job-boards.greenhouse.io/<company>/jobs/<job-id>")
        print("  Lever:      https://jobs.lever.co/<company>/<job-id>/apply")
        sys.exit(1)

    asyncio.run(inspect(sys.argv[1]))