#!/usr/bin/env python3
"""Capture the screenshots used by the reference manual.

Launches the real Streamlit app, drives it with Playwright, and writes PNGs to
``docs/_static/screenshots/manual/``. Captures that need a language model or the
dataset graph make live calls, so the run takes several minutes.

A credential guard runs before every capture: the values of the credential
variables in ``.env`` are checked against the page's rendered text and every
visible input value. A hit aborts that capture instead of writing a PNG that
would carry a key into a distributed PDF.

Usage::

    python scripts/capture_screenshots.py                # everything
    python scripts/capture_screenshots.py --only curator # curator captures only
    python scripts/capture_screenshots.py --keep-server  # reuse a running app
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "_static" / "screenshots" / "manual"
PORT = 8599
URL = f"http://localhost:{PORT}"

# A short, deliberately incomplete description. A mediocre input makes both the
# rubric breakdown and the enhancement visible, which a well-written one would not.
SAMPLE_DESCRIPTION = (
    "A North Sea sandstone was cleaned, dried, and imaged using micro-CT. "
    "The sample was then cut and polished for 2D mineral mapping. "
    "The mineral map was registered to the 3D image and segmented into groups. "
    "The data is uploaded as netCDF blocks."
)

SAMPLE_FEEDBACK = (
    "Add the imaging resolution, the scan duration, the porosity and permeability "
    "values, and the number of mineral groups the segmentation produced."
)


# ---------------------------------------------------------------------------
# Credential guard
# ---------------------------------------------------------------------------


def env_secrets() -> list[tuple[str, str]]:
    env = REPO / ".env"
    if not env.exists():
        return []
    out = []
    for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip().strip("'\"")
        if len(value) < 8 or not re.search(r"(KEY|TOKEN|PASSWORD|SECRET)$", name):
            continue
        out.append((name, value))
    return out


SECRETS = env_secrets()


def assert_no_secrets(page, label: str) -> None:
    """Fail the capture if any .env credential is visible on the page."""
    text = page.evaluate("() => document.body.innerText || ''")
    values = page.evaluate(
        "() => Array.from(document.querySelectorAll('input,textarea'))"
        ".map(e => e.value || '').join('\\n')"
    )
    haystack = f"{text}\n{values}"
    for name, value in SECRETS:
        if value in haystack:
            raise RuntimeError(
                f"credential guard: value of {name} is visible on screen for '{label}'"
            )


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def port_open() -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def start_server() -> subprocess.Popen:
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "rocco_ui.py",
            "--server.port", str(PORT),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=str(REPO),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    for _ in range(120):
        if port_open():
            time.sleep(3)  # let the first script run finish
            return proc
        time.sleep(1)
    proc.terminate()
    raise SystemExit("Streamlit did not start")


# ---------------------------------------------------------------------------
# Page helpers
# ---------------------------------------------------------------------------


def settle(page, timeout: int = 300) -> None:
    """Wait until Streamlit stops running a script."""
    page.wait_for_timeout(700)
    deadline = time.time() + timeout
    while time.time() < deadline:
        running = page.evaluate(
            "() => !!document.querySelector('[data-testid=\"stStatusWidget\"]')"
            " || !!document.querySelector('.stSpinner')"
        )
        if not running:
            page.wait_for_timeout(900)
            hide_chrome(page)
            page.wait_for_timeout(150)
            return
        page.wait_for_timeout(1000)
    print("      ! settle() timed out; capturing anyway")


def shoot(page, name: str, locator=None, label: str = "") -> Path | None:
    assert_no_secrets(page, label or name)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / name
    target = locator if locator is not None else page
    try:
        target.screenshot(path=str(dest))
    except Exception as exc:  # noqa: BLE001
        print(f"      ! {name}: {exc}")
        return None
    size = dest.stat().st_size
    print(f"      + {name} ({size // 1024} KB)")
    return dest


# Streamlit's toolbar, "Deploy" button, and status widget are host chrome, not
# part of Rocco. They add noise to a printed figure and date it to a Streamlit
# version, so they are hidden before any capture.
HIDE_CHROME = """
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
#MainMenu, header, footer { display: none !important; visibility: hidden !important; }
[data-testid="stMain"], [data-testid="stMainBlockContainer"] { overflow: visible !important; }
"""


def hide_chrome(page) -> None:
    page.add_style_tag(content=HIDE_CHROME)


def main_area(page):
    # stMainBlockContainer is the stable id across recent Streamlit versions;
    # stMain is absent in some of them, which silently resolves to no element and
    # then fails as a screenshot timeout rather than a missing-selector error.
    return page.locator('[data-testid="stMainBlockContainer"]').first


def column(page, index: int):
    """One of Streamlit's side-by-side columns, for a tighter figure crop."""
    cols = page.locator('[data-testid="stColumn"]')
    return cols.nth(index) if cols.count() > index else main_area(page)


def goto_page(page, name: str) -> None:
    page.get_by_role("button", name=name, exact=True).first.click()
    settle(page)


# ---------------------------------------------------------------------------
# Captures
# ---------------------------------------------------------------------------


def capture_curator(page) -> None:
    print("  Description Curator")
    goto_page(page, "Description Curator")
    shoot(page, "01_navigation.png", label="navigation")
    shoot(page, "02_curator_empty.png", main_area(page), label="curator empty state")

    page.locator("textarea").first.fill(SAMPLE_DESCRIPTION)
    page.get_by_role("button", name="Evaluate Description").click()
    settle(page)
    shoot(page, "03_evaluation_results.png", column(page, 0), label="evaluation results")

    uploader = page.locator('[data-testid="stFileUploader"]').first
    if uploader.count():
        # A real paper makes the enhancement produce document citations, which is
        # the part of the output the manual needs to show.
        paper = REPO / "data" / "DPMP-461.pdf"
        if paper.exists():
            page.locator('[data-testid="stFileUploader"] input[type="file"]').first.set_input_files(
                str(paper)
            )
            settle(page)
        shoot(page, "04_upload.png", uploader, label="document uploader")

    feedback = page.get_by_role("textbox", name=re.compile("Provide feedback", re.I))
    if not feedback.count():
        feedback = page.locator("textarea").last
    feedback.first.fill(SAMPLE_FEEDBACK)
    # Streamlit commits a text_area on blur, so the value must be flushed before
    # the enhance button leaves its disabled state.
    feedback.first.press("Tab")
    settle(page)

    clicked = False
    for label in ("Enhance with Rocco", "Enhance Description", "Enhance"):
        button = page.get_by_role("button", name=re.compile(label, re.I))
        if button.count() and button.first.is_enabled():
            button.first.click()
            clicked = True
            break
    if not clicked:
        print("      ! enhance button never became enabled; skipping 05-07")
        return
    settle(page)
    shoot(page, "05_enhanced_citations.png", main_area(page), label="enhanced description")

    adopt = page.get_by_role("button", name=re.compile("Adopt Rocco", re.I))
    if adopt.count():
        controls = page.locator('[data-testid="stHorizontalBlock"]').filter(
            has=page.get_by_role("button", name=re.compile("Adopt Rocco", re.I))
        )
        shoot(page, "06_accept_reject.png",
              controls.first if controls.count() else main_area(page),
              label="accept or reject")
        adopt.first.click()
        settle(page)

    # The context manager is nested under `if st.session_state.evaluation:`, and
    # adopting clears the evaluation. Re-evaluating the now-enhanced description
    # brings it back, which is also the real iterate-again flow.
    evaluate = page.get_by_role("button", name="Evaluate Description")
    if evaluate.count():
        evaluate.first.click()
        settle(page)

    manage = page.locator('summary, [data-testid="stExpander"] summary').filter(
        has_text=re.compile("Manage Context", re.I)
    )
    if not manage.count():
        manage = page.get_by_text(re.compile("Manage Context", re.I))
    if manage.count():
        manage.first.click()
        settle(page)
        expander = page.locator('[data-testid="stExpander"]').filter(
            has_text=re.compile("Manage Context", re.I)
        )
        shoot(page, "07_manage_context.png",
              expander.first if expander.count() else main_area(page),
              label="manage context")
    else:
        print("      ! Manage Context expander not found; skipping 07")


def ask(page, question: str) -> bool:
    box = page.get_by_placeholder(re.compile("Ask about datasets", re.I))
    if not box.count():
        print("      ! chat input not found")
        return False
    box.first.fill(question)
    box.first.press("Enter")
    settle(page)
    return True


def last_answer(page):
    """The most recent assistant message, so a long thread stays legible in print."""
    messages = page.locator('[data-testid="stChatMessage"]')
    return messages.last if messages.count() else main_area(page)


def capture_assistant(page) -> None:
    print("  General Assistant")
    goto_page(page, "General Assistant")
    shoot(page, "08_assistant_empty.png", main_area(page), label="assistant empty state")

    # A search turn, then a follow-up that resolves against its results.
    if ask(page, "Show me datasets from coal samples"):
        shoot(page, "09_dataset_search.png", main_area(page), label="dataset search")
    if ask(page, "Tell me more about the first one"):
        shoot(page, "10_dataset_profile.png", last_answer(page), label="dataset profile")

    # Reload for a clean thread, so the content-reasoning figure is one exchange.
    page.reload(wait_until="domcontentloaded")
    settle(page)
    goto_page(page, "General Assistant")
    if ask(page, "Which datasets have both raw and segmented images?"):
        shoot(page, "11_content_reasoning.png", last_answer(page), label="content reasoning")
    if ask(page, "Of these, which are sandstone?"):
        shoot(page, "12_multi_turn.png", last_answer(page), label="multi-turn refinement")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["curator", "assistant"])
    parser.add_argument("--keep-server", action="store_true")
    args = parser.parse_args()

    print(f"Credential guard armed with {len(SECRETS)} .env value(s)")

    proc = None
    if port_open():
        print(f"Reusing the app already listening on {PORT}")
    else:
        print(f"Starting Streamlit on {PORT}")
        proc = start_server()

    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            context = browser.new_context(
                viewport={"width": 1440, "height": 2200}, device_scale_factor=2
            )
            page = context.new_page()
            page.goto(URL, wait_until="domcontentloaded")
            settle(page)

            if args.only != "assistant":
                capture_curator(page)
            if args.only != "curator":
                capture_assistant(page)

            browser.close()
    finally:
        if proc is not None and not args.keep_server:
            proc.terminate()

    written = sorted(OUT.glob("*.png"))
    print(f"\n{len(written)} screenshot(s) in {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
