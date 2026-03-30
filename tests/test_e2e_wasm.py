"""End-to-end test: export a marimo notebook as WASM and verify it loads in a real browser.

Requires: pip install -e ".[e2e]" && playwright install chromium
"""

import os
import socket
import subprocess
import sys
import time

import pytest

pw = pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

NOTEBOOK = os.path.join(os.path.dirname(__file__), "..", "examples", "lj_dimer_precompute.py")


def _free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def wasm_server(tmp_path_factory):
    """Precompute cache, export WASM notebook, and serve over HTTP."""
    out = str(tmp_path_factory.mktemp("wasm"))
    nb = os.path.abspath(NOTEBOOK)

    # 1. Run the notebook to generate the persistent_cache files in public/
    #    Use --no-sandbox to run in the current env (PEP 723 metadata
    #    would otherwise trigger uv sandbox creation).
    result = subprocess.run(
        [sys.executable, "-m", "marimo", "export", "html", nb,
         "--no-include-code", "--no-sandbox", "-o", os.devnull],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"Static export failed:\n{result.stderr}"

    # 2. Export as WASM HTML (bundles public/ automatically)
    result = subprocess.run(
        [sys.executable, "-m", "marimo", "export", "html-wasm", nb,
         "-o", os.path.join(out, "index.html"),
         "--mode", "run", "--no-show-code", "--no-sandbox", "-f"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"WASM export failed:\n{result.stderr}"

    # Verify manifest was bundled
    manifest = os.path.join(out, "public", "__marimo_precompute__", "manifest.json")
    assert os.path.exists(manifest), f"manifest.json not bundled in export: {os.listdir(out)}"

    # 3. Start HTTP server
    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--directory", out],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    # Give it a moment to bind
    time.sleep(1)
    assert server.poll() is None, "HTTP server failed to start"

    yield f"http://localhost:{port}/index.html"

    server.terminate()
    server.wait(timeout=5)


def test_wasm_notebook_loads_without_errors(wasm_server):
    """The WASM notebook loads in a real browser and renders output without Python errors."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        console_messages = []
        page.on("console", lambda msg: console_messages.append(
            {"type": msg.type, "text": msg.text}
        ))

        page.goto(wasm_server, wait_until="commit")

        # Wait for marimo to finish — look for rendered plot images.
        # Marimo WASM takes 60-180 seconds to load Pyodide + packages + run cells.
        try:
            page.wait_for_selector("img[src^='data:image']", timeout=240_000)
        except Exception:
            # Dump console for debugging
            stderr_msgs = [m["text"] for m in console_messages if m["type"] == "error"]
            all_msgs = [f"[{m['type']}] {m['text'][:200]}" for m in console_messages[-30:]]
            pytest.fail(
                f"Timed out waiting for notebook output.\n"
                f"Console errors ({len(stderr_msgs)}):\n"
                + "\n".join(stderr_msgs[:20])
                + f"\n\nLast 30 console messages:\n"
                + "\n".join(all_msgs)
            )

        # Check for Python tracebacks in console
        tracebacks = [
            m["text"] for m in console_messages
            if "Traceback" in m["text"] or "raise " in m["text"]
        ]
        assert not tracebacks, (
            f"Python errors in browser console:\n" + "\n".join(tracebacks[:5])
        )

        # Verify at least one plot rendered (matplotlib outputs base64 images)
        images = page.query_selector_all("img[src^='data:image']")
        assert len(images) >= 1, (
            f"Expected at least one rendered plot, found {len(images)} images"
        )

        browser.close()
