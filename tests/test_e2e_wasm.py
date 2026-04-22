"""End-to-end test: build marimo-precompute as a wheel, serve it locally,
export a notebook as WASM that installs from that local wheel, and verify
it loads in a real browser.

Why build from a wheel instead of letting Pyodide fetch from PyPI? Because
the notebook's default PEP 723 dep resolves to whatever is on PyPI at
page-load time. That's fine for users but terrible for CI — you can ship
a broken release and the e2e still passes against the previous good one.
This fixture exercises the *code in this checkout*.

Requires: pip install -e ".[e2e]" && playwright install chromium
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pw = pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "examples" / "lj_dimer_precompute.py"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _build_wheel(tmpdir: Path) -> Path:
    """Build the current checkout into a wheel under tmpdir and return its path."""
    out_dir = tmpdir / "wheel-build"
    out_dir.mkdir()
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out_dir), str(REPO)],
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(out_dir.glob("marimo_precompute-*.whl"))
    assert wheels, f"No wheel built into {out_dir}: {list(out_dir.iterdir())}"
    return wheels[0]


def _rewrite_pep723_to_local_wheel(src: Path, dst: Path, wheel_url: str) -> None:
    """Copy src to dst, replacing the marimo-precompute PEP 723 dep with the
    given URL. Matches ``"marimo-precompute..."`` in the PEP 723 block."""
    text = src.read_text()
    new = re.sub(
        r'"marimo-precompute[^"]*"',
        f'"marimo-precompute @ {wheel_url}"',
        text,
        count=1,
    )
    assert new != text, f"PEP 723 line not found in {src}"
    dst.write_text(new)


@pytest.fixture(scope="module")
def wasm_server(tmp_path_factory):
    """Build wheel, rewrite notebook PEP 723 to install from it, export
    WASM, serve everything from a single HTTP origin, yield the URL."""
    staging = tmp_path_factory.mktemp("wasm-e2e")

    wheel = _build_wheel(staging)

    port = _free_port()
    dist_dir = staging / "dist"
    dist_dir.mkdir()
    shutil.copy(wheel, dist_dir / wheel.name)
    wheel_url = f"http://127.0.0.1:{port}/dist/{wheel.name}"

    modified_nb = staging / "notebook.py"
    _rewrite_pep723_to_local_wheel(NOTEBOOK, modified_nb, wheel_url)

    env = os.environ.copy()
    env.pop("PYTHONDONTWRITEBYTECODE", None)

    # 1. Native export runs the notebook → populates public/ next to notebook
    result = subprocess.run(
        [
            sys.executable, "-m", "marimo", "export", "html",
            str(modified_nb),
            "--no-include-code", "--no-sandbox",
            "-o", os.devnull,
        ],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert result.returncode == 0, f"html export failed:\n{result.stderr}"

    # 2. WASM export bundles public/ next to the output HTML
    out_dir = staging / "out"
    out_dir.mkdir()
    result = subprocess.run(
        [
            sys.executable, "-m", "marimo", "export", "html-wasm",
            str(modified_nb),
            "-o", str(out_dir / "index.html"),
            "--mode", "run", "--no-show-code", "--no-sandbox", "-f",
        ],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert result.returncode == 0, f"html-wasm export failed:\n{result.stderr}"

    # 3. Stage everything under a single origin so cross-origin fetches
    #    (wheel + notebook cache + index.html) all succeed without CORS
    #    headers.
    serve_root = staging / "serve"
    serve_root.mkdir()
    shutil.copytree(dist_dir, serve_root / "dist")
    shutil.copy(out_dir / "index.html", serve_root / "index.html")
    if (out_dir / "public").exists():
        shutil.copytree(out_dir / "public", serve_root / "public")
    for extra in ("assets", "files"):
        src = out_dir / extra
        if src.exists():
            shutil.copytree(src, serve_root / extra)

    # 4. Sanity-check the manifest was bundled
    manifest = serve_root / "public" / "__marimo_precompute__" / "manifest.json"
    assert manifest.exists(), (
        f"manifest.json not bundled: {list(serve_root.rglob('manifest.json'))}"
    )

    # 5. Start HTTP server
    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port),
         "--bind", "127.0.0.1", "--directory", str(serve_root)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    time.sleep(1)
    assert server.poll() is None, "HTTP server failed to start"

    yield f"http://127.0.0.1:{port}/index.html"

    server.terminate()
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()


def test_wasm_notebook_loads_without_errors(wasm_server):
    """The WASM notebook loads from our locally-built wheel and renders
    output without Python tracebacks."""
    t0 = time.time()
    print(f"\n[test] navigating to {wasm_server}")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        console_messages: list[dict] = []
        page.on("console", lambda msg: console_messages.append(
            {"type": msg.type, "text": msg.text}
        ))

        ready = {"seen": False}
        page.on(
            "console",
            lambda msg: ready.__setitem__("seen", True)
            if "[marimo-precompute] ready" in msg.text
            else None,
        )

        page.goto(wasm_server, wait_until="commit")
        print(f"[test] page committed at t={time.time()-t0:.1f}s")

        # Wait until our loader's install() has completed in Pyodide.
        # This proves the wheel was installed from our local HTTP server
        # and not from PyPI, rules out a silently-failing install(), and
        # ensures the image we assert on isn't the static export's
        # pre-rendered copy.
        ready_deadline = time.time() + 240
        while time.time() < ready_deadline and not ready["seen"]:
            page.wait_for_timeout(500)
        print(f"[test] ready={ready['seen']} at t={time.time()-t0:.1f}s")
        assert ready["seen"], (
            "Never saw [marimo-precompute] ready — loader install() either "
            "failed silently or the wheel wasn't loaded. Last 20 console "
            "messages:\n"
            + "\n".join(
                f"  [{m['type']}] {m['text'][:200]}"
                for m in console_messages[-20:]
            )
        )

        try:
            page.wait_for_selector("img[src^='data:image']", timeout=240_000)
        except Exception:
            stderr_msgs = [m["text"] for m in console_messages if "STDERR" in m["text"]]
            all_msgs = [f"[{m['type']}] {m['text'][:200]}" for m in console_messages[-30:]]
            pytest.fail(
                f"Timed out waiting for notebook output.\n"
                f"STDERR lines ({len(stderr_msgs)}):\n"
                + "\n".join(stderr_msgs[:20])
                + "\n\nLast 30 console messages:\n"
                + "\n".join(all_msgs)
            )

        # Fail on any traceback, cache exception, or "Invalid method" error —
        # these are the signals of the WASM-integration bugs we've shipped
        # in past releases.
        failure_patterns = (
            "Traceback",
            "CacheException",
            "Invalid method",
            "install failed",
            "restore_cache failed",
        )
        failures = [
            m["text"] for m in console_messages
            if any(p in m["text"] for p in failure_patterns)
        ]
        assert not failures, (
            "Python errors in browser console:\n"
            + "\n".join(f"  {f[:300]}" for f in failures[:5])
        )

        images = page.query_selector_all("img[src^='data:image']")
        assert len(images) >= 1, (
            f"Expected at least one rendered plot, found {len(images)} images"
        )

        browser.close()
