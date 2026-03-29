"""Monkey-patch marimo to register our custom loader and store."""

from __future__ import annotations

_installed = False


def install() -> None:
    """Register NumpyJsonLoader in marimo's PERSISTENT_LOADERS.

    Safe to call multiple times — only patches once.
    """
    global _installed
    if _installed:
        return
    _installed = True

    from marimo._save.loaders import PERSISTENT_LOADERS
    from marimo_precompute.numpy_json import NumpyJsonLoader

    PERSISTENT_LOADERS["numpy_json"] = NumpyJsonLoader  # type: ignore[assignment]
