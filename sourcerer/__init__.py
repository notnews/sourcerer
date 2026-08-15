"""Sourcerer: Analyze epistemological support for claims in news articles.

API-calling submodules are re-exported through ``__getattr__`` rather than at
import time, so importing this package does not pull in the Anthropic SDK.
``sourcerer.schema`` is pure dataclasses and enums and stays usable on its own.

This is not cosmetic: eager re-exports made ``import sourcerer.schema`` fail
with ``ModuleNotFoundError: anthropic`` in any environment without the SDK,
which meant the test suite could not be collected at all there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sourcerer.config import MODEL, VALIDATION_MODEL
from sourcerer.schema import AnalyzedArticle, Claim, Evidence

if TYPE_CHECKING:  # import-time types for checkers, not at runtime
    from sourcerer.analyze import analyze_article, compute_metrics
    from sourcerer.batch import analyze_batch, analyze_batch_tooluse
    from sourcerer.evidence import classify_evidence
    from sourcerer.extract import extract_claims
    from sourcerer.pipeline import run_pipeline

__version__ = "0.1.0"

# name -> submodule providing it, resolved on first attribute access
_LAZY: dict[str, str] = {
    "analyze_article": "analyze",
    "compute_metrics": "analyze",
    "analyze_batch": "batch",
    "analyze_batch_tooluse": "batch",
    "classify_evidence": "evidence",
    "extract_claims": "extract",
    "run_pipeline": "pipeline",
}

# Spelled out rather than built from _LAZY: a computed __all__ defeats both
# ruff's unused-import check and pyright's export analysis, so the names would
# stop being verified as actually resolvable.
__all__ = [
    "MODEL",
    "VALIDATION_MODEL",
    "AnalyzedArticle",
    "Claim",
    "Evidence",
    "analyze_batch",
    "analyze_batch_tooluse",
    "analyze_article",
    "classify_evidence",
    "compute_metrics",
    "extract_claims",
    "run_pipeline",
]


def __getattr__(name: str) -> Any:
    """Resolve API-calling exports on first use (PEP 562)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f"sourcerer.{module}"), name)


def __dir__() -> list[str]:
    return sorted(__all__)
