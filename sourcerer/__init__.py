"""Sourcerer: Analyze epistemological support for claims in news articles."""

from sourcerer.analyze import analyze_article, compute_metrics
from sourcerer.batch import analyze_batch, analyze_batch_tooluse
from sourcerer.evidence import classify_evidence
from sourcerer.extract import extract_claims
from sourcerer.pipeline import run_pipeline
from sourcerer.schema import AnalyzedArticle, Claim, Evidence

__version__ = "0.1.0"
__all__ = [
    "Claim",
    "Evidence",
    "AnalyzedArticle",
    "extract_claims",
    "classify_evidence",
    "analyze_article",
    "compute_metrics",
    "analyze_batch",
    "analyze_batch_tooluse",
    "run_pipeline",
]
