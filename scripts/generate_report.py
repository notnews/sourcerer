#!/usr/bin/env python3
"""Generate HTML and Markdown reports from analysis results.

Usage:
    python scripts/generate_report.py --input data/results.json --format html
    python scripts/generate_report.py --input data/batch_results.json --format markdown
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sourcerer.analyze import compute_metrics
from sourcerer.report import (
    export_claims_csv,
    generate_html_report,
    generate_markdown_report,
)
from sourcerer.schema import (
    AnalyzedArticle,
    Claim,
    ClaimCategory,
    ClaimerRole,
    ClaimType,
    ContentType,
    Evidence,
    EvidenceSubtype,
    EvidenceType,
)


def load_results(path: Path) -> tuple[list[AnalyzedArticle], dict]:
    """Load results from JSON file."""
    with open(path) as f:
        data = json.load(f)

    metrics = data.get("metrics", {})

    articles = []
    for a_data in data.get("articles", []):
        claims = []
        for c_data in a_data.get("claims", []):
            evidence = []
            for e_data in c_data.get("evidence", []):
                try:
                    ev_type = EvidenceType(e_data.get("evidence_type", "unsupported"))
                except ValueError:
                    ev_type = EvidenceType.UNSUPPORTED
                try:
                    ev_subtype = EvidenceSubtype(e_data.get("evidence_subtype", "no_attribution"))
                except ValueError:
                    ev_subtype = EvidenceSubtype.NO_ATTRIBUTION

                evidence.append(
                    Evidence(
                        quote=e_data.get("evidence_quote", e_data.get("text", "")),
                        evidence_type=ev_type,
                        evidence_subtype=ev_subtype,
                        source_entity=e_data.get("source_entity"),
                        source_person=e_data.get("source_person"),
                        source_role=e_data.get("source_role"),
                        is_named=e_data.get("is_named", False),
                        is_primary=e_data.get("is_primary", False),
                    )
                )

            try:
                claim_type = ClaimType(c_data.get("claim_type", "factual"))
            except ValueError:
                claim_type = ClaimType.FACTUAL
            try:
                claimer_role = ClaimerRole(c_data.get("claimer_role", "anchor"))
            except ValueError:
                claimer_role = ClaimerRole.ANCHOR

            category = None
            if c_data.get("category"):
                try:
                    category = ClaimCategory(c_data["category"])
                except ValueError:
                    pass

            claims.append(
                Claim(
                    text=c_data.get("text", ""),
                    claimer=c_data.get("claimer", "unknown"),
                    claimer_role=claimer_role,
                    claim_type=claim_type,
                    category=category,
                    claim_id=c_data.get("claim_id"),
                    evidence=evidence,
                )
            )

        content_type = None
        if a_data.get("content_type"):
            try:
                content_type = ContentType(a_data["content_type"])
            except ValueError:
                pass

        articles.append(
            AnalyzedArticle(
                article_id=a_data.get("article_id", ""),
                network=a_data.get("network", "unknown"),
                program=a_data.get("program", "unknown"),
                date=a_data.get("date", ""),
                text="",
                word_count=a_data.get("word_count", 0),
                url=a_data.get("url", ""),
                content_type=content_type,
                claims=claims,
            )
        )

    return articles, metrics


def main():
    parser = argparse.ArgumentParser(description="Generate reports from analysis results")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/batch_results.json"),
        help="Input JSON file with results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: data/report.html or data/report.md)",
    )
    parser.add_argument(
        "--format",
        choices=["html", "markdown", "csv", "all"],
        default="html",
        help="Output format (csv exports claim-level data)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    print(f"Loading results from {args.input}...")
    articles, _ = load_results(args.input)
    metrics = compute_metrics(articles)
    print(f"Loaded {len(articles)} articles with {sum(len(a.claims) for a in articles)} claims")

    if args.format in ("html", "all"):
        output = args.output if args.format == "html" else Path("data/report.html")
        if not args.output:
            output = Path("data/report.html")
        generate_html_report(articles, metrics, output)
        print(f"HTML report saved to {output}")

    if args.format in ("markdown", "all"):
        output = args.output if args.format == "markdown" else Path("data/report.md")
        if not args.output or args.format == "all":
            output = Path("data/report.md")
        md = generate_markdown_report(articles, metrics, output)
        print(f"Markdown report saved to {output}")

        if args.format == "markdown":
            print("\n" + "=" * 70)
            print(md)

    if args.format in ("csv", "all"):
        output = args.output if args.format == "csv" else Path("data/claims.csv")
        if not args.output or args.format == "all":
            output = Path("data/claims.csv")
        df = export_claims_csv(articles, output)
        print(f"CSV exported to {output} ({len(df)} rows)")

        if args.format == "csv":
            print("\nFirst 10 rows:")
            print(df.head(10).to_string())


if __name__ == "__main__":
    main()
