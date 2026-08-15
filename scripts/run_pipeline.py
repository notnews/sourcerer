#!/usr/bin/env python3
"""Run the sourcerer pipeline with staged processing.

Usage:
    # Run full pipeline
    python scripts/run_pipeline.py --stage all --n-articles 50 --seed 42

    # Run specific stages
    python scripts/run_pipeline.py --stage sample --n-articles 100
    python scripts/run_pipeline.py --stage extract
    python scripts/run_pipeline.py --stage filter --category substantive_news
    python scripts/run_pipeline.py --stage evidence

    # Generate report from final output
    python scripts/generate_report.py --input data/batch_results.json --format all
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sourcerer.extract import get_client
from sourcerer.pipeline import (
    classify_evidence_batch,
    convert_to_analyzed_format,
    extract_claims_batch,
    filter_claims,
    run_pipeline,
    sample_articles,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run the sourcerer pipeline with staged processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  sample   - Sample articles from input (creates articles_sampled.json)
  extract  - Extract claims using batch API (creates claims_extracted.json)
  filter   - Filter claims by category (creates claims_filtered.json)
  evidence - Classify evidence for claims (creates evidence_classified.json)
  all      - Run all stages in sequence

Output files:
  data/articles_sampled.json   - Sampled articles with IDs
  data/claims_extracted.json   - Extracted claims per article
  data/claims_filtered.json    - Filtered claims (substantive_news)
  data/evidence_classified.json - Claims with evidence classification
  data/batch_results.json      - Final output (compatible with generate_report.py)
""",
    )

    parser.add_argument(
        "--stage",
        type=str,
        choices=["sample", "extract", "filter", "evidence", "all"],
        default="all",
        help="Stage(s) to run",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/samples.json"),
        help="Input JSON file with articles (for sample stage)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for intermediate files",
    )
    parser.add_argument(
        "--n-articles",
        type=int,
        default=100,
        help="Number of articles to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="substantive_news",
        help="Category to filter to (or 'all' for no filter)",
    )
    parser.add_argument(
        "--max-claims-per-article",
        type=int,
        default=10,
        help="Maximum claims to keep per article after filtering",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Claude model to use",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=10,
        help="Seconds between batch status checks",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=500,
        help="Minimum word count for article inclusion",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "articles_sampled": args.output_dir / "articles_sampled.json",
        "claims_extracted": args.output_dir / "claims_extracted.json",
        "claims_filtered": args.output_dir / "claims_filtered.json",
        "evidence_classified": args.output_dir / "evidence_classified.json",
        "batch_results": args.output_dir / "batch_results.json",
    }

    if args.stage == "all":
        if not args.input.exists():
            print(f"Input file not found: {args.input}")
            print("Run: python scripts/fetch_samples.py first")
            sys.exit(1)

        print("=" * 60)
        print("SOURCERER PIPELINE")
        print("=" * 60)
        print(f"Input: {args.input}")
        print(f"Output dir: {args.output_dir}")
        print(f"Articles: {args.n_articles}, Seed: {args.seed}")
        print(f"Filter: {args.category}, Max per article: {args.max_claims_per_article}")
        print("=" * 60)

        try:
            client = get_client()
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

        run_pipeline(
            input_path=args.input,
            output_dir=args.output_dir,
            n_articles=args.n_articles,
            seed=args.seed,
            category=args.category,
            max_claims_per_article=args.max_claims_per_article,
            model=args.model,
            poll_interval=args.poll_interval,
            client=client,
        )

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print("\nOutput files:")
        for name, path in paths.items():
            if path.exists():
                print(f"  {name}: {path}")
        print("\nTo generate reports:")
        print(f"  python scripts/generate_report.py --input {paths['batch_results']} --format all")

    elif args.stage == "sample":
        if not args.input.exists():
            print(f"Input file not found: {args.input}")
            sys.exit(1)

        sample_articles(
            args.input,
            paths["articles_sampled"],
            args.n_articles,
            args.seed,
            args.min_words,
        )

    elif args.stage == "extract":
        if not paths["articles_sampled"].exists():
            print(f"Required input not found: {paths['articles_sampled']}")
            print("Run: python scripts/run_pipeline.py --stage sample first")
            sys.exit(1)

        try:
            client = get_client()
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

        extract_claims_batch(
            paths["articles_sampled"],
            paths["claims_extracted"],
            client,
            args.model,
            args.poll_interval,
        )

    elif args.stage == "filter":
        if not paths["claims_extracted"].exists():
            print(f"Required input not found: {paths['claims_extracted']}")
            print("Run: python scripts/run_pipeline.py --stage extract first")
            sys.exit(1)

        filter_claims(
            paths["claims_extracted"],
            paths["claims_filtered"],
            args.category,
            args.max_claims_per_article,
            args.seed,
        )

    elif args.stage == "evidence":
        if not paths["claims_filtered"].exists():
            print(f"Required input not found: {paths['claims_filtered']}")
            print("Run: python scripts/run_pipeline.py --stage filter first")
            sys.exit(1)

        try:
            client = get_client()
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

        classify_evidence_batch(
            paths["claims_filtered"],
            paths["evidence_classified"],
            client,
            args.model,
            args.poll_interval,
        )

        convert_to_analyzed_format(
            paths["evidence_classified"],
            paths["batch_results"],
        )

        print(f"\nReady for reporting: {paths['batch_results']}")


if __name__ == "__main__":
    main()
