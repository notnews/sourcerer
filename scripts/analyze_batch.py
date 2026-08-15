#!/usr/bin/env python3
"""Analyze transcripts using Anthropic Batch API (50% cost savings).

Usage:
    ANTHROPIC_API_KEY=your-key python scripts/analyze_batch.py --input data/samples.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sourcerer.analyze import compute_metrics, print_metrics
from sourcerer.batch import analyze_batch
from sourcerer.extract import get_client


def main():
    parser = argparse.ArgumentParser(description="Analyze transcripts using batch API")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/samples.json"),
        help="Input JSON file with samples",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/batch_results.json"),
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Maximum articles to analyze",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=0,
        help=(
            "Truncate articles to this many words (0 = no truncation). The old "
            "default of 3000 silently discarded ~47%% of the mean CNN "
            "transcript, so claims in the back half of an hour of television "
            "were never seen and nothing recorded that they were dropped."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=10,
        help="Seconds between batch status checks",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        print("Run: python scripts/fetch_samples.py first")
        sys.exit(1)

    with open(args.input) as f:
        samples = json.load(f)

    if args.max_articles:
        samples = samples[: args.max_articles]

    articles = []
    for i, sample in enumerate(samples):
        text = sample["text"]
        words = text.split()
        # 0 disables truncation. Record what was actually seen: silently
        # dropping half a transcript makes every downstream rate a rate over an
        # unknown denominator.
        truncated = bool(args.max_words) and len(words) > args.max_words
        if truncated:
            text = " ".join(words[: args.max_words])

        articles.append(
            {
                "id": str(i),
                "network": sample.get("network", "CNN"),
                "program": sample.get("program", "Unknown"),
                "date": sample.get("date", ""),
                "url": sample.get("url", ""),
                "text": text,
                "word_count": len(words),
                "words_seen": len(text.split()),
                "truncated": truncated,
            }
        )

    n_truncated = sum(a["truncated"] for a in articles)
    seen = sum(a["words_seen"] for a in articles)
    total = sum(a["word_count"] for a in articles)
    print(f"Loaded {len(articles)} articles for batch processing")
    print(f"  {n_truncated} truncated; {seen:,}/{total:,} words seen ({seen / total:.1%})")
    print("(Batch API provides 50% cost savings vs standard API)")

    print("\nInitializing Claude client...")
    try:
        client = get_client()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    analyzed = analyze_batch(
        articles,
        client=client,
        poll_interval=args.poll_interval,
    )

    print(f"\nAnalyzed {len(analyzed)} articles")

    metrics = compute_metrics(analyzed)
    print_metrics(metrics)

    output_data = {
        "metrics": metrics,
        "articles": [a.to_dict() for a in analyzed],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
