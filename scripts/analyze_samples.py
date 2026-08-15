#!/usr/bin/env python3
"""Analyze fetched sample transcripts.

Usage:
    ANTHROPIC_API_KEY=your-key python scripts/analyze_samples.py --input data/samples.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from sourcerer.analyze import analyze_article, compute_metrics, print_metrics
from sourcerer.extract import get_client


def main():
    parser = argparse.ArgumentParser(description="Analyze sample transcripts")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/samples.json"),
        help="Input JSON file with samples",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results.json"),
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
        default=3000,
        help="Truncate articles to this many words (to reduce API cost)",
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

    print(f"Loaded {len(samples)} samples")

    print("\nInitializing Claude client...")
    try:
        client = get_client()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    articles = []

    for i, sample in enumerate(tqdm(samples, desc="Analyzing")):
        text = sample["text"]
        words = text.split()
        if len(words) > args.max_words:
            text = " ".join(words[: args.max_words])
            print(f"\n  Truncated article {i + 1} from {len(words)} to {args.max_words} words")

        try:
            article = analyze_article(
                article_id=str(i),
                network=sample.get("network", "CNN"),
                program=sample.get("program", "Unknown"),
                date=sample.get("date", ""),
                text=text,
                client=client,
            )
            articles.append(article)
            print(f"\n  Article {i + 1}: {len(article.claims)} claims extracted")
        except Exception as e:
            print(f"\n  Error analyzing article {i + 1}: {e}")
            continue

    print(f"\n\nAnalyzed {len(articles)} articles successfully")

    metrics = compute_metrics(articles)
    print_metrics(metrics)

    output_data = {
        "metrics": metrics,
        "articles": [a.to_dict() for a in articles],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {args.output}")

    print("\n\nSAMPLE CLAIMS WITH EVIDENCE:")
    print("=" * 70)
    for article in articles[:2]:
        print(f"\n[{article.program}]")
        for j, claim in enumerate(article.claims[:5], 1):
            print(f"\n  {j}. {claim.text[:100]}...")
            print(f"     Claimer: {claim.claimer} ({claim.claimer_role.value})")
            print(f"     Type: {claim.claim_type.value}")
            for ev in claim.evidence:
                print(f"     Evidence: {ev.evidence_type.value}/{ev.evidence_subtype.value}")
                if ev.source_entity:
                    print(f"     Source: {ev.source_entity}")


if __name__ == "__main__":
    main()
