#!/usr/bin/env python3
"""Run epistemological analysis on sample news transcripts.

This script performs the empirical exercise:
1. Load sample transcripts from each network
2. Extract claims and classify evidence
3. Compute and display aggregate metrics

Usage:
    python scripts/sample_analysis.py --data-dir /path/to/data

The data directory should contain CSV files from Harvard Dataverse:
    - cnn-8.csv (or any cnn-*.csv)
    - fox_transcripts.csv (or similar)
    - msnbc_transcripts_2022.csv.gz (or similar)
    - nbc_transcripts.csv (or similar)
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from sourcerer.analyze import analyze_article, compute_metrics, print_metrics
from sourcerer.config import MODEL
from sourcerer.data import iter_transcripts, load_csv
from sourcerer.extract import get_client


def find_data_files(data_dir: Path) -> dict[str, Path]:
    """Find transcript CSV files in data directory."""
    files = {}

    for pattern, network in [
        ("cnn*.csv", "CNN"),
        ("fox*.csv", "Fox"),
        ("msnbc*.csv*", "MSNBC"),
        ("nbc*.csv", "NBC"),
    ]:
        matches = list(data_dir.glob(pattern))
        if matches:
            files[network] = sorted(matches)[-1]

    return files


def main():
    parser = argparse.ArgumentParser(description="Analyze news transcripts for epistemology")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing transcript CSV files",
    )
    parser.add_argument(
        "--n-per-network",
        type=int,
        default=10,
        help="Number of transcripts to analyze per network (default: 10)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=500,
        help="Minimum word count for transcripts (default: 500)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help="Claude model to use",
    )
    args = parser.parse_args()

    print("Finding data files...")
    data_files = find_data_files(args.data_dir)

    if not data_files:
        print(f"No transcript files found in {args.data_dir}")
        print("Expected files matching: cnn*.csv, fox*.csv, msnbc*.csv*, nbc*.csv")
        sys.exit(1)

    print(f"Found files for: {', '.join(data_files.keys())}")
    for network, path in data_files.items():
        print(f"  {network}: {path.name}")

    print("\nInitializing Claude client...")
    try:
        client = get_client()
    except ValueError as e:
        print(f"Error: {e}")
        print("Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    all_articles = []

    for network, path in data_files.items():
        print(f"\nProcessing {network}...")

        df = load_csv(path, nrows=1000)
        print(f"  Loaded {len(df)} rows")

        if "text" in df.columns:
            df["_word_count"] = df["text"].fillna("").apply(lambda x: len(str(x).split()))
            df = df[df["_word_count"] >= args.min_words]

        if len(df) > args.n_per_network:
            df = df.sample(n=args.n_per_network, random_state=42)

        print(f"  Analyzing {len(df)} transcripts...")

        for transcript in tqdm(iter_transcripts(df), total=len(df), desc=f"  {network}"):
            transcript["network"] = network

            try:
                article = analyze_article(
                    article_id=transcript["article_id"],
                    network=network,
                    program=transcript["program"],
                    date=transcript["date"],
                    text=transcript["text"],
                    client=client,
                    model=args.model,
                )
                all_articles.append(article)
            except Exception as e:
                print(f"\n  Warning: Failed to analyze {transcript['article_id']}: {e}")
                continue

    print(f"\nAnalyzed {len(all_articles)} articles total")

    metrics = compute_metrics(all_articles)

    print_metrics(metrics)

    if args.output:
        output_data = {
            "metrics": metrics,
            "articles": [a.to_dict() for a in all_articles],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
