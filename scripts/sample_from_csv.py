#!/usr/bin/env python3
"""Sample random articles from CNN CSV data.

Usage:
    python scripts/sample_from_csv.py --input data/cnn-8.csv.gz --output data/samples_100.json --n 100
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Sample random articles from CNN CSV")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/cnn-8.csv.gz"),
        help="Input CSV file (gzipped)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/samples_100.json"),
        help="Output JSON file",
    )
    parser.add_argument(
        "--n",
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
    args = parser.parse_args()

    print(f"Reading {args.input}...")
    df = pd.read_csv(args.input, compression="gzip")
    print(f"Loaded {len(df)} articles")

    df = df[df["text"].notna() & (df["text"].str.len() > 500)]
    print(f"After filtering: {len(df)} articles with text > 500 chars")

    random.seed(args.seed)
    sample_indices = random.sample(range(len(df)), min(args.n, len(df)))
    sampled = df.iloc[sample_indices]

    samples = []
    for _, row in sampled.iterrows():
        date_str = f"{int(row['year'])}-{int(row['month']):02d}-{int(row['date']):02d}" if pd.notna(row.get("year")) else ""
        samples.append({
            "url": row["url"],
            "network": row.get("channel.name", "CNN"),
            "program": row.get("program.name", "Unknown"),
            "date": date_str,
            "text": row["text"],
            "word_count": row.get("wordcount", len(str(row["text"]).split())),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Saved {len(samples)} random samples to {args.output}")


if __name__ == "__main__":
    main()
