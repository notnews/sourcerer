#!/usr/bin/env python3
"""Fetch sample transcripts directly from CNN for testing.

Usage:
    python scripts/fetch_samples.py --output data/samples.json
"""

import argparse
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Install dependencies: pip install requests beautifulsoup4")
    sys.exit(1)


BASE_URL = "https://transcripts.cnn.com"


def fetch_cnn_transcript(url: str) -> dict | None:
    """Fetch and parse a single CNN transcript."""
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        head = soup.find("p", {"class": "cnnTransStoryHead"})
        subhead = soup.find("p", {"class": "cnnTransSubHead"})

        text_parts = []
        air_date = None

        for p in soup.find_all("p", {"class": "cnnBodyText"}):
            text = p.get_text(strip=True)
            m = re.match(r"Aired (.*\d{4})", text)
            if m:
                air_date = m.group(1)
            elif not text.startswith("THIS IS A RUSH TRANSCRIPT"):
                text_parts.append(text)

        content = " ".join(text_parts)
        if len(content) < 500:
            return None

        return {
            "url": url,
            "network": "CNN",
            "program": head.get_text(strip=True) if head else "Unknown",
            "date": air_date or "",
            "subhead": subhead.get_text(strip=True) if subhead else "",
            "text": content,
            "word_count": len(content.split()),
        }
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def fetch_cnn_day_urls(year: int, month: int, day: int) -> list[str]:
    """Get transcript URLs for a specific day."""
    url = f"{BASE_URL}/TRANSCRIPTS/{year:04d}.{month:02d}.{day:02d}.html"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        urls = []

        for link in soup.find_all("a", href=True):
            href = str(link["href"])
            if href.startswith("/show/") and "/segment/" in href:
                full_url = BASE_URL + href
                urls.append(full_url)

        return urls
    except Exception as e:
        print(f"Error fetching day index: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Fetch sample CNN transcripts")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/samples.json"),
        help="Output JSON file",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of transcripts to fetch",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-15",
        help="Start date (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    year, month, day = map(int, args.start_date.split("-"))
    current = date(year, month, day)

    samples = []
    attempts = 0
    max_attempts = 30

    print(f"Fetching {args.n_samples} CNN transcripts starting from {args.start_date}...")

    while len(samples) < args.n_samples and attempts < max_attempts:
        print(f"  Checking {current}...")
        urls = fetch_cnn_day_urls(current.year, current.month, current.day)
        print(f"    Found {len(urls)} transcript links")

        for url in urls[:5]:
            if len(samples) >= args.n_samples:
                break
            transcript = fetch_cnn_transcript(url)
            if transcript:
                samples.append(transcript)
                print(f"    Got: {transcript['program'][:40]}... ({transcript['word_count']} words)")

        current -= timedelta(days=1)
        attempts += 1

    if not samples:
        print("No transcripts found.")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"\nSaved {len(samples)} transcripts to {args.output}")


if __name__ == "__main__":
    main()
