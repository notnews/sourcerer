#!/usr/bin/env python3
"""Demo script to analyze a single piece of text.

Usage:
    python scripts/demo.py "Your news text here"
    python scripts/demo.py --file transcript.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sourcerer.analyze import analyze_article
from sourcerer.extract import get_client

SAMPLE_TEXT = """
ANDERSON COOPER, CNN ANCHOR: Good evening. Tonight, the White House announced
new sanctions on Russia. According to a senior administration official, these
sanctions will target key sectors of the Russian economy.

Dr. James Wilson, a foreign policy expert at Georgetown University, told CNN
that the sanctions could have significant impact. "Based on our analysis of
similar measures in the past, we expect these to reduce Russian oil exports
by up to 15 percent," Dr. Wilson said.

Critics say the measures don't go far enough. Sources familiar with the
discussions say European allies wanted stronger action. The State Department
declined to comment on those internal deliberations.

Meanwhile, video footage obtained by CNN shows Russian military movements near
the border. Pentagon officials confirmed the authenticity of the video but would
not speculate on Russia's intentions.
"""


def main():
    parser = argparse.ArgumentParser(description="Analyze text for claims and evidence")
    parser.add_argument("text", nargs="?", help="Text to analyze")
    parser.add_argument("--file", type=Path, help="File containing text to analyze")
    parser.add_argument("--demo", action="store_true", help="Run with sample text")
    args = parser.parse_args()

    if args.demo:
        text = SAMPLE_TEXT
    elif args.file:
        text = args.file.read_text()
    elif args.text:
        text = args.text
    else:
        print("Provide text, --file, or --demo")
        sys.exit(1)

    print("Initializing Claude client...")
    try:
        client = get_client()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\nAnalyzing text ({len(text.split())} words)...")
    print("-" * 60)

    article = analyze_article(
        article_id="demo",
        network="Demo",
        program="Demo",
        date="2024-01-01",
        text=text,
        client=client,
    )

    print(f"\nExtracted {len(article.claims)} claims:\n")

    for i, claim in enumerate(article.claims, 1):
        print(f"{i}. [{claim.claim_type.value.upper()}] {claim.text}")
        print(f"   Claimer: {claim.claimer} ({claim.claimer_role.value})")
        for evidence in claim.evidence:
            print(f"   Evidence: {evidence.evidence_type.value}/{evidence.evidence_subtype.value}")
            if evidence.source_entity:
                print(f"   Source: {evidence.source_entity}")
            print(f"   Named: {evidence.is_named}, Primary: {evidence.is_primary}")
        print()

    print("-" * 60)
    print("\nEvidence Distribution:")
    for etype, count in article.evidence_distribution().items():
        print(f"  {etype}: {count}")

    print(f"\nClaims per 100 words: {article.claims_per_100_words:.2f}")


if __name__ == "__main__":
    main()
