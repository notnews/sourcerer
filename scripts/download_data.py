#!/usr/bin/env python3
"""Download transcript data from Harvard Dataverse.

Data sources:
- CNN: https://doi.org/10.7910/DVN/ISDPJU (restricted)
- Fox News: https://doi.org/10.7910/DVN/Q2KIES
- MSNBC: https://doi.org/10.7910/DVN/UPJDE1
- NBC: https://doi.org/10.7910/DVN/ND1TCV

Usage:
    python scripts/download_data.py --output-dir data/

Note: Some datasets require Dataverse authentication for restricted access.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)


DATAVERSE_API = "https://dataverse.harvard.edu/api"

DATASETS = {
    "cnn": {
        "doi": "10.7910/DVN/ISDPJU",
        "description": "CNN Transcripts 2000-2025 (336k transcripts, restricted)",
        "files": ["cnn-8.csv"],  # Most recent
    },
    "fox": {
        "doi": "10.7910/DVN/Q2KIES",
        "description": "Fox News Transcripts 2003-2025 (24k transcripts)",
        "files": ["fox_transcripts.csv"],
    },
    "msnbc": {
        "doi": "10.7910/DVN/UPJDE1",
        "description": "MSNBC Transcripts 2010-2025 (10k transcripts)",
        "files": ["msnbc_transcripts_2022.csv.gz"],
    },
    "nbc": {
        "doi": "10.7910/DVN/ND1TCV",
        "description": "NBC Transcripts 2008-2014 (5k transcripts)",
        "files": ["nbc_transcripts.csv"],
    },
}


def get_dataset_files(doi: str, api_token: str = None) -> list:
    """Get list of files in a Dataverse dataset."""
    headers = {}
    if api_token:
        headers["X-Dataverse-key"] = api_token

    url = f"{DATAVERSE_API}/datasets/:persistentId/?persistentId=doi:{doi}"
    resp = requests.get(url, headers=headers)

    if resp.status_code != 200:
        print(f"Error fetching dataset info: {resp.status_code}")
        return []

    data = resp.json()
    files = []
    for f in data.get("data", {}).get("latestVersion", {}).get("files", []):
        files.append(
            {
                "id": f["dataFile"]["id"],
                "name": f["dataFile"]["filename"],
                "size": f["dataFile"].get("filesize", 0),
            }
        )
    return files


def download_file(file_id: int, output_path: Path, api_token: str = None) -> bool:
    """Download a file from Dataverse."""
    headers = {}
    if api_token:
        headers["X-Dataverse-key"] = api_token

    url = f"{DATAVERSE_API}/access/datafile/{file_id}"
    resp = requests.get(url, headers=headers, stream=True)

    if resp.status_code != 200:
        print(f"Error downloading file: {resp.status_code}")
        if resp.status_code == 403:
            print("Access restricted. Provide API token with --api-token")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    return True


def main():
    parser = argparse.ArgumentParser(description="Download transcript data from Dataverse")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for downloaded files",
    )
    parser.add_argument(
        "--source",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to download",
    )
    parser.add_argument(
        "--api-token",
        help="Dataverse API token (for restricted datasets)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List available files without downloading",
    )
    args = parser.parse_args()

    api_token = args.api_token or os.environ.get("DATAVERSE_API_TOKEN")

    sources = list(DATASETS.keys()) if args.source == "all" else [args.source]

    for source in sources:
        info = DATASETS[source]
        print(f"\n{'=' * 60}")
        print(f"{source.upper()}: {info['description']}")
        print(f"DOI: {info['doi']}")
        print("=" * 60)

        files = get_dataset_files(info["doi"], api_token)

        if not files:
            print("Could not fetch file list")
            continue

        print(f"\nAvailable files ({len(files)}):")
        for f in files:
            size_mb = f["size"] / (1024 * 1024)
            print(f"  - {f['name']} ({size_mb:.1f} MB)")

        if args.list_only:
            continue

        for f in files:
            if info["files"] and f["name"] not in info["files"]:
                continue

            output_path = args.output_dir / source / f["name"]
            if output_path.exists():
                print(f"\nSkipping {f['name']} (already exists)")
                continue

            print(f"\nDownloading {f['name']}...")
            if download_file(f["id"], output_path, api_token):
                print(f"Saved to {output_path}")
            else:
                print(f"Failed to download {f['name']}")


if __name__ == "__main__":
    main()
