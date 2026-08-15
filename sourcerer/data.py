"""Data loading utilities for news transcripts."""

import gzip
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd


def load_csv(
    path: str | Path,
    text_column: str = "text",
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load a transcript CSV file.

    Args:
        path: Path to CSV file (supports .csv and .csv.gz)
        text_column: Name of column containing transcript text
        nrows: Limit number of rows to load

    Returns:
        DataFrame with transcript data
    """
    path = Path(path)

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            df = pd.read_csv(f, nrows=nrows)
    else:
        df = pd.read_csv(path, nrows=nrows)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available: {list(df.columns)}")

    return df


def iter_transcripts(
    df: pd.DataFrame,
    text_column: str = "text",
    id_column: Optional[str] = None,
    network_column: str = "channel.name",
    program_column: str = "program.name",
    date_columns: tuple[str, str, str] = ("year", "month", "date"),
) -> Iterator[dict]:
    """Iterate over transcripts in a DataFrame.

    Args:
        df: DataFrame with transcript data
        text_column: Column containing transcript text
        id_column: Column for unique ID (uses index if None)
        network_column: Column for network name
        program_column: Column for program name
        date_columns: Columns for year, month, day

    Yields:
        Dict with article_id, network, program, date, text, word_count
    """
    year_col, month_col, day_col = date_columns

    for idx, row in df.iterrows():
        text = str(row.get(text_column, ""))
        if not text or text == "nan":
            continue

        article_id = str(row.get(id_column, idx)) if id_column else str(idx)

        network = str(row.get(network_column, "unknown"))
        program = str(row.get(program_column, "unknown"))

        year = row.get(year_col, "")
        month = row.get(month_col, "")
        day = row.get(day_col, "")
        date = f"{year}-{month:02d}-{day:02d}" if year and month and day else ""

        word_count = len(text.split())

        yield {
            "article_id": article_id,
            "network": network,
            "program": program,
            "date": date,
            "text": text,
            "word_count": word_count,
        }


def sample_transcripts(
    paths: dict[str, str | Path],
    n_per_source: int = 10,
    seed: int = 42,
    min_words: int = 500,
) -> pd.DataFrame:
    """Sample transcripts from multiple sources.

    Args:
        paths: Dict mapping network name to CSV path
        n_per_source: Number of transcripts to sample per source
        seed: Random seed for reproducibility
        min_words: Minimum word count for inclusion

    Returns:
        DataFrame with sampled transcripts
    """
    samples = []

    for network, path in paths.items():
        df = load_csv(path)

        if "text" in df.columns:
            df["word_count"] = df["text"].fillna("").apply(lambda x: len(str(x).split()))
            df = df[df["word_count"] >= min_words]

        if len(df) > n_per_source:
            df = df.sample(n=n_per_source, random_state=seed)

        df = df.copy()
        df["network_source"] = network
        samples.append(df)

    return pd.concat(samples, ignore_index=True)
