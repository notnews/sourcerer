"""Pipeline orchestration with staged processing and intermediate outputs.

Stages:
0. sample_articles: Reproducible sampling from input JSON
1. extract_claims: Batch claim extraction with tool_use
2. filter_claims: Filter by category and sample within articles
3. classify_evidence: Batched evidence classification
"""

import json
import random
import time
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from sourcerer.config import (
    EVIDENCE_MAX_CHARS,
    EXTRACTION_MAX_CHARS,
    MAX_TOKENS,
    MODEL,
    THINKING,
)
from sourcerer.evidence import SUBTYPE_TO_TYPE
from sourcerer.prompts import (
    BATCHED_EVIDENCE_CLASSIFICATION_USER,
    CLAIM_EXTRACTION_SYSTEM,
    CLAIM_EXTRACTION_USER,
    EVIDENCE_CLASSIFICATION_SYSTEM,
)
from sourcerer.schema import (
    ClaimCategory,
    ClaimerRole,
    ClaimType,
    EvidenceSubtype,
    SourceType,
)
from sourcerer.tools import (
    BATCHED_EVIDENCE_CLASSIFICATION_TOOL,
    CLAIM_EXTRACTION_TOOL,
)


def generate_article_id(network: str, date: str, index: int) -> str:
    """Generate consistent article ID.

    Format: {network}_{date}_{index:03d}
    Example: cnn_2024-01-15_001
    """
    network_clean = network.lower().replace(" ", "_")
    date_clean = date.replace(" ", "_").replace(",", "").replace("/", "-")
    return f"{network_clean}_{date_clean}_{index:03d}"


def generate_claim_id(article_id: str, claim_index: int) -> str:
    """Generate claim ID from article ID and index.

    Format: {article_id}_{claim_index:03d}
    Example: cnn_2024-01-15_001_003
    """
    return f"{article_id}_{claim_index:03d}"


def sample_articles(
    input_path: Path,
    output_path: Path,
    n_articles: int,
    seed: int,
    min_words: int = 500,
) -> dict:
    """Stage 0: Sample articles reproducibly.

    Args:
        input_path: Path to samples.json with all articles
        output_path: Path to write articles_sampled.json
        n_articles: Number of articles to sample
        seed: Random seed for reproducibility
        min_words: Minimum word count for inclusion

    Returns:
        Dict with metadata and sampled articles
    """
    with open(input_path) as f:
        all_articles = json.load(f)

    eligible = []
    for i, article in enumerate(all_articles):
        text = article.get("text", "")
        word_count = len(text.split())
        if word_count >= min_words:
            eligible.append((i, article, word_count))

    random.seed(seed)
    if len(eligible) > n_articles:
        sampled_indices = random.sample(range(len(eligible)), n_articles)
        sampled = [eligible[i] for i in sorted(sampled_indices)]
    else:
        sampled = eligible

    articles_out = []
    for idx, (orig_idx, article, word_count) in enumerate(sampled):
        network = article.get("network", "unknown")
        date = article.get("date", "")
        article_id = generate_article_id(network, date, idx)

        articles_out.append(
            {
                "article_id": article_id,
                "original_index": orig_idx,
                "network": network,
                "program": article.get("program", "unknown"),
                "date": date,
                "url": article.get("url", ""),
                "subhead": article.get("subhead", ""),
                "text": article.get("text", ""),
                "word_count": word_count,
            }
        )

    output = {
        "metadata": {
            "stage": "sample",
            "seed": seed,
            "n_requested": n_articles,
            "n_sampled": len(articles_out),
            "min_words": min_words,
            "source_file": str(input_path),
        },
        "articles": articles_out,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Sampled {len(articles_out)} articles (seed={seed}) -> {output_path}")
    return output


def extract_claims_batch(
    input_path: Path,
    output_path: Path,
    client: Anthropic,
    model: str = MODEL,
    poll_interval: int = 10,
    max_wait: int = 3600,
) -> dict:
    """Stage 1: Extract claims using batch API with tool_use.

    Args:
        input_path: Path to articles_sampled.json
        output_path: Path to write claims_extracted.json
        client: Anthropic client
        model: Model to use
        poll_interval: Seconds between batch status checks
        max_wait: Maximum seconds to wait for batch

    Returns:
        Dict with extracted claims per article
    """
    with open(input_path) as f:
        data = json.load(f)

    articles = data["articles"]
    print(f"Extracting claims from {len(articles)} articles...")

    requests = []
    for article in articles:
        text = article["text"]
        if len(text) > EXTRACTION_MAX_CHARS:
            text = text[:EXTRACTION_MAX_CHARS]

        requests.append(
            {
                "custom_id": f"claims_{article['article_id']}",
                "params": {
                    "model": model,
                    "max_tokens": MAX_TOKENS,
                    "thinking": THINKING,
                    "system": CLAIM_EXTRACTION_SYSTEM,
                    "messages": [{"role": "user", "content": CLAIM_EXTRACTION_USER.format(text=text)}],
                    "tools": [CLAIM_EXTRACTION_TOOL],
                    "tool_choice": {"type": "tool", "name": "extract_claims"},
                },
            }
        )

    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    print(f"Created batch: {batch_id}")

    results = _wait_for_batch(batch_id, client, poll_interval, max_wait)

    articles_out = []
    total_claims = 0

    for article in articles:
        article_id = article["article_id"]
        result_key = f"claims_{article_id}"

        claims = []
        if result_key in results:
            claims_data = results[result_key]
            for i, claim_data in enumerate(claims_data.get("claims", [])):
                claim_id = generate_claim_id(article_id, i)
                claims.append(
                    {
                        "claim_id": claim_id,
                        "text": claim_data.get("text", ""),
                        "claimer": claim_data.get("claimer", "unknown"),
                        "claimer_role": claim_data.get("claimer_role", "anchor"),
                        "claim_type": claim_data.get("claim_type", "factual"),
                        "category": claim_data.get("category", "substantive_news"),
                    }
                )
                total_claims += 1

        articles_out.append(
            {
                "article_id": article_id,
                "network": article["network"],
                "program": article["program"],
                "date": article["date"],
                "url": article.get("url", ""),
                "word_count": article["word_count"],
                "text": article["text"],
                "claims": claims,
            }
        )

    output = {
        "metadata": {
            "stage": "extract",
            "model": model,
            "batch_id": batch_id,
            "total_articles": len(articles_out),
            "total_claims": total_claims,
            "source_file": str(input_path),
        },
        "articles": articles_out,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Extracted {total_claims} claims from {len(articles_out)} articles -> {output_path}")
    return output


def filter_claims(
    input_path: Path,
    output_path: Path,
    category: str = "substantive_news",
    max_claims_per_article: int = 10,
    seed: int = 42,
) -> dict:
    """Stage 2: Filter claims by category and sample within articles.

    Args:
        input_path: Path to claims_extracted.json
        output_path: Path to write claims_filtered.json
        category: Category to filter to (or "all" for no filter)
        max_claims_per_article: Maximum claims to keep per article
        seed: Random seed for sampling within articles

    Returns:
        Dict with filtered claims per article
    """
    with open(input_path) as f:
        data = json.load(f)

    articles = data["articles"]
    random.seed(seed)

    articles_out = []
    total_before = 0
    total_after = 0

    for article in articles:
        all_claims = article.get("claims", [])
        total_before += len(all_claims)

        if category != "all":
            filtered = [c for c in all_claims if c.get("category") == category]
        else:
            filtered = all_claims

        if len(filtered) > max_claims_per_article:
            sampled = random.sample(filtered, max_claims_per_article)
            sampled.sort(key=lambda c: c["claim_id"])
        else:
            sampled = filtered

        total_after += len(sampled)

        articles_out.append(
            {
                "article_id": article["article_id"],
                "network": article["network"],
                "program": article["program"],
                "date": article["date"],
                "url": article.get("url", ""),
                "word_count": article["word_count"],
                "text": article["text"],
                "claims": sampled,
            }
        )

    output = {
        "metadata": {
            "stage": "filter",
            "filter_category": category,
            "max_per_article": max_claims_per_article,
            "seed": seed,
            "claims_before": total_before,
            "claims_after": total_after,
            "source_file": str(input_path),
        },
        "articles": articles_out,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Filtered {total_before} -> {total_after} claims (category={category}, max={max_claims_per_article})")
    print(f"  -> {output_path}")
    return output


def classify_evidence_batch(
    input_path: Path,
    output_path: Path,
    client: Anthropic,
    model: str = MODEL,
    poll_interval: int = 10,
    max_wait: int = 3600,
    max_claims_per_batch: int = 15,
) -> dict:
    """Stage 3: Classify evidence using batched requests per article.

    Groups claims by article to share context. Each article's claims
    are processed in a single request.

    Args:
        input_path: Path to claims_filtered.json
        output_path: Path to write evidence_classified.json
        client: Anthropic client
        model: Model to use
        poll_interval: Seconds between batch status checks
        max_wait: Maximum seconds to wait for batch
        max_claims_per_batch: Maximum claims per request

    Returns:
        Dict with evidence-classified claims per article
    """
    with open(input_path) as f:
        data = json.load(f)

    articles = data["articles"]

    articles_with_claims = [a for a in articles if a.get("claims")]
    total_claims = sum(len(a["claims"]) for a in articles_with_claims)
    print(f"Classifying evidence for {total_claims} claims across {len(articles_with_claims)} articles...")

    if total_claims == 0:
        output = {
            "metadata": {
                "stage": "evidence",
                "model": model,
                "batch_id": None,
                "total_articles": len(articles),
                "total_claims": 0,
                "source_file": str(input_path),
            },
            "articles": articles,
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        return output

    requests = []
    request_map = {}

    for article in articles_with_claims:
        article_id = article["article_id"]
        claims = article["claims"]

        for batch_idx in range(0, len(claims), max_claims_per_batch):
            batch_claims = claims[batch_idx : batch_idx + max_claims_per_batch]
            request_id = f"evidence_{article_id}_{batch_idx}"

            claims_list_text = _format_claims_for_prompt(batch_claims)

            requests.append(
                {
                    "custom_id": request_id,
                    "params": {
                        "model": model,
                        "max_tokens": MAX_TOKENS,
                        "thinking": THINKING,
                        "system": EVIDENCE_CLASSIFICATION_SYSTEM,
                        "messages": [
                            {
                                "role": "user",
                                "content": BATCHED_EVIDENCE_CLASSIFICATION_USER.format(
                                    article_text=article["text"][:EVIDENCE_MAX_CHARS],
                                    claims_list=claims_list_text,
                                ),
                            }
                        ],
                        "tools": [BATCHED_EVIDENCE_CLASSIFICATION_TOOL],
                        "tool_choice": {"type": "tool", "name": "classify_evidence_batch"},
                    },
                }
            )

            request_map[request_id] = {
                "article_id": article_id,
                "claim_ids": [c["claim_id"] for c in batch_claims],
            }

    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    print(f"Created batch: {batch_id}")

    results = _wait_for_batch(batch_id, client, poll_interval, max_wait)

    evidence_by_claim: dict[str, dict] = {}
    for request_id, result_data in results.items():
        evidence_list = result_data.get("evidence_list", [])
        for ev in evidence_list:
            claim_id = ev.get("claim_id")
            if claim_id:
                evidence_by_claim[claim_id] = ev

    articles_out = []
    for article in articles:
        claims_out = []
        for claim in article.get("claims", []):
            claim_id = claim["claim_id"]
            ev_data = evidence_by_claim.get(claim_id, {})

            subtype_str = ev_data.get("evidence_subtype", "no_attribution")
            evidence_type = SUBTYPE_TO_TYPE.get(subtype_str)
            if evidence_type:
                evidence_type_str = evidence_type.value
            else:
                evidence_type_str = ev_data.get("evidence_type", "unsupported")

            source_type_str = ev_data.get("source_type")
            if source_type_str:
                try:
                    SourceType(source_type_str)
                except ValueError:
                    source_type_str = None

            claims_out.append(
                {
                    "claim_id": claim_id,
                    "text": claim["text"],
                    "claimer": claim["claimer"],
                    "claimer_role": claim["claimer_role"],
                    "claim_type": claim["claim_type"],
                    "category": claim.get("category"),
                    "evidence": {
                        "evidence_quote": ev_data.get("evidence_quote"),
                        "evidence_type": evidence_type_str,
                        "evidence_subtype": subtype_str,
                        "source_entity": ev_data.get("source_entity"),
                        "source_person": ev_data.get("source_person"),
                        "source_role": ev_data.get("source_role"),
                        "source_type": source_type_str,
                        "is_named": ev_data.get("is_named", False),
                        "is_primary": ev_data.get("is_primary", False),
                    },
                }
            )

        articles_out.append(
            {
                "article_id": article["article_id"],
                "network": article["network"],
                "program": article["program"],
                "date": article["date"],
                "url": article.get("url", ""),
                "word_count": article["word_count"],
                "claims": claims_out,
            }
        )

    output = {
        "metadata": {
            "stage": "evidence",
            "model": model,
            "batch_id": batch_id,
            "total_articles": len(articles_out),
            "total_claims": total_claims,
            "source_file": str(input_path),
        },
        "articles": articles_out,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Classified evidence for {total_claims} claims -> {output_path}")
    return output


def _format_claims_for_prompt(claims: list[dict]) -> str:
    """Format claims list for the batched evidence prompt."""
    lines = []
    for claim in claims:
        lines.append(f"- claim_id: {claim['claim_id']}")
        lines.append(f'  text: "{claim["text"]}"')
        lines.append(f"  claimer: {claim['claimer']} ({claim['claimer_role']})")
        lines.append("")
    return "\n".join(lines)


def _wait_for_batch(
    batch_id: str,
    client: Anthropic,
    poll_interval: int,
    max_wait: int,
) -> dict[str, Any]:
    """Wait for batch to complete and parse tool_use results."""
    start = time.time()

    while time.time() - start < max_wait:
        batch = client.messages.batches.retrieve(batch_id)

        if batch.processing_status == "ended":
            break

        succeeded = batch.request_counts.succeeded
        total = batch.request_counts.processing + succeeded
        print(f"  Batch {batch_id[:12]}... status: {batch.processing_status}, completed: {succeeded}/{total}")
        time.sleep(poll_interval)
    else:
        raise TimeoutError(f"Batch {batch_id} did not complete in {max_wait}s")

    results = {}
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            content = result.result.message.content
            for block in content:
                if block.type == "tool_use":
                    results[result.custom_id] = block.input
                    break
            else:
                if content and content[0].type == "text":
                    try:
                        results[result.custom_id] = json.loads(content[0].text)
                    except json.JSONDecodeError:
                        pass

    return results


def convert_to_analyzed_format(input_path: Path, output_path: Path) -> dict:
    """Convert evidence_classified.json to the format expected by report.py.

    This bridges the new pipeline output to the existing reporting infrastructure.

    Args:
        input_path: Path to evidence_classified.json
        output_path: Path to write batch_results.json (compatible format)

    Returns:
        Dict in the format expected by generate_report.py
    """
    with open(input_path) as f:
        data = json.load(f)

    articles_out = []
    for article in data["articles"]:
        claims_out = []
        for claim in article.get("claims", []):
            ev = claim.get("evidence", {})
            try:
                claimer_role = ClaimerRole(claim.get("claimer_role", "anchor")).value
            except ValueError:
                claimer_role = "anchor"
            try:
                claim_type = ClaimType(claim.get("claim_type", "factual")).value
            except ValueError:
                claim_type = "factual"
            category = claim.get("category")
            if category:
                try:
                    category = ClaimCategory(category).value
                except ValueError:
                    category = None

            try:
                ev_subtype = EvidenceSubtype(ev.get("evidence_subtype", "no_attribution")).value
            except ValueError:
                ev_subtype = "no_attribution"

            ev_type = SUBTYPE_TO_TYPE.get(ev_subtype)
            if ev_type:
                ev_type_str = ev_type.value
            else:
                ev_type_str = ev.get("evidence_type", "unsupported")

            claims_out.append(
                {
                    "claim_id": claim.get("claim_id"),
                    "text": claim.get("text", ""),
                    "claimer": claim.get("claimer", "unknown"),
                    "claimer_role": claimer_role,
                    "claim_type": claim_type,
                    "category": category,
                    "evidence": [
                        {
                            "evidence_quote": ev.get("evidence_quote", ""),
                            "evidence_type": ev_type_str,
                            "evidence_subtype": ev_subtype,
                            "source_entity": ev.get("source_entity"),
                            "source_person": ev.get("source_person"),
                            "source_role": ev.get("source_role"),
                            "source_type": ev.get("source_type"),
                            "is_named": ev.get("is_named", False),
                            "is_primary": ev.get("is_primary", False),
                        }
                    ],
                }
            )

        articles_out.append(
            {
                "article_id": article["article_id"],
                "network": article["network"],
                "program": article["program"],
                "date": article["date"],
                "url": article.get("url", ""),
                "word_count": article["word_count"],
                "num_claims": len(claims_out),
                "claims": claims_out,
            }
        )

    output = {
        "metrics": {},
        "articles": articles_out,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Converted to report format -> {output_path}")
    return output


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    n_articles: int = 100,
    seed: int = 42,
    category: str = "substantive_news",
    max_claims_per_article: int = 10,
    model: str = MODEL,
    poll_interval: int = 10,
    stages: list[str] | None = None,
    client: Anthropic | None = None,
) -> dict[str, Path]:
    """Run the full pipeline or specific stages.

    Args:
        input_path: Path to input samples.json
        output_dir: Directory for all output files
        n_articles: Number of articles to sample
        seed: Random seed
        category: Claim category to filter to
        max_claims_per_article: Max claims per article after filtering
        model: Claude model to use
        poll_interval: Seconds between batch status checks
        stages: List of stages to run, or None for all
        client: Anthropic client

    Returns:
        Dict mapping stage names to output paths
    """
    if client is None:
        from sourcerer.extract import get_client

        client = get_client()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "articles_sampled": output_dir / "articles_sampled.json",
        "claims_extracted": output_dir / "claims_extracted.json",
        "claims_filtered": output_dir / "claims_filtered.json",
        "evidence_classified": output_dir / "evidence_classified.json",
        "batch_results": output_dir / "batch_results.json",
    }

    if stages is None:
        stages = ["sample", "extract", "filter", "evidence"]

    if "sample" in stages:
        sample_articles(input_path, paths["articles_sampled"], n_articles, seed)

    if "extract" in stages:
        extract_claims_batch(
            paths["articles_sampled"],
            paths["claims_extracted"],
            client,
            model,
            poll_interval,
        )

    if "filter" in stages:
        filter_claims(
            paths["claims_extracted"],
            paths["claims_filtered"],
            category,
            max_claims_per_article,
            seed,
        )

    if "evidence" in stages:
        classify_evidence_batch(
            paths["claims_filtered"],
            paths["evidence_classified"],
            client,
            model,
            poll_interval,
        )
        convert_to_analyzed_format(
            paths["evidence_classified"],
            paths["batch_results"],
        )

    return paths
