"""Batch processing using Anthropic Message Batches API.

The batch API provides:
- 50% cost reduction compared to standard API
- Up to 100,000 requests per batch
- Results available within 24 hours (usually much faster)

Supports both legacy JSON mode and structured tool_use mode.
"""

import json
import time
from typing import Any, Optional

from anthropic import Anthropic

from sourcerer.config import EXTRACTION_MAX_CHARS, MAX_TOKENS, MODEL, THINKING
from sourcerer.evidence import SUBTYPE_TO_TYPE, get_context_window
from sourcerer.prompts import (
    CLAIM_EXTRACTION_SYSTEM,
    CLAIM_EXTRACTION_USER,
    EVIDENCE_CLASSIFICATION_SYSTEM,
    EVIDENCE_CLASSIFICATION_USER,
)
from sourcerer.schema import (
    AnalyzedArticle,
    Claim,
    ClaimCategory,
    ClaimerRole,
    ClaimType,
    Evidence,
    EvidenceSubtype,
    EvidenceType,
    SourceType,
)
from sourcerer.tools import CLAIM_EXTRACTION_TOOL, EVIDENCE_CLASSIFICATION_TOOL


def create_claim_extraction_batch(
    articles: list[dict],
    client: Anthropic,
    model: str = MODEL,
) -> str:
    """Create a batch for claim extraction.

    Args:
        articles: List of dicts with 'id' and 'text' keys
        client: Anthropic client
        model: Model to use

    Returns:
        Batch ID
    """
    requests = []
    for article in articles:
        text = article["text"]
        if len(text) > EXTRACTION_MAX_CHARS:
            text = text[:EXTRACTION_MAX_CHARS]

        requests.append(
            {
                "custom_id": f"claims_{article['id']}",
                "params": {
                    "model": model,
                    "max_tokens": MAX_TOKENS,
                    "thinking": THINKING,
                    "system": CLAIM_EXTRACTION_SYSTEM,
                    "messages": [{"role": "user", "content": CLAIM_EXTRACTION_USER.format(text=text)}],
                },
            }
        )

    batch = client.messages.batches.create(requests=requests)
    return batch.id


def create_evidence_batch(
    claims_with_context: list[dict],
    client: Anthropic,
    model: str = MODEL,
) -> str:
    """Create a batch for evidence classification.

    Args:
        claims_with_context: List of dicts with 'id', 'claim_text', 'context' keys
        client: Anthropic client
        model: Model to use

    Returns:
        Batch ID
    """
    requests = []
    for item in claims_with_context:
        requests.append(
            {
                "custom_id": f"evidence_{item['id']}",
                "params": {
                    "model": model,
                    "max_tokens": 1024,
                    "thinking": THINKING,
                    "system": EVIDENCE_CLASSIFICATION_SYSTEM,
                    "messages": [
                        {
                            "role": "user",
                            "content": EVIDENCE_CLASSIFICATION_USER.format(
                                claim_text=item["claim_text"],
                                context=item["context"],
                            ),
                        }
                    ],
                },
            }
        )

    batch = client.messages.batches.create(requests=requests)
    return batch.id


def wait_for_batch(
    batch_id: str,
    client: Anthropic,
    poll_interval: int = 10,
    max_wait: int = 3600,
) -> dict:
    """Wait for batch to complete and return results.

    Args:
        batch_id: Batch ID to wait for
        client: Anthropic client
        poll_interval: Seconds between status checks
        max_wait: Maximum seconds to wait

    Returns:
        Dict mapping custom_id to result
    """
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
            content = result.result.message.content[0].text
            results[result.custom_id] = content

    return results


def parse_claims_response(content: str) -> list[dict]:
    """Parse JSON response from claim extraction."""
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return []


def parse_evidence_response(content: str) -> dict:
    """Parse JSON response from evidence classification."""
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "evidence_type": "unsupported",
            "evidence_subtype": "no_attribution",
            "is_named": False,
            "is_primary": False,
        }


def analyze_batch(
    articles: list[dict],
    client: Optional[Anthropic] = None,
    model: str = MODEL,
    poll_interval: int = 10,
) -> list[AnalyzedArticle]:
    """Analyze articles using batch API.

    Args:
        articles: List of dicts with keys: id, network, program, date, text
        client: Anthropic client
        model: Model to use
        poll_interval: Seconds between status checks

    Returns:
        List of AnalyzedArticle objects
    """
    if client is None:
        from sourcerer.extract import get_client

        client = get_client()

    print(f"\nPhase 1: Extracting claims from {len(articles)} articles...")
    batch_id = create_claim_extraction_batch(articles, client, model)
    print(f"  Created batch: {batch_id}")

    claims_results = wait_for_batch(batch_id, client, poll_interval)
    print(f"  Got results for {len(claims_results)} articles")

    all_claims = {}
    claims_for_evidence = []

    for article in articles:
        article_id = article["id"]
        result_key = f"claims_{article_id}"

        if result_key not in claims_results:
            all_claims[article_id] = []
            continue

        claims_data = parse_claims_response(claims_results[result_key])
        all_claims[article_id] = claims_data

        for i, claim_data in enumerate(claims_data):
            claim_text = claim_data.get("text", "")
            context = get_context_window(article["text"], claim_text)
            claims_for_evidence.append(
                {
                    "id": f"{article_id}_{i}",
                    "article_id": article_id,
                    "claim_index": i,
                    "claim_text": claim_text,
                    "context": context,
                }
            )

    print(f"\nPhase 2: Classifying evidence for {len(claims_for_evidence)} claims...")

    if claims_for_evidence:
        batch_id = create_evidence_batch(claims_for_evidence, client, model)
        print(f"  Created batch: {batch_id}")

        evidence_results = wait_for_batch(batch_id, client, poll_interval)
        print(f"  Got results for {len(evidence_results)} claims")
    else:
        evidence_results = {}

    print("\nPhase 3: Building final results...")
    analyzed = []

    for article in articles:
        article_id = article["id"]
        claims_data = all_claims.get(article_id, [])

        claims = []
        for i, claim_data in enumerate(claims_data):
            try:
                claim_type = ClaimType(claim_data.get("claim_type", "factual"))
            except ValueError:
                claim_type = ClaimType.FACTUAL

            try:
                claimer_role = ClaimerRole(claim_data.get("claimer_role", "anchor"))
            except ValueError:
                claimer_role = ClaimerRole.ANCHOR

            evidence_key = f"evidence_{article_id}_{i}"
            if evidence_key in evidence_results:
                ev_data = parse_evidence_response(evidence_results[evidence_key])
            else:
                ev_data = {
                    "evidence_type": "unsupported",
                    "evidence_subtype": "no_attribution",
                }

            subtype_str = ev_data.get("evidence_subtype", "no_attribution")
            try:
                evidence_subtype = EvidenceSubtype(subtype_str)
            except ValueError:
                evidence_subtype = EvidenceSubtype.NO_ATTRIBUTION

            evidence_type = SUBTYPE_TO_TYPE.get(subtype_str, EvidenceType.UNSUPPORTED)

            evidence = Evidence(
                quote=ev_data.get("evidence_quote") or "",
                evidence_type=evidence_type,
                evidence_subtype=evidence_subtype,
                source_entity=ev_data.get("source_entity"),
                source_person=ev_data.get("source_person"),
                source_role=ev_data.get("source_role"),
                is_named=ev_data.get("is_named", False),
                is_primary=ev_data.get("is_primary", False),
            )

            claim = Claim(
                text=claim_data.get("text", ""),
                claimer=claim_data.get("claimer", "unknown"),
                claimer_role=claimer_role,
                claim_type=claim_type,
                evidence=[evidence],
            )
            claims.append(claim)

        analyzed_article = AnalyzedArticle(
            article_id=article_id,
            network=article.get("network", "unknown"),
            program=article.get("program", "unknown"),
            date=article.get("date", ""),
            text=article["text"],
            word_count=len(article["text"].split()),
            url=article.get("url", ""),
            claims=claims,
        )
        analyzed.append(analyzed_article)

    return analyzed


def create_claim_extraction_batch_tooluse(
    articles: list[dict],
    client: Anthropic,
    model: str = MODEL,
) -> str:
    """Create a batch for claim extraction using tool_use.

    This version uses structured output via tool_use for guaranteed
    schema compliance.

    Args:
        articles: List of dicts with 'id' and 'text' keys
        client: Anthropic client
        model: Model to use

    Returns:
        Batch ID
    """
    requests = []
    for article in articles:
        text = article["text"]
        if len(text) > EXTRACTION_MAX_CHARS:
            text = text[:EXTRACTION_MAX_CHARS]

        requests.append(
            {
                "custom_id": f"claims_{article['id']}",
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
    return batch.id


def create_evidence_batch_tooluse(
    claims_with_context: list[dict],
    client: Anthropic,
    model: str = MODEL,
) -> str:
    """Create a batch for evidence classification using tool_use.

    Args:
        claims_with_context: List of dicts with 'id', 'claim_text', 'claimer',
            'claimer_role', 'context' keys
        client: Anthropic client
        model: Model to use

    Returns:
        Batch ID
    """
    requests = []
    for item in claims_with_context:
        requests.append(
            {
                "custom_id": f"evidence_{item['id']}",
                "params": {
                    "model": model,
                    "max_tokens": 1024,
                    "thinking": THINKING,
                    "system": EVIDENCE_CLASSIFICATION_SYSTEM,
                    "messages": [
                        {
                            "role": "user",
                            "content": EVIDENCE_CLASSIFICATION_USER.format(
                                claim_text=item["claim_text"],
                                claimer=item.get("claimer", "unknown"),
                                claimer_role=item.get("claimer_role", "anchor"),
                                context=item["context"],
                            ),
                        }
                    ],
                    "tools": [EVIDENCE_CLASSIFICATION_TOOL],
                    "tool_choice": {"type": "tool", "name": "classify_evidence"},
                },
            }
        )

    batch = client.messages.batches.create(requests=requests)
    return batch.id


def wait_for_batch_tooluse(
    batch_id: str,
    client: Anthropic,
    poll_interval: int = 10,
    max_wait: int = 3600,
) -> dict[str, Any]:
    """Wait for batch to complete and return tool_use results.

    Args:
        batch_id: Batch ID to wait for
        client: Anthropic client
        poll_interval: Seconds between status checks
        max_wait: Maximum seconds to wait

    Returns:
        Dict mapping custom_id to tool input (parsed JSON)
    """
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


def analyze_batch_tooluse(
    articles: list[dict],
    client: Anthropic | None = None,
    model: str = MODEL,
    poll_interval: int = 10,
) -> list[AnalyzedArticle]:
    """Analyze articles using batch API with tool_use for structured output.

    This is the preferred method as it guarantees schema compliance.

    Args:
        articles: List of dicts with keys: id, network, program, date, text
        client: Anthropic client
        model: Model to use
        poll_interval: Seconds between status checks

    Returns:
        List of AnalyzedArticle objects
    """
    if client is None:
        from sourcerer.extract import get_client

        client = get_client()

    print(f"\nPhase 1: Extracting claims from {len(articles)} articles (tool_use)...")
    batch_id = create_claim_extraction_batch_tooluse(articles, client, model)
    print(f"  Created batch: {batch_id}")

    claims_results = wait_for_batch_tooluse(batch_id, client, poll_interval)
    print(f"  Got results for {len(claims_results)} articles")

    all_claims: dict[str, list[dict]] = {}
    claims_for_evidence: list[dict] = []

    for article in articles:
        article_id = article["id"]
        result_key = f"claims_{article_id}"

        if result_key not in claims_results:
            all_claims[article_id] = []
            continue

        claims_data = claims_results[result_key].get("claims", [])
        all_claims[article_id] = claims_data

        for i, claim_data in enumerate(claims_data):
            claim_text = claim_data.get("text", "")
            context = get_context_window(article["text"], claim_text)
            claims_for_evidence.append(
                {
                    "id": f"{article_id}_{i}",
                    "article_id": article_id,
                    "claim_index": i,
                    "claim_text": claim_text,
                    "claimer": claim_data.get("claimer", "unknown"),
                    "claimer_role": claim_data.get("claimer_role", "anchor"),
                    "context": context,
                }
            )

    print(f"\nPhase 2: Classifying evidence for {len(claims_for_evidence)} claims (tool_use)...")

    if claims_for_evidence:
        batch_id = create_evidence_batch_tooluse(claims_for_evidence, client, model)
        print(f"  Created batch: {batch_id}")

        evidence_results = wait_for_batch_tooluse(batch_id, client, poll_interval)
        print(f"  Got results for {len(evidence_results)} claims")
    else:
        evidence_results = {}

    print("\nPhase 3: Building final results...")
    analyzed = []

    for article in articles:
        article_id = article["id"]
        claims_data = all_claims.get(article_id, [])

        claims = []
        for i, claim_data in enumerate(claims_data):
            try:
                claim_type = ClaimType(claim_data.get("claim_type", "factual"))
            except ValueError:
                claim_type = ClaimType.FACTUAL

            try:
                claimer_role = ClaimerRole(claim_data.get("claimer_role", "anchor"))
            except ValueError:
                claimer_role = ClaimerRole.ANCHOR

            category = None
            if claim_data.get("category"):
                try:
                    category = ClaimCategory(claim_data["category"])
                except ValueError:
                    category = ClaimCategory.SUBSTANTIVE_NEWS

            evidence_key = f"evidence_{article_id}_{i}"
            if evidence_key in evidence_results:
                ev_data = evidence_results[evidence_key]
            else:
                ev_data = {
                    "evidence_type": "unsupported",
                    "evidence_subtype": "no_attribution",
                }

            subtype_str = ev_data.get("evidence_subtype", "no_attribution")
            try:
                evidence_subtype = EvidenceSubtype(subtype_str)
            except ValueError:
                evidence_subtype = EvidenceSubtype.NO_ATTRIBUTION

            evidence_type = SUBTYPE_TO_TYPE.get(subtype_str, EvidenceType.UNSUPPORTED)

            source_type = None
            source_type_str = ev_data.get("source_type")
            if source_type_str:
                try:
                    source_type = SourceType(source_type_str)
                except ValueError:
                    source_type = SourceType.OTHER

            evidence = Evidence(
                quote=ev_data.get("evidence_quote") or "",
                evidence_type=evidence_type,
                evidence_subtype=evidence_subtype,
                source_entity=ev_data.get("source_entity"),
                source_person=ev_data.get("source_person"),
                source_role=ev_data.get("source_role"),
                source_type=source_type,
                is_named=ev_data.get("is_named", False),
                is_primary=ev_data.get("is_primary", False),
            )

            claim = Claim(
                text=claim_data.get("text", ""),
                claimer=claim_data.get("claimer", "unknown"),
                claimer_role=claimer_role,
                claim_type=claim_type,
                category=category,
                claim_id=f"{article_id}_{i:03d}",
                evidence=[evidence],
            )
            claims.append(claim)

        analyzed_article = AnalyzedArticle(
            article_id=article_id,
            network=article.get("network", "unknown"),
            program=article.get("program", "unknown"),
            date=article.get("date", ""),
            text=article["text"],
            word_count=len(article["text"].split()),
            url=article.get("url", ""),
            claims=claims,
        )
        analyzed.append(analyzed_article)

    return analyzed
