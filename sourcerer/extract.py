"""Claim extraction using Claude API."""

import json
import os

from anthropic import Anthropic

from sourcerer.prompts import CLAIM_EXTRACTION_SYSTEM, CLAIM_EXTRACTION_USER
from sourcerer.schema import Claim, ClaimCategory, ClaimerRole, ClaimType


def get_client() -> Anthropic:
    """Get Anthropic client from environment."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return Anthropic(api_key=api_key)


def extract_claims(
    text: str,
    client: Anthropic | None = None,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096,
) -> list[Claim]:
    """Extract claims from article text using Claude.

    Args:
        text: The article/transcript text
        client: Anthropic client (created if not provided)
        model: Claude model to use
        max_tokens: Maximum tokens in response

    Returns:
        List of Claim objects
    """
    if client is None:
        client = get_client()

    if len(text) > 50000:
        text = text[:50000]

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=CLAIM_EXTRACTION_SYSTEM,
        messages=[{"role": "user", "content": CLAIM_EXTRACTION_USER.format(text=text)}],
    )

    content = response.content[0].text.strip()

    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        claims_data = json.loads(content)
    except json.JSONDecodeError:
        return []

    claims = []
    for item in claims_data:
        try:
            claim_type = ClaimType(item.get("claim_type", "factual"))
        except ValueError:
            claim_type = ClaimType.FACTUAL

        try:
            claimer_role = ClaimerRole(item.get("claimer_role", "anchor"))
        except ValueError:
            claimer_role = ClaimerRole.ANCHOR

        category = None
        if item.get("category"):
            try:
                category = ClaimCategory(item["category"])
            except ValueError:
                category = ClaimCategory.SUBSTANTIVE_NEWS

        claim = Claim(
            text=item.get("text", ""),
            claimer=item.get("claimer", "unknown"),
            claimer_role=claimer_role,
            claim_type=claim_type,
            category=category,
            evidence=[],
        )
        claims.append(claim)

    return claims
