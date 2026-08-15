"""Evidence classification using Claude API."""

import json

from anthropic import Anthropic

from sourcerer.extract import get_client
from sourcerer.prompts import EVIDENCE_CLASSIFICATION_SYSTEM, EVIDENCE_CLASSIFICATION_USER
from sourcerer.schema import Claim, Evidence, EvidenceSubtype, EvidenceType, SourceType

SUBTYPE_TO_TYPE = {
    "government_statement": EvidenceType.OFFICIAL,
    "agency_data": EvidenceType.OFFICIAL,
    "court_documents": EvidenceType.OFFICIAL,
    "press_release": EvidenceType.OFFICIAL,
    "named_expert": EvidenceType.EXPERT,
    "unnamed_expert": EvidenceType.EXPERT,
    "academic_study": EvidenceType.EXPERT,
    "documents_records": EvidenceType.DIRECT,
    "video_audio": EvidenceType.DIRECT,
    "data_analysis": EvidenceType.DIRECT,
    "named_witness": EvidenceType.WITNESS,
    "unnamed_witness": EvidenceType.WITNESS,
    "reporter_observation": EvidenceType.WITNESS,
    "named_secondary": EvidenceType.ATTRIBUTION,
    "unnamed_secondary": EvidenceType.ATTRIBUTION,
    "anonymous": EvidenceType.ATTRIBUTION,
    "no_attribution": EvidenceType.UNSUPPORTED,
    "vague_attribution": EvidenceType.UNSUPPORTED,
}


def get_context_window(text: str, claim_text: str, window_chars: int = 500) -> str:
    """Extract context around a claim from the full text."""
    claim_pos = text.find(claim_text[:50]) if len(claim_text) > 50 else text.find(claim_text)
    if claim_pos == -1:
        return text[:2000]

    start = max(0, claim_pos - window_chars)
    end = min(len(text), claim_pos + len(claim_text) + window_chars)
    return text[start:end]


def classify_evidence(
    claim: Claim,
    full_text: str,
    client: Anthropic | None = None,
    model: str = "claude-sonnet-4-20250514",
) -> Evidence:
    """Classify evidence supporting a claim.

    Args:
        claim: The claim to analyze
        full_text: Full article text for context
        client: Anthropic client
        model: Claude model to use

    Returns:
        Evidence object
    """
    if client is None:
        client = get_client()

    context = get_context_window(full_text, claim.text)

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=EVIDENCE_CLASSIFICATION_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": EVIDENCE_CLASSIFICATION_USER.format(
                    claim_text=claim.text,
                    claimer=claim.claimer,
                    claimer_role=claim.claimer_role.value,
                    context=context,
                ),
            }
        ],
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
        data = json.loads(content)
    except json.JSONDecodeError:
        return Evidence(
            quote="",
            evidence_type=EvidenceType.UNSUPPORTED,
            evidence_subtype=EvidenceSubtype.NO_ATTRIBUTION,
            source_entity=None,
            is_named=False,
            is_primary=False,
        )

    subtype_str = data.get("evidence_subtype", "no_attribution")
    try:
        evidence_subtype = EvidenceSubtype(subtype_str)
    except ValueError:
        evidence_subtype = EvidenceSubtype.NO_ATTRIBUTION

    evidence_type = SUBTYPE_TO_TYPE.get(subtype_str, EvidenceType.UNSUPPORTED)

    source_type_str = data.get("source_type")
    source_type = None
    if source_type_str:
        try:
            source_type = SourceType(source_type_str)
        except ValueError:
            source_type = SourceType.OTHER

    return Evidence(
        quote=data.get("evidence_quote") or "",
        evidence_type=evidence_type,
        evidence_subtype=evidence_subtype,
        source_entity=data.get("source_entity"),
        source_person=data.get("source_person"),
        source_role=data.get("source_role"),
        source_type=source_type,
        is_named=data.get("is_named", False),
        is_primary=data.get("is_primary", False),
    )


def classify_all_evidence(
    claims: list[Claim],
    full_text: str,
    client: Anthropic | None = None,
    model: str = "claude-sonnet-4-20250514",
) -> list[Claim]:
    """Classify evidence for all claims.

    Args:
        claims: List of claims to analyze
        full_text: Full article text
        client: Anthropic client
        model: Claude model to use

    Returns:
        Claims with evidence attached
    """
    if client is None:
        client = get_client()

    for claim in claims:
        evidence = classify_evidence(claim, full_text, client, model)
        claim.evidence.append(evidence)

    return claims
