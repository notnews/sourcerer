"""Tests for schema module."""

from sourcerer.schema import (
    AnalyzedArticle,
    Claim,
    ClaimerRole,
    ClaimType,
    Evidence,
    EvidenceSubtype,
    EvidenceType,
    SourceType,
)


def test_evidence_creation():
    evidence = Evidence(
        quote="According to CDC data",
        evidence_type=EvidenceType.OFFICIAL,
        evidence_subtype=EvidenceSubtype.AGENCY_DATA,
        source_entity="CDC",
        is_named=True,
        is_primary=True,
    )
    assert evidence.evidence_type == EvidenceType.OFFICIAL
    assert evidence.is_named is True
    assert evidence.quote == "According to CDC data"


def test_claim_creation():
    claim = Claim(
        text="The unemployment rate fell to 3.5%",
        claimer="anchor",
        claimer_role=ClaimerRole.ANCHOR,
        claim_type=ClaimType.FACTUAL,
        evidence=[],
    )
    assert claim.claim_type == ClaimType.FACTUAL
    assert claim.claimer_role == ClaimerRole.ANCHOR


def test_claim_with_evidence():
    evidence = Evidence(
        quote="Bureau of Labor Statistics reported",
        evidence_type=EvidenceType.OFFICIAL,
        evidence_subtype=EvidenceSubtype.AGENCY_DATA,
        source_entity="Bureau of Labor Statistics",
        is_named=True,
        is_primary=True,
    )
    claim = Claim(
        text="The unemployment rate fell to 3.5%",
        claimer="anchor",
        claimer_role=ClaimerRole.ANCHOR,
        claim_type=ClaimType.FACTUAL,
        evidence=[evidence],
    )
    assert len(claim.evidence) == 1
    assert claim.evidence[0].source_entity == "Bureau of Labor Statistics"


def test_analyzed_article():
    article = AnalyzedArticle(
        article_id="test-001",
        network="CNN",
        program="Anderson Cooper 360",
        date="2024-01-15",
        text="This is a test article with one hundred words. " * 10,
        word_count=100,
        claims=[],
    )
    assert article.claims_per_100_words == 0.0


def test_analyzed_article_with_claims():
    claim = Claim(
        text="Test claim",
        claimer="anchor",
        claimer_role=ClaimerRole.ANCHOR,
        claim_type=ClaimType.FACTUAL,
        evidence=[
            Evidence(
                quote="Source said",
                evidence_type=EvidenceType.ATTRIBUTION,
                evidence_subtype=EvidenceSubtype.ANONYMOUS,
                is_named=False,
                is_primary=False,
            )
        ],
    )
    article = AnalyzedArticle(
        article_id="test-002",
        network="Fox",
        program="Hannity",
        date="2024-01-16",
        text="Test " * 200,
        word_count=200,
        claims=[claim],
    )
    assert article.claims_per_100_words == 0.5
    dist = article.evidence_distribution()
    assert dist.get("attribution") == 1


def test_to_dict():
    evidence = Evidence(
        quote="Dr. Smith said",
        evidence_type=EvidenceType.EXPERT,
        evidence_subtype=EvidenceSubtype.NAMED_EXPERT,
        source_entity="Harvard University",
        source_person="Dr. Smith",
        source_role="epidemiologist",
        is_named=True,
        is_primary=True,
    )
    claim = Claim(
        text="The vaccine is effective",
        claimer="Dr. Smith",
        claimer_role=ClaimerRole.GUEST,
        claim_type=ClaimType.FACTUAL,
        evidence=[evidence],
    )
    article = AnalyzedArticle(
        article_id="test-003",
        network="MSNBC",
        program="Rachel Maddow Show",
        date="2024-01-17",
        text="Test",
        word_count=100,
        url="https://example.com",
        claims=[claim],
    )

    result = article.to_dict()
    assert result["article_id"] == "test-003"
    assert result["network"] == "MSNBC"
    assert result["url"] == "https://example.com"
    assert len(result["claims"]) == 1
    assert result["claims"][0]["claimer"] == "Dr. Smith"
    assert result["claims"][0]["evidence"][0]["source_entity"] == "Harvard University"
    assert result["claims"][0]["evidence"][0]["source_person"] == "Dr. Smith"
    assert result["claims"][0]["evidence"][0]["source_role"] == "epidemiologist"
    assert result["claims"][0]["evidence"][0]["evidence_quote"] == "Dr. Smith said"


def test_source_type_enum():
    assert SourceType.GOVERNMENT_ELECTED.value == "government_elected"
    assert SourceType.ACADEMIC.value == "academic"
    assert SourceType.EYEWITNESS.value == "eyewitness"
    assert SourceType.ANONYMOUS_OFFICIAL.value == "anonymous_official"


def test_evidence_with_source_type():
    evidence = Evidence(
        quote="Senator Smith said",
        evidence_type=EvidenceType.OFFICIAL,
        evidence_subtype=EvidenceSubtype.GOVERNMENT_STATEMENT,
        source_entity="US Senate",
        source_person="Senator Smith",
        source_role="Senator",
        source_type=SourceType.GOVERNMENT_ELECTED,
        is_named=True,
        is_primary=True,
    )
    assert evidence.source_type == SourceType.GOVERNMENT_ELECTED
    result = evidence.to_dict()
    assert result["source_type"] == "government_elected"


def test_evidence_source_type_none_for_newsorg():
    evidence = Evidence(
        quote="According to data",
        evidence_type=EvidenceType.OFFICIAL,
        evidence_subtype=EvidenceSubtype.AGENCY_DATA,
        source_entity="CDC",
        source_type=None,
        is_named=True,
        is_primary=False,
    )
    assert evidence.source_type is None
    result = evidence.to_dict()
    assert result["source_type"] is None
