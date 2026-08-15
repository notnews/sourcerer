"""Data models for claims and evidence."""

from dataclasses import dataclass, field
from enum import Enum


class ClaimType(str, Enum):
    FACTUAL = "factual"
    CAUSAL = "causal"
    PREDICTIVE = "predictive"
    EVALUATIVE = "evaluative"
    ATTRIBUTIVE = "attributive"


class EvidenceType(str, Enum):
    OFFICIAL = "official"
    EXPERT = "expert"
    DIRECT = "direct"
    WITNESS = "witness"
    ATTRIBUTION = "attribution"
    UNSUPPORTED = "unsupported"


class EvidenceSubtype(str, Enum):
    # Official/Institutional
    GOVERNMENT_STATEMENT = "government_statement"
    AGENCY_DATA = "agency_data"
    COURT_DOCUMENTS = "court_documents"
    PRESS_RELEASE = "press_release"
    # Expert
    NAMED_EXPERT = "named_expert"
    UNNAMED_EXPERT = "unnamed_expert"
    ACADEMIC_STUDY = "academic_study"
    # Direct Evidence
    DOCUMENTS_RECORDS = "documents_records"
    VIDEO_AUDIO = "video_audio"
    DATA_ANALYSIS = "data_analysis"
    # Witness/Firsthand
    NAMED_WITNESS = "named_witness"
    UNNAMED_WITNESS = "unnamed_witness"
    REPORTER_OBSERVATION = "reporter_observation"
    # Attribution (hearsay)
    NAMED_SECONDARY = "named_secondary"
    UNNAMED_SECONDARY = "unnamed_secondary"
    ANONYMOUS = "anonymous"
    # Unsupported
    NO_ATTRIBUTION = "no_attribution"
    VAGUE_ATTRIBUTION = "vague_attribution"


class ClaimerRole(str, Enum):
    ANCHOR = "anchor"
    GUEST = "guest"
    REPORTER = "reporter"
    QUOTED_SOURCE = "quoted_source"


class SourceType(str, Enum):
    """Classification of source credentials for attributed claims.

    Based on journalism research on news source typology.
    """

    # Elite/Institutional - Government
    GOVERNMENT_ELECTED = "government_elected"
    GOVERNMENT_APPOINTED = "government_appointed"
    GOVERNMENT_AGENCY = "government_agency"
    JUDICIAL = "judicial"
    # Elite/Institutional - Business
    BUSINESS_EXECUTIVE = "business_executive"
    BUSINESS_ANALYST = "business_analyst"
    # Expert
    ACADEMIC = "academic"
    THINK_TANK = "think_tank"
    PROFESSIONAL_EXPERT = "professional_expert"
    # Civil Society
    NGO_ADVOCACY = "ngo_advocacy"
    UNION_LABOR = "union_labor"
    RELIGIOUS_LEADER = "religious_leader"
    # Media
    JOURNALIST_OTHER = "journalist_other"
    MEDIA_PERSONALITY = "media_personality"
    # Ordinary
    EYEWITNESS = "eyewitness"
    AFFECTED_PERSON = "affected_person"
    ORDINARY_CITIZEN = "ordinary_citizen"
    # Anonymous
    ANONYMOUS_OFFICIAL = "anonymous_official"
    ANONYMOUS_INSIDER = "anonymous_insider"
    ANONYMOUS_OTHER = "anonymous_other"
    # Fallback
    OTHER = "other"


class ContentType(str, Enum):
    POLITICAL_NEWS = "political_news"
    BREAKING_NEWS = "breaking_news"
    ANALYSIS = "analysis"
    INTERVIEW = "interview"
    ENTERTAINMENT = "entertainment"
    LIFESTYLE = "lifestyle"
    SPORTS = "sports"
    OTHER = "other"


class ClaimCategory(str, Enum):
    SUBSTANTIVE_NEWS = "substantive_news"
    ENTERTAINMENT = "entertainment"
    SPORTS = "sports"
    PERSONAL_ANECDOTE = "personal_anecdote"
    HOUSEKEEPING = "housekeeping"


@dataclass
class Evidence:
    """Evidence supporting a claim."""

    quote: str  # Verbatim quote from article supporting classification
    evidence_type: EvidenceType
    evidence_subtype: EvidenceSubtype
    source_entity: str | None = None
    source_person: str | None = None
    source_role: str | None = None
    source_type: SourceType | None = None
    is_named: bool = False
    is_primary: bool = False

    def to_dict(self) -> dict:
        return {
            "evidence_quote": self.quote,
            "evidence_type": self.evidence_type.value,
            "evidence_subtype": self.evidence_subtype.value,
            "source_entity": self.source_entity,
            "source_person": self.source_person,
            "source_role": self.source_role,
            "source_type": self.source_type.value if self.source_type else None,
            "is_named": self.is_named,
            "is_primary": self.is_primary,
        }


@dataclass
class Claim:
    """A claim extracted from an article with its evidence."""

    text: str
    claimer: str
    claimer_role: ClaimerRole
    claim_type: ClaimType
    category: ClaimCategory | None = None
    claim_id: str | None = None
    evidence: list[Evidence] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "claimer": self.claimer,
            "claimer_role": self.claimer_role.value,
            "claim_type": self.claim_type.value,
            "category": self.category.value if self.category else None,
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass
class AnalyzedArticle:
    """An article with extracted claims and evidence."""

    article_id: str
    network: str
    program: str
    date: str
    text: str
    word_count: int
    url: str = ""
    content_type: ContentType | None = None
    claims: list[Claim] = field(default_factory=list)

    @property
    def claims_per_100_words(self) -> float:
        if self.word_count == 0:
            return 0.0
        return (len(self.claims) / self.word_count) * 100

    def evidence_distribution(self) -> dict[str, int]:
        """Count evidence by type across all claims."""
        counts: dict[str, int] = {}
        for claim in self.claims:
            for evidence in claim.evidence:
                key = evidence.evidence_type.value
                counts[key] = counts.get(key, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "article_id": self.article_id,
            "network": self.network,
            "program": self.program,
            "date": self.date,
            "url": self.url,
            "content_type": self.content_type.value if self.content_type else None,
            "word_count": self.word_count,
            "num_claims": len(self.claims),
            "claims_per_100_words": self.claims_per_100_words,
            "evidence_distribution": self.evidence_distribution(),
            "claims": [c.to_dict() for c in self.claims],
        }
