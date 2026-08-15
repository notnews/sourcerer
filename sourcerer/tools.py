"""Tool definitions for structured output via Claude tool_use.

These tools ensure guaranteed schema compliance for claim extraction
and evidence classification responses.
"""

from sourcerer.schema import (
    ClaimCategory,
    ClaimerRole,
    ClaimType,
    EvidenceSubtype,
    EvidenceType,
    SourceType,
)

CLAIM_EXTRACTION_TOOL = {
    "name": "extract_claims",
    "description": "Extract claims from news text with structured metadata",
    "input_schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "description": "List of claims extracted from the text",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The claim as a complete, self-contained sentence",
                        },
                        "claimer": {
                            "type": "string",
                            "description": "Who made the claim (name or role)",
                        },
                        "claimer_role": {
                            "type": "string",
                            "enum": [r.value for r in ClaimerRole],
                            "description": "The role of the person making the claim",
                        },
                        "claim_type": {
                            "type": "string",
                            "enum": [t.value for t in ClaimType],
                            "description": "The type of claim",
                        },
                        "category": {
                            "type": "string",
                            "enum": [c.value for c in ClaimCategory],
                            "description": "The category of the claim",
                        },
                    },
                    "required": ["text", "claimer", "claimer_role", "claim_type", "category"],
                },
            }
        },
        "required": ["claims"],
    },
}

EVIDENCE_CLASSIFICATION_TOOL = {
    "name": "classify_evidence",
    "description": "Classify evidence supporting a claim",
    "input_schema": {
        "type": "object",
        "properties": {
            "evidence_quote": {
                "type": ["string", "null"],
                "description": "Exact verbatim quote from context supporting the classification (null if unsupported)",
            },
            "evidence_type": {
                "type": "string",
                "enum": [e.value for e in EvidenceType],
                "description": "The type of evidence",
            },
            "evidence_subtype": {
                "type": "string",
                "enum": [s.value for s in EvidenceSubtype],
                "description": "The specific subtype of evidence",
            },
            "source_entity": {
                "type": ["string", "null"],
                "description": "Organization/institution (e.g., 'CDC', 'White House')",
            },
            "source_person": {
                "type": ["string", "null"],
                "description": "Specific person name if given",
            },
            "source_role": {
                "type": ["string", "null"],
                "description": "Role/title if given (e.g., 'spokesperson', 'senator')",
            },
            "source_type": {
                "type": ["string", "null"],
                "enum": [st.value for st in SourceType] + [None],
                "description": "Classification of source credentials (for attributed claims only)",
            },
            "is_named": {
                "type": "boolean",
                "description": "True if source_person or source_entity is explicitly named",
            },
            "is_primary": {
                "type": "boolean",
                "description": "True if this is firsthand/primary source",
            },
        },
        "required": ["evidence_type", "evidence_subtype", "is_named", "is_primary"],
    },
}

BATCHED_EVIDENCE_CLASSIFICATION_TOOL = {
    "name": "classify_evidence_batch",
    "description": "Classify evidence for multiple claims from the same article",
    "input_schema": {
        "type": "object",
        "properties": {
            "evidence_list": {
                "type": "array",
                "description": "Evidence classification for each claim",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {
                            "type": "string",
                            "description": "The ID of the claim this evidence applies to",
                        },
                        "evidence_quote": {
                            "type": ["string", "null"],
                            "description": "Exact verbatim quote from context (null if unsupported)",
                        },
                        "evidence_type": {
                            "type": "string",
                            "enum": [e.value for e in EvidenceType],
                        },
                        "evidence_subtype": {
                            "type": "string",
                            "enum": [s.value for s in EvidenceSubtype],
                        },
                        "source_entity": {
                            "type": ["string", "null"],
                        },
                        "source_person": {
                            "type": ["string", "null"],
                        },
                        "source_role": {
                            "type": ["string", "null"],
                        },
                        "source_type": {
                            "type": ["string", "null"],
                            "enum": [st.value for st in SourceType] + [None],
                        },
                        "is_named": {
                            "type": "boolean",
                        },
                        "is_primary": {
                            "type": "boolean",
                        },
                    },
                    "required": ["claim_id", "evidence_type", "evidence_subtype", "is_named", "is_primary"],
                },
            }
        },
        "required": ["evidence_list"],
    },
}
