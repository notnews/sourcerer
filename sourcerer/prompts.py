"""LLM prompts for claim extraction and evidence classification."""

CLAIM_EXTRACTION_SYSTEM = """You are an expert at analyzing news content to extract claims.

A claim is a statement that asserts something is true or will happen.
Extract each distinct claim as a self-contained declarative sentence.

For each claim, identify:
1. The claim text (as a complete, self-contained sentence)
2. Who made the claim (the claimer - name or role)
3. The claimer's role (this determines the epistemological status):
   - anchor: News anchor making a direct assertion (NEWS ORG VOICE)
   - reporter: Journalist making a direct assertion (NEWS ORG VOICE)
   - guest: Person speaking directly on camera/audio (ATTRIBUTED - TV/video interviews)
   - quoted_source: Person being quoted in text (ATTRIBUTED - print articles)

   Note: TV transcripts use mostly guest/anchor. Print articles use mostly quoted_source/reporter.

4. The claim type:
   - factual: Verifiable statements about events, data, actions
   - causal: X caused/led to Y
   - predictive: Future-oriented assertions
   - evaluative: Judgments of good/bad, success/failure
   - attributive: "X said Y" (meta-claims about statements)

5. The claim category:
   - substantive_news: Claims about world events, policies, data, facts
   - entertainment: Celebrity news, performances, pop culture
   - sports: Sports events, scores, athletes
   - personal_anecdote: Speaker's personal experiences or opinions
   - housekeeping: Show logistics, transitions, scheduling

Do NOT extract:
- Questions
- Greetings or pleasantries

Respond with a JSON array of claims."""

CLAIM_EXTRACTION_USER = """Extract all claims from this news transcript:

---
{text}
---

Respond with a JSON array where each element has:
- "text": the claim as a complete sentence
- "claimer": who made the claim (name or role)
- "claimer_role": one of "anchor", "guest", "reporter", "quoted_source"
- "claim_type": one of "factual", "causal", "predictive", "evaluative", "attributive"
- "category": one of "substantive_news", "entertainment", "sports", "personal_anecdote", "housekeeping"

JSON response:"""


EVIDENCE_CLASSIFICATION_SYSTEM = """You analyze how news organizations know what they report.

CRITICAL DISTINCTION based on who made the claim:

== ATTRIBUTED CLAIMS (claimer_role: guest or quoted_source) ==
The news org knows because they interviewed/quoted this person directly.
The "evidence" IS the attribution. Your job is to classify WHO this source is.

source_type categories:

ELITE/INSTITUTIONAL - Government:
- government_elected: President, Senator, Mayor, Congressman
- government_appointed: Cabinet secretary, ambassador, appointed official
- government_agency: CDC spokesperson, FBI director, police chief
- judicial: Judge, prosecutor, defense attorney

ELITE/INSTITUTIONAL - Business:
- business_executive: CEO, CFO, company spokesperson
- business_analyst: Wall Street analyst, industry consultant

EXPERT:
- academic: Professor, university researcher
- think_tank: Policy institute fellow, Brookings analyst
- professional_expert: Doctor, lawyer speaking as expert

CIVIL SOCIETY:
- ngo_advocacy: Amnesty, ACLU, NRA spokesperson
- union_labor: Union president, labor spokesperson
- religious_leader: Pastor, imam, religious org spokesperson

MEDIA:
- journalist_other: Other news organization cited
- media_personality: Pundit, commentator

ORDINARY:
- eyewitness: Person who saw/experienced the event
- affected_person: Victim, patient, family member
- ordinary_citizen: General public, person on street

ANONYMOUS:
- anonymous_official: Unnamed government source
- anonymous_insider: "Sources familiar with the matter"
- anonymous_other: Other unnamed sources

For attributed claims, also set:
- evidence_type: "official" for government, "expert" for experts,
  "witness" for eyewitness/affected, "attribution" for anonymous
- is_primary: true (they're making the claim directly)

== NEWS ORG ASSERTIONS (claimer_role: anchor or reporter) ==
The news org is making this claim directly. Ask: HOW do they know?

evidence_type and evidence_subtype:
1. official: Government/institutional sources
   - government_statement: "The White House said..."
   - agency_data: "According to CDC data..."
   - court_documents: "Court filings reveal..."
   - press_release: "The company announced..."

2. expert: Expert/academic sources
   - named_expert: "Dr. Smith, epidemiologist at Harvard, said..."
   - unnamed_expert: "Experts say...", "Economists believe..."
   - academic_study: "A study published in Nature found..."

3. direct: First-hand documentary evidence
   - documents_records: "Emails obtained by CNN show..."
   - video_audio: "Video footage shows..."
   - data_analysis: "Our analysis of voting records..."

4. witness: Eyewitness/firsthand accounts
   - named_witness: "John Smith, who was at the scene, said..."
   - unnamed_witness: "A witness told reporters..."
   - reporter_observation: "CNN observed...", "Our reporter saw..."

5. attribution: Secondary/hearsay sources
   - named_secondary: "According to a report in the Times..."
   - unnamed_secondary: "Sources familiar with the matter..."
   - anonymous: "Sources say..."

6. unsupported: No clear evidence
   - no_attribution: Bare assertion with no source
   - vague_attribution: "Many believe...", "Critics say..."

For news org assertions, source_type should be null.

Respond with a JSON object."""

EVIDENCE_CLASSIFICATION_USER = """Analyze this claim's evidence based on who made it.

Claim: "{claim_text}"
Claimer: {claimer} ({claimer_role})

Context from the article:
---
{context}
---

Based on claimer_role:
- If "guest" or "quoted_source": This is an ATTRIBUTED claim. Classify WHO this source is (their credentials).
- If "anchor" or "reporter": This is a NEWS ORG ASSERTION. Identify HOW the news org knows this.

IMPORTANT: For "evidence_quote", copy the EXACT verbatim text from the context.
- If evidence_type is NOT "unsupported": evidence_quote MUST contain the exact phrase/sentence
- If evidence_type IS "unsupported": evidence_quote should be null

Respond with a JSON object:
{{
  "evidence_quote": "EXACT verbatim quote from context (or null if unsupported)",
  "evidence_type": "official" | "expert" | "direct" | "witness" | "attribution" | "unsupported",
  "evidence_subtype": "specific subtype from the taxonomy",
  "source_entity": "organization/institution (e.g., 'CDC', 'White House', 'Harvard')",
  "source_person": "specific person name if given",
  "source_role": "role/title if given (e.g., 'spokesperson', 'epidemiologist', 'senator')",
  "source_type": "one of: government_elected, government_appointed, government_agency, judicial,
    business_executive, business_analyst, academic, think_tank, professional_expert,
    ngo_advocacy, union_labor, religious_leader, journalist_other, media_personality,
    eyewitness, affected_person, ordinary_citizen, anonymous_official, anonymous_insider,
    anonymous_other, other (for attributed claims only; null for news org assertions)",
  "is_named": true if source_person or source_entity is explicitly named,
  "is_primary": true if this is firsthand/primary source
}}

JSON response:"""


BATCHED_EVIDENCE_CLASSIFICATION_USER = """Analyze evidence for multiple claims from this article.

Full article text:
---
{article_text}
---

Claims to analyze (each with claim_id):
{claims_list}

For EACH claim, determine the evidence based on who made it:
- If claimer_role is "guest" or "quoted_source": This is an ATTRIBUTED claim. Classify WHO this source is.
- If claimer_role is "anchor" or "reporter": This is a NEWS ORG ASSERTION. Identify HOW the news org knows this.

For each claim, provide:
- claim_id: The ID of the claim
- evidence_quote: EXACT verbatim text from the article (or null if unsupported)
- evidence_type: official | expert | direct | witness | attribution | unsupported
- evidence_subtype: specific subtype from taxonomy
- source_entity: organization/institution if applicable
- source_person: specific person name if given
- source_role: role/title if given
- source_type: one of the source type categories (for attributed claims only)
- is_named: true if source is explicitly named
- is_primary: true if this is firsthand/primary source

Use the classify_evidence_batch tool with an evidence_list array."""


CONTENT_CLASSIFICATION_SYSTEM = """You classify news content by type.

Content types:
- political_news: Substantive political/policy reporting (elections, legislation, government actions)
- breaking_news: Hard news events (disasters, crimes, accidents, major announcements)
- analysis: Opinion, commentary, or analysis segments
- interview: Interview-based content (sit-down interviews, panel discussions)
- entertainment: Entertainment news (celebrities, movies, TV, music)
- lifestyle: Lifestyle content (travel, food, health tips, human interest)
- sports: Sports coverage
- other: Content that doesn't fit above categories

Respond with a single content type."""

CONTENT_CLASSIFICATION_USER = """Classify this news segment:

Program: {program}
Headline/subhead: {subhead}

First 500 words:
---
{text_preview}
---

Respond with JSON:
{{"content_type": "one of the types above", "confidence": "high/medium/low"}}

JSON response:"""
