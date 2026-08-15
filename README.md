## Sourcerer: Analyzing the Evidence Behind News Claims

Sourcerer analyzes news articles to classify *how claims are epistemologically supported*—the types of evidence backing each claim (official sources, expert testimony, documents, hearsay, etc.).

### Why This Matters

The information age has brought both bounty and pestilence. We're deluged with correct and incorrect information, and fact-checking every claim is implausibly expensive.

Rather than verify claims directly, we can assess the *residue of the process*—the sources. Features of these sources help infer credibility:
- **Number of sources**: More corroboration → higher confidence
- **Diversity of sources**: Independent confirmations strengthen claims
- **Source credibility**: Official records vs. anonymous tips

Sourcerer provides systematic measurement of these epistemological qualities.

---

## Pipeline

```
[News Article/Transcript]
           │
           ▼
┌──────────────────────────┐
│ 1. Content Classification │  Political news? Entertainment? Sports?
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ 2. Claim Extraction       │  Extract atomic claims with claimer info
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ 3. Evidence Classification │  Classify supporting evidence per claim
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ 4. Aggregation           │  Article/network-level metrics
└──────────────────────────┘
           │
           ▼
    [Structured Output]
     - JSON with full details
     - CSV for analysis
     - HTML/Markdown reports
```

---

## Taxonomy

### Content Types
| Type | Description |
|------|-------------|
| `political_news` | Substantive political/policy reporting |
| `breaking_news` | Hard news events |
| `analysis` | Opinion/commentary segments |
| `interview` | Interview-based content |
| `entertainment` | Entertainment news |
| `lifestyle` | Lifestyle, travel, human interest |
| `sports` | Sports coverage |
| `other` | Doesn't fit above |

### Claim Types
| Type | Description | Example |
|------|-------------|---------|
| `factual` | Verifiable statements | "The unemployment rate is 3.5%" |
| `causal` | X caused Y | "Tax cuts led to the deficit" |
| `predictive` | Future assertions | "Inflation will rise next quarter" |
| `evaluative` | Judgments | "The policy was a failure" |
| `attributive` | X said Y | "The Senator stated that..." |

### Evidence Types

| Category | Subtype | Example |
|----------|---------|---------|
| **official** | `government_statement` | "The White House said..." |
| | `agency_data` | "According to CDC data..." |
| | `court_documents` | "Court filings reveal..." |
| | `press_release` | "The company announced..." |
| **expert** | `named_expert` | "Dr. Fauci, infectious disease specialist, said..." |
| | `unnamed_expert` | "Experts say..." |
| | `academic_study` | "A study in Nature found..." |
| **direct** | `documents_records` | "Emails obtained by CNN show..." |
| | `video_audio` | "Video footage shows..." |
| | `data_analysis` | "Our analysis of voting records..." |
| **witness** | `named_witness` | "John Smith, who was present, said..." |
| | `unnamed_witness` | "A witness told reporters..." |
| | `reporter_observation` | "CNN observed..." |
| **attribution** | `named_secondary` | "According to a Times report..." |
| | `unnamed_secondary` | "Sources familiar with the matter..." |
| | `anonymous` | "Sources say..." |
| **unsupported** | `no_attribution` | Bare assertion with no source |
| | `vague_attribution` | "Many believe...", "Critics say..." |

### Source Attribution Fields
For granular tracking of *who* is being cited:
- `source_entity`: Organization/institution (e.g., "CDC", "White House", "Harvard")
- `source_person`: Specific person (e.g., "Dr. Anthony Fauci", "Press Secretary")
- `source_role`: Title/role (e.g., "epidemiologist", "spokesperson", "senator")

---

## Output Format

### CSV Columns
Each row is one claim-evidence pair:

| Column | Description |
|--------|-------------|
| `network` | News organization |
| `program` | Show name |
| `date` | Air date |
| `url` | Source URL |
| `content_type` | political_news, entertainment, etc. |
| `claim` | The claim text |
| `claimer` | Who made the claim |
| `claimer_role` | anchor/guest/reporter/quoted_source |
| `claim_type` | factual/causal/predictive/evaluative/attributive |
| `evidence_type` | official/expert/direct/witness/attribution/unsupported |
| `evidence_subtype` | Specific subtype |
| `evidence_quote` | Verbatim quote from article |
| `source_entity` | Organization cited |
| `source_person` | Person cited |
| `source_role` | Role/title |
| `is_named` | Is source explicitly named? |
| `is_primary` | Is this firsthand evidence? |

### Example Output

```
claim: "The unemployment rate fell to 3.5% in December."
claimer: Anderson Cooper
claimer_role: anchor
claim_type: factual
evidence_type: official
evidence_subtype: agency_data
evidence_quote: "According to Bureau of Labor Statistics data released Friday"
source_entity: Bureau of Labor Statistics
source_person: null
source_role: null
is_named: true
is_primary: true
```

```
claim: "The administration is considering additional sanctions."
claimer: White House correspondent
claimer_role: reporter
claim_type: factual
evidence_type: attribution
evidence_subtype: unnamed_secondary
evidence_quote: "Sources familiar with the deliberations say"
source_entity: White House
source_person: null
source_role: null
is_named: false
is_primary: false
```

---

## Installation

```bash
pip install sourcerer
# or
pip install -e ".[dev]"
```

Requires `ANTHROPIC_API_KEY` environment variable.

## Usage

### Analyze Transcripts (Batch API - 50% cost savings)
```bash
python scripts/analyze_batch.py --input data/samples.json --output data/results.json
```

### Generate Reports
```bash
# HTML report
python scripts/generate_report.py --format html

# CSV for analysis
python scripts/generate_report.py --format csv

# All formats
python scripts/generate_report.py --format all
```

### Demo on Single Text
```bash
python scripts/demo.py "Your news text here..."
python scripts/demo.py --file transcript.txt
```

---

## Data

Currently analyzing CNN transcripts from 2024, focusing on substantive political news content (filtering out entertainment, lifestyle, and sports segments).

Sample data structure in `data/samples.json`:
```json
{
  "url": "https://transcripts.cnn.com/...",
  "network": "CNN",
  "program": "Amanpour",
  "date": "January 15, 2024",
  "subhead": "Interview with...",
  "text": "..."
}
```

---

## Authors

Gaurav Sood

## Background

The argument motivating this package — why checking the provenance of claims
scales better than checking the claims themselves, and where it stops working —
is in [`docs/motivation.md`](docs/motivation.md).

The source *typology* half of that argument is built in
[`covered`](https://github.com/notnews/covered), which works at the level of the
source rather than the claim.
