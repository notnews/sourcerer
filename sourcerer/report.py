"""Generate HTML and Markdown reports from analysis results."""

from collections import defaultdict
from pathlib import Path

import pandas as pd

from sourcerer.schema import AnalyzedArticle, EvidenceSubtype, EvidenceType


def generate_markdown_report(
    articles: list[AnalyzedArticle],
    metrics: dict,
    output_path: Path | None = None,
) -> str:
    """Generate a Markdown report from analysis results.

    Args:
        articles: List of analyzed articles
        metrics: Computed metrics dict
        output_path: Optional path to write report

    Returns:
        Markdown string
    """
    lines = []
    lines.append("# Sourcerer: Epistemological Analysis Report\n")

    summary = metrics.get("summary", {})
    lines.append("## Summary\n")
    lines.append(f"- **Articles analyzed:** {summary.get('total_articles', 0)}")
    lines.append(f"- **Total claims:** {summary.get('total_claims', 0)}")
    lines.append(f"- **Total words:** {summary.get('total_words', 0):,}")
    lines.append(f"- **Claims per 100 words:** {summary.get('claims_per_100_words', 0):.2f}")
    lines.append(f"- **Avg claims per article:** {summary.get('avg_claims_per_article', 0):.1f}")
    lines.append("")

    lines.append("## Evidence Type Distribution\n")
    lines.append("| Evidence Type | Count | Percentage |")
    lines.append("|---------------|-------|------------|")
    ev_dist = metrics.get("evidence_distribution", {})
    total_ev = sum(ev_dist.values()) if ev_dist else 1
    for etype, count in sorted(ev_dist.items(), key=lambda x: -x[1]):
        pct = (count / total_ev * 100) if total_ev else 0
        lines.append(f"| {etype} | {count} | {pct:.1f}% |")
    lines.append("")

    lines.append("## Claim Type Distribution\n")
    lines.append("| Claim Type | Count | Percentage |")
    lines.append("|------------|-------|------------|")
    claim_dist = metrics.get("claim_type_distribution", {})
    total_claims = sum(claim_dist.values()) if claim_dist else 1
    for ctype, count in sorted(claim_dist.items(), key=lambda x: -x[1]):
        pct = (count / total_claims * 100) if total_claims else 0
        lines.append(f"| {ctype} | {count} | {pct:.1f}% |")
    lines.append("")

    sq = metrics.get("source_quality", {})
    lines.append("## Source Quality\n")
    lines.append(f"- **Named source rate:** {sq.get('named_source_rate', 0):.1%}")
    lines.append(f"- **Primary source rate:** {sq.get('primary_source_rate', 0):.1%}")
    lines.append("")

    if metrics.get("by_network"):
        lines.append("## By Network\n")
        lines.append("| Network | Articles | Claims | Claims/100 words |")
        lines.append("|---------|----------|--------|------------------|")
        for network, data in metrics["by_network"].items():
            cper100 = data["claims_per_100_words"]
            lines.append(f"| {network} | {data['articles']} | {data['total_claims']} | {cper100:.2f} |")
        lines.append("")

    lines.append("## Sample Claims with Evidence\n")
    for article in articles[:3]:
        lines.append(f"### {article.program} ({article.date})\n")
        for i, claim in enumerate(article.claims[:5], 1):
            claim_text = claim.text[:150] + "..." if len(claim.text) > 150 else claim.text
            lines.append(f"**{i}. {claim_text}**\n")
            lines.append(f"- *Claimer:* {claim.claimer} ({claim.claimer_role.value})")
            lines.append(f"- *Claim type:* {claim.claim_type.value}")
            for ev in claim.evidence:
                lines.append(f"- *Evidence:* {ev.evidence_type.value} / {ev.evidence_subtype.value}")
                if ev.source_entity:
                    lines.append(f"- *Source:* {ev.source_entity}")
            lines.append("")
        lines.append("---\n")

    content = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)

    return content


SUBTYPE_TO_PARENT = {
    EvidenceSubtype.GOVERNMENT_STATEMENT: EvidenceType.OFFICIAL,
    EvidenceSubtype.AGENCY_DATA: EvidenceType.OFFICIAL,
    EvidenceSubtype.COURT_DOCUMENTS: EvidenceType.OFFICIAL,
    EvidenceSubtype.PRESS_RELEASE: EvidenceType.OFFICIAL,
    EvidenceSubtype.NAMED_EXPERT: EvidenceType.EXPERT,
    EvidenceSubtype.UNNAMED_EXPERT: EvidenceType.EXPERT,
    EvidenceSubtype.ACADEMIC_STUDY: EvidenceType.EXPERT,
    EvidenceSubtype.DOCUMENTS_RECORDS: EvidenceType.DIRECT,
    EvidenceSubtype.VIDEO_AUDIO: EvidenceType.DIRECT,
    EvidenceSubtype.DATA_ANALYSIS: EvidenceType.DIRECT,
    EvidenceSubtype.NAMED_WITNESS: EvidenceType.WITNESS,
    EvidenceSubtype.UNNAMED_WITNESS: EvidenceType.WITNESS,
    EvidenceSubtype.REPORTER_OBSERVATION: EvidenceType.WITNESS,
    EvidenceSubtype.NAMED_SECONDARY: EvidenceType.ATTRIBUTION,
    EvidenceSubtype.UNNAMED_SECONDARY: EvidenceType.ATTRIBUTION,
    EvidenceSubtype.ANONYMOUS: EvidenceType.ATTRIBUTION,
    EvidenceSubtype.NO_ATTRIBUTION: EvidenceType.UNSUPPORTED,
    EvidenceSubtype.VAGUE_ATTRIBUTION: EvidenceType.UNSUPPORTED,
}


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def generate_html_report(
    articles: list[AnalyzedArticle],
    metrics: dict,
    output_path: Path | None = None,
) -> str:
    """Generate an HTML report from analysis results.

    Args:
        articles: List of analyzed articles
        metrics: Computed metrics dict
        output_path: Optional path to write report

    Returns:
        HTML string
    """
    summary = metrics.get("summary", {})
    ev_dist = metrics.get("evidence_distribution", {})
    ev_subtype_dist = metrics.get("evidence_subtype_distribution", {})
    claim_dist = metrics.get("claim_type_distribution", {})
    category_dist = metrics.get("claim_category_distribution", {})
    source_entities = metrics.get("source_entities", {})
    source_roles = metrics.get("source_roles", {})
    sq = metrics.get("source_quality", {})
    total_ev = sum(ev_dist.values()) if ev_dist else 1
    total_subtype_ev = sum(ev_subtype_dist.values()) if ev_subtype_dist else 1
    total_claims = sum(claim_dist.values()) if claim_dist else 1
    total_category = sum(category_dist.values()) if category_dist else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sourcerer: Epistemological Analysis Report</title>
    <style>
        :root {{
            --primary: #2563eb;
            --success: #16a34a;
            --warning: #d97706;
            --danger: #dc2626;
            --gray: #6b7280;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background: #f9fafb;
            color: #1f2937;
        }}
        h1 {{
            color: var(--primary);
            border-bottom: 3px solid var(--primary);
            padding-bottom: 0.5rem;
        }}
        h2 {{ color: #374151; margin-top: 2rem; }}
        h3 {{ color: #4b5563; }}
        h4 {{ color: #6b7280; margin-top: 1.5rem; margin-bottom: 0.5rem; }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0 2rem;
        }}
        .stat-card {{
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .stat-value {{ font-size: 2rem; font-weight: bold; color: var(--primary); }}
        .stat-label {{ color: var(--gray); font-size: 0.9rem; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }}
        th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #e5e7eb; }}
        th {{ background: #f3f4f6; font-weight: 600; }}
        tr:hover {{ background: #f9fafb; }}
        .parent-row {{ background: #e5e7eb; font-weight: 600; }}
        .parent-row:hover {{ background: #d1d5db; }}
        .subtype-row td:first-child {{ padding-left: 2rem; }}
        .claim-card {{
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .claim-text {{ font-size: 1.1rem; font-weight: 500; margin-bottom: 0.75rem; }}
        .claim-meta {{ color: var(--gray); font-size: 0.9rem; }}
        .evidence-quote {{
            background: #f3f4f6;
            border-left: 3px solid var(--primary);
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            font-style: italic;
            font-size: 0.9rem;
        }}
        .evidence-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 500;
            margin: 0.25rem 0.25rem 0.25rem 0;
        }}
        .ev-official {{ background: #dbeafe; color: #1e40af; }}
        .ev-expert {{ background: #dcfce7; color: #166534; }}
        .ev-direct {{ background: #fef3c7; color: #92400e; }}
        .ev-witness {{ background: #f3e8ff; color: #6b21a8; }}
        .ev-attribution {{ background: #fee2e2; color: #991b1b; }}
        .ev-unsupported {{ background: #f3f4f6; color: #4b5563; }}
        .bar-chart {{ margin: 1rem 0; }}
        .bar {{
            height: 24px;
            background: var(--primary);
            border-radius: 4px;
            margin: 0.25rem 0;
            display: flex;
            align-items: center;
            padding: 0 0.5rem;
            color: white;
            font-size: 0.8rem;
        }}
        .bar-label {{ display: flex; justify-content: space-between; font-size: 0.9rem; }}
        .two-column {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }}
        @media (max-width: 768px) {{ .two-column {{ grid-template-columns: 1fr; }} }}
        .subtype-section {{ margin-top: 1.5rem; }}
        .subtype-header {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }}
    </style>
</head>
<body>
    <h1>Sourcerer: Epistemological Analysis Report</h1>

    <h2>Summary</h2>
    <div class="summary-grid">
        <div class="stat-card">
            <div class="stat-value">{summary.get("total_articles", 0)}</div>
            <div class="stat-label">Articles Analyzed</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{summary.get("total_claims", 0)}</div>
            <div class="stat-label">Claims Extracted</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{summary.get("claims_per_100_words", 0):.2f}</div>
            <div class="stat-label">Claims per 100 Words</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{sq.get("named_source_rate", 0):.0%}</div>
            <div class="stat-label">Named Source Rate</div>
        </div>
    </div>

    <h2>Evidence Type Distribution</h2>
    <div class="bar-chart">
"""

    for etype, count in sorted(ev_dist.items(), key=lambda x: -x[1]):
        pct = (count / total_ev * 100) if total_ev else 0
        width = max(pct, 5)
        html += f'        <div class="bar-label"><span>{etype}</span>'
        html += f"<span>{count} ({pct:.1f}%)</span></div>\n"
        html += f'        <div class="bar ev-{etype}" style="width: {width}%">{count}</div>\n'

    html += """    </div>
"""

    if category_dist:
        html += """
    <h2>Claim Category Distribution</h2>
    <p>Classification of claims by topic/domain (filtering to substantive_news
    recommended for analysis):</p>
    <div class="bar-chart">
"""
        category_colors = {
            "substantive_news": "#16a34a",
            "entertainment": "#f59e0b",
            "sports": "#3b82f6",
            "personal_anecdote": "#8b5cf6",
            "housekeeping": "#6b7280",
        }
        for rel, count in sorted(category_dist.items(), key=lambda x: -x[1]):
            pct = (count / total_category * 100) if total_category else 0
            width = max(pct, 5)
            color = category_colors.get(rel, "#6b7280")
            display = rel.replace("_", " ")
            html += f'        <div class="bar-label"><span>{display}</span>'
            html += f"<span>{count} ({pct:.1f}%)</span></div>\n"
            html += f'        <div class="bar" style="width: {width}%; background: {color}">'
            html += f"{count}</div>\n"
        html += "    </div>\n"

    html += """
    <h2>Evidence Subtype Distribution (17 Categories)</h2>
    <table>
        <thead><tr><th>Evidence Type / Subtype</th><th>Count</th><th>Percentage</th></tr></thead>
        <tbody>
"""

    subtypes_by_parent: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for subtype_str, count in ev_subtype_dist.items():
        try:
            subtype_enum = EvidenceSubtype(subtype_str)
            parent = SUBTYPE_TO_PARENT.get(subtype_enum)
            if parent:
                subtypes_by_parent[parent.value].append((subtype_str, count))
        except ValueError:
            subtypes_by_parent["unknown"].append((subtype_str, count))

    parent_order = [
        "official",
        "expert",
        "direct",
        "witness",
        "attribution",
        "unsupported",
    ]
    for parent_type in parent_order:
        if parent_type not in ev_dist:
            continue
        parent_count = ev_dist.get(parent_type, 0)
        parent_pct = (parent_count / total_ev * 100) if total_ev else 0
        ptype_up = parent_type.upper()
        html += '            <tr class="parent-row">'
        html += f"<td>{ptype_up}</td><td>{parent_count}</td><td>{parent_pct:.1f}%</td></tr>\n"

        subtypes = subtypes_by_parent.get(parent_type, [])
        for subtype_str, count in sorted(subtypes, key=lambda x: -x[1]):
            pct = (count / total_subtype_ev * 100) if total_subtype_ev else 0
            display_name = subtype_str.replace("_", " ")
            html += '            <tr class="subtype-row">'
            html += f"<td>{display_name}</td><td>{count}</td><td>{pct:.1f}%</td></tr>\n"

    html += """        </tbody>
    </table>
"""

    attributed = metrics.get("attributed_claims", {})
    newsorg = metrics.get("newsorg_claims", {})

    if attributed.get("total") or newsorg.get("total"):
        attr_total = attributed.get("total", 0)
        news_total = newsorg.get("total", 0)
        total = attr_total + news_total
        attr_pct = (attr_total / total * 100) if total else 0
        news_pct = (news_total / total * 100) if total else 0

        html += f"""
    <h2>Claim Attribution Analysis</h2>
    <p>Distinguishing attributed claims (quotes/guests) from news org assertions:</p>
    <div class="summary-grid">
        <div class="stat-card">
            <div class="stat-value">{attr_total}</div>
            <div class="stat-label">Attributed Claims ({attr_pct:.0f}%)</div>
            <div class="claim-meta">Quotes & guests - evidence is who said it</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{news_total}</div>
            <div class="stat-label">News Org Assertions ({news_pct:.0f}%)</div>
            <div class="claim-meta">Anchors & reporters - how do they know?</div>
        </div>
    </div>
"""

        if attributed.get("source_types"):
            html += """
    <h3>Attributed Claims: Source Type Distribution</h3>
    <p>Classification of sources by category (who is being quoted/interviewed):</p>
    <div class="bar-chart">
"""
            source_types = attributed["source_types"]
            total_st = sum(source_types.values()) if source_types else 1
            source_type_colors = {
                "government_elected": "#1e40af",
                "government_appointed": "#1e40af",
                "government_agency": "#1e40af",
                "judicial": "#1e40af",
                "business_executive": "#0369a1",
                "business_analyst": "#0369a1",
                "academic": "#166534",
                "think_tank": "#166534",
                "professional_expert": "#166534",
                "ngo_advocacy": "#7c3aed",
                "union_labor": "#7c3aed",
                "religious_leader": "#7c3aed",
                "journalist_other": "#ea580c",
                "media_personality": "#ea580c",
                "eyewitness": "#0891b2",
                "affected_person": "#0891b2",
                "ordinary_citizen": "#0891b2",
                "anonymous_official": "#6b7280",
                "anonymous_insider": "#6b7280",
                "anonymous_other": "#6b7280",
                "other": "#9ca3af",
            }
            for stype, count in sorted(source_types.items(), key=lambda x: -x[1]):
                pct = (count / total_st * 100) if total_st else 0
                width = max(pct, 5)
                color = source_type_colors.get(stype, "#6b7280")
                display = stype.replace("_", " ")
                html += f'        <div class="bar-label"><span>{display}</span>'
                html += f"<span>{count} ({pct:.1f}%)</span></div>\n"
                html += f'        <div class="bar" style="width: {width}%; background: {color}">'
                html += f"{count}</div>\n"
            html += "    </div>\n"

        if attributed.get("source_roles"):
            html += """
    <h4>Source Roles (Titles/Positions)</h4>
    <table>
        <thead><tr><th>Source Role</th><th>Count</th></tr></thead>
        <tbody>
"""
            for role, count in sorted(attributed["source_roles"].items(), key=lambda x: -x[1])[:15]:
                html += f"            <tr><td>{_escape_html(role)}</td><td>{count}</td></tr>\n"
            html += """        </tbody>
    </table>
"""

        if newsorg.get("evidence_types"):
            html += """
    <h3>News Org Assertions: How Do They Know?</h3>
    <p>Evidence supporting claims made directly by anchors and reporters:</p>
    <div class="bar-chart">
"""
            newsorg_ev = newsorg["evidence_types"]
            total_newsorg_ev = sum(newsorg_ev.values()) if newsorg_ev else 1
            for etype, count in sorted(newsorg_ev.items(), key=lambda x: -x[1]):
                pct = (count / total_newsorg_ev * 100) if total_newsorg_ev else 0
                width = max(pct, 5)
                html += f'        <div class="bar-label"><span>{etype}</span>'
                html += f"<span>{count} ({pct:.1f}%)</span></div>\n"
                html += f'        <div class="bar ev-{etype}" style="width: {width}%">'
                html += f"{count}</div>\n"
            html += "    </div>\n"

    html += """
    <h2>Source Entity Analysis</h2>
    <p>Top cited source entities (organizations, institutions, agencies):</p>
    <table>
        <thead><tr><th>Source Entity</th><th>Citations</th></tr></thead>
        <tbody>
"""

    for entity, count in sorted(source_entities.items(), key=lambda x: -x[1])[:20]:
        html += f"            <tr><td>{_escape_html(entity)}</td><td>{count}</td></tr>\n"

    if not source_entities:
        html += '            <tr><td colspan="2"><em>No source entities identified</em></td></tr>\n'

    html += """        </tbody>
    </table>

    <h2>Source Role Distribution</h2>
    <table>
        <thead><tr><th>Role</th><th>Count</th></tr></thead>
        <tbody>
"""

    total_roles = sum(source_roles.values()) if source_roles else 1
    for role, count in sorted(source_roles.items(), key=lambda x: -x[1])[:15]:
        pct = (count / total_roles * 100) if total_roles else 0
        html += f"            <tr><td>{_escape_html(role)}</td><td>{count} ({pct:.1f}%)</td></tr>\n"

    if not source_roles:
        html += '            <tr><td colspan="2"><em>No source roles identified</em></td></tr>\n'

    html += """        </tbody>
    </table>

    <h2>Claim Type Distribution</h2>
    <table>
        <thead><tr><th>Claim Type</th><th>Count</th><th>Percentage</th></tr></thead>
        <tbody>
"""

    for ctype, count in sorted(claim_dist.items(), key=lambda x: -x[1]):
        pct = (count / total_claims * 100) if total_claims else 0
        html += f"            <tr><td>{ctype}</td><td>{count}</td><td>{pct:.1f}%</td></tr>\n"

    html += """        </tbody>
    </table>

    <h2>Sample Claims with Evidence</h2>
"""

    has_category = any(c.category for a in articles for c in a.claims)
    if has_category:
        news_claims_total = sum(
            1 for a in articles for c in a.claims if c.category and c.category.value == "substantive_news"
        )
        all_claims_total = sum(len(a.claims) for a in articles)
        html += "    <p>Showing substantive news claims only "
        html += f"({news_claims_total} of {all_claims_total} total claims):</p>\n"
    else:
        html += "    <p>Showing up to 30 claims (10 articles × 3 claims each):</p>\n"

    shown_count = 0
    for article in articles:
        if shown_count >= 30:
            break
        article_claims = []
        for claim in article.claims:
            if has_category:
                if claim.category and claim.category.value == "substantive_news":
                    article_claims.append(claim)
            else:
                article_claims.append(claim)
            if len(article_claims) >= 3:
                break
        if not article_claims:
            continue
        html += f"    <h3>{_escape_html(article.program)} ({article.date})</h3>\n"
        for i, claim in enumerate(article_claims, 1):
            shown_count += 1
            claim_text = claim.text[:200] + "..." if len(claim.text) > 200 else claim.text
            ev_class = claim.evidence[0].evidence_type.value if claim.evidence else "unsupported"
            if claim.evidence:
                ev = claim.evidence[0]
                ev_text = f"{ev.evidence_type.value}/{ev.evidence_subtype.value}"
            else:
                ev_text = "unsupported"
            source = claim.evidence[0].source_entity if claim.evidence and claim.evidence[0].source_entity else ""
            quote = claim.evidence[0].quote if claim.evidence else ""

            html += f"""    <div class="claim-card">
        <div class="claim-text">{i}. {_escape_html(claim_text)}</div>
        <div class="claim-meta">
            <strong>Claimer:</strong> {_escape_html(claim.claimer)} ({claim.claimer_role.value}) &nbsp;|&nbsp;
            <strong>Type:</strong> {claim.claim_type.value}
        </div>
        <div style="margin-top: 0.5rem;">
            <span class="evidence-badge ev-{ev_class}">{ev_text}</span>
            {f'<span class="claim-meta">Source: {_escape_html(source)}</span>' if source else ""}
        </div>
"""
            if quote:
                quote_display = quote[:300] + "..." if len(quote) > 300 else quote
                escaped_quote = _escape_html(quote_display)
                html += f'        <div class="evidence-quote">"{escaped_quote}"</div>\n'

            html += "    </div>\n"

    examples_by_subtype: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for article in articles:
        for claim in article.claims:
            if has_category:
                if not claim.category or claim.category.value != "substantive_news":
                    continue
            for evidence in claim.evidence:
                subtype = evidence.evidence_subtype.value
                if len(examples_by_subtype[subtype]) < 2:
                    examples_by_subtype[subtype].append(
                        (
                            (claim.text[:150] + "..." if len(claim.text) > 150 else claim.text),
                            (evidence.quote[:250] + "..." if len(evidence.quote) > 250 else evidence.quote),
                            evidence.source_entity or "",
                            article.program,
                        )
                    )

    html += """
    <h2>Evidence by Subtype: Examples</h2>
    <p>Example claims for each evidence subtype to illustrate the classification taxonomy:</p>
"""

    for parent_type in parent_order:
        subtypes_in_parent = [s.value for s, p in SUBTYPE_TO_PARENT.items() if p.value == parent_type]
        has_examples = any(examples_by_subtype.get(s) for s in subtypes_in_parent)

        if not has_examples:
            continue

        html += '    <div class="subtype-section">\n'
        ptype_up = parent_type.upper()
        html += '    <h3 style="margin-bottom: 0.5rem;">'
        html += f'<span class="evidence-badge ev-{parent_type}">{ptype_up}</span></h3>\n'

        for subtype in subtypes_in_parent:
            examples = examples_by_subtype.get(subtype, [])
            if not examples:
                continue

            display_name = subtype.replace("_", " ").title()
            html += f"    <h4>{display_name}</h4>\n"

            for claim_text, quote, source, program in examples:
                esc_claim = _escape_html(claim_text)
                esc_quote = _escape_html(quote)
                esc_prog = _escape_html(program)
                src_str = f" | Source: {_escape_html(source)}" if source else ""
                html += '    <div class="claim-card" style="margin: 0.5rem 0; padding: 1rem;">\n'
                html += '        <div class="claim-text" style="font-size: 0.95rem;">'
                html += f"{esc_claim}</div>\n"
                html += f'        <div class="evidence-quote">"{esc_quote}"</div>\n'
                html += f'        <div class="claim-meta">{esc_prog}{src_str}</div>\n'
                html += "    </div>\n"

        html += "    </div>\n"

    html += """
</body>
</html>
"""

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)

    return html


def export_claims_csv(
    articles: list[AnalyzedArticle],
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Export claims to a flat CSV file.

    Each row is one claim with its evidence. Columns:
    - network, program, date, url, content_type
    - claim, claimer, claimer_role, claim_type
    - evidence_type, evidence_subtype, evidence_quote
    - source_entity, source_person, source_role, source_type, is_named, is_primary

    Args:
        articles: List of analyzed articles
        output_path: Optional path to write CSV

    Returns:
        DataFrame with all claims
    """
    rows = []

    for article in articles:
        content_type = article.content_type.value if article.content_type else ""
        for claim in article.claims:
            for evidence in claim.evidence:
                rows.append(
                    {
                        "article_id": article.article_id,
                        "claim_id": claim.claim_id or "",
                        "network": article.network,
                        "program": article.program,
                        "date": article.date,
                        "url": article.url,
                        "content_type": content_type,
                        "claim": claim.text,
                        "claimer": claim.claimer,
                        "claimer_role": claim.claimer_role.value,
                        "claim_type": claim.claim_type.value,
                        "claim_category": (claim.category.value if claim.category else ""),
                        "evidence_type": evidence.evidence_type.value,
                        "evidence_subtype": evidence.evidence_subtype.value,
                        "evidence_quote": evidence.quote,
                        "source_entity": evidence.source_entity or "",
                        "source_person": evidence.source_person or "",
                        "source_role": evidence.source_role or "",
                        "source_type": (evidence.source_type.value if evidence.source_type else ""),
                        "is_named": evidence.is_named,
                        "is_primary": evidence.is_primary,
                    }
                )

    df = pd.DataFrame(rows)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df
