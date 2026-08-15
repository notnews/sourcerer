"""Analysis and aggregation functions."""

from collections import Counter

from anthropic import Anthropic

from sourcerer.config import MODEL
from sourcerer.evidence import classify_all_evidence
from sourcerer.extract import extract_claims, get_client
from sourcerer.schema import AnalyzedArticle


def analyze_article(
    article_id: str,
    network: str,
    program: str,
    date: str,
    text: str,
    client: Anthropic | None = None,
    model: str = MODEL,
) -> AnalyzedArticle:
    """Analyze a single article for claims and evidence.

    Args:
        article_id: Unique identifier
        network: Network name (CNN, Fox, etc.)
        program: Program name
        date: Date string
        text: Article text
        client: Anthropic client
        model: Claude model to use

    Returns:
        AnalyzedArticle with claims and evidence
    """
    if client is None:
        client = get_client()

    word_count = len(text.split())

    claims = extract_claims(text, client, model)

    claims = classify_all_evidence(claims, text, client, model)

    return AnalyzedArticle(
        article_id=article_id,
        network=network,
        program=program,
        date=date,
        text=text,
        word_count=word_count,
        claims=claims,
    )


def compute_metrics(articles: list[AnalyzedArticle]) -> dict:
    """Compute aggregate metrics across articles.

    Args:
        articles: List of analyzed articles

    Returns:
        Dict with aggregate metrics
    """
    if not articles:
        return {}

    total_claims = sum(len(a.claims) for a in articles)
    total_words = sum(a.word_count for a in articles)

    evidence_counts: Counter[str] = Counter()
    evidence_subtype_counts: Counter[str] = Counter()
    claim_type_counts: Counter[str] = Counter()
    claimer_role_counts: Counter[str] = Counter()
    source_entity_counts: Counter[str] = Counter()
    source_role_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()

    attributed_claims_count = 0
    attributed_named_count = 0
    attributed_source_roles: Counter[str] = Counter()
    attributed_source_entities: Counter[str] = Counter()
    attributed_claimer_types: Counter[str] = Counter()
    attributed_source_types: Counter[str] = Counter()

    newsorg_claims_count = 0
    newsorg_evidence_types: Counter[str] = Counter()
    newsorg_evidence_subtypes: Counter[str] = Counter()

    for article in articles:
        for claim in article.claims:
            claim_type_counts[claim.claim_type.value] += 1
            claimer_role_counts[claim.claimer_role.value] += 1
            if claim.category:
                category_counts[claim.category.value] += 1

            is_attributed = claim.claimer_role.value in ("quoted_source", "guest")

            if is_attributed:
                attributed_claims_count += 1
                attributed_claimer_types[claim.claimer_role.value] += 1
                for evidence in claim.evidence:
                    if evidence.is_named:
                        attributed_named_count += 1
                    if evidence.source_role:
                        attributed_source_roles[evidence.source_role] += 1
                    if evidence.source_entity:
                        attributed_source_entities[evidence.source_entity] += 1
                    if evidence.source_type:
                        attributed_source_types[evidence.source_type.value] += 1
            else:
                newsorg_claims_count += 1
                for evidence in claim.evidence:
                    newsorg_evidence_types[evidence.evidence_type.value] += 1
                    newsorg_evidence_subtypes[evidence.evidence_subtype.value] += 1

            for evidence in claim.evidence:
                evidence_counts[evidence.evidence_type.value] += 1
                evidence_subtype_counts[evidence.evidence_subtype.value] += 1
                if evidence.source_entity:
                    source_entity_counts[evidence.source_entity] += 1
                if evidence.source_role:
                    source_role_counts[evidence.source_role] += 1

    by_network: dict[str, dict] = {}
    for article in articles:
        network = article.network
        if network not in by_network:
            by_network[network] = {
                "articles": 0,
                "claims": 0,
                "words": 0,
                "evidence": Counter(),
            }
        by_network[network]["articles"] += 1
        by_network[network]["claims"] += len(article.claims)
        by_network[network]["words"] += article.word_count
        for claim in article.claims:
            for evidence in claim.evidence:
                by_network[network]["evidence"][evidence.evidence_type.value] += 1

    network_metrics = {}
    for network, data in by_network.items():
        network_metrics[network] = {
            "articles": data["articles"],
            "total_claims": data["claims"],
            "total_words": data["words"],
            "claims_per_100_words": ((data["claims"] / data["words"] * 100) if data["words"] else 0),
            "evidence_distribution": dict(data["evidence"]),
        }

    named_sources = 0
    primary_sources = 0
    total_evidence = 0
    for article in articles:
        for claim in article.claims:
            for evidence in claim.evidence:
                total_evidence += 1
                if evidence.is_named:
                    named_sources += 1
                if evidence.is_primary:
                    primary_sources += 1

    return {
        "summary": {
            "total_articles": len(articles),
            "total_claims": total_claims,
            "total_words": total_words,
            "claims_per_100_words": ((total_claims / total_words * 100) if total_words else 0),
            "avg_claims_per_article": total_claims / len(articles) if articles else 0,
        },
        "evidence_distribution": dict(evidence_counts),
        "evidence_subtype_distribution": dict(evidence_subtype_counts),
        "claim_type_distribution": dict(claim_type_counts),
        "claimer_role_distribution": dict(claimer_role_counts),
        "claim_category_distribution": dict(category_counts),
        "source_entities": dict(source_entity_counts.most_common(50)),
        "source_roles": dict(source_role_counts),
        "attributed_claims": {
            "total": attributed_claims_count,
            "named_rate": (attributed_named_count / attributed_claims_count if attributed_claims_count else 0),
            "by_claimer_type": dict(attributed_claimer_types),
            "source_types": dict(attributed_source_types),
            "source_roles": dict(attributed_source_roles.most_common(20)),
            "source_entities": dict(attributed_source_entities.most_common(20)),
        },
        "newsorg_claims": {
            "total": newsorg_claims_count,
            "evidence_types": dict(newsorg_evidence_types),
            "evidence_subtypes": dict(newsorg_evidence_subtypes),
        },
        "source_quality": {
            "total_evidence_instances": total_evidence,
            "named_sources": named_sources,
            "named_source_rate": (named_sources / total_evidence if total_evidence else 0),
            "primary_sources": primary_sources,
            "primary_source_rate": (primary_sources / total_evidence if total_evidence else 0),
        },
        "by_network": network_metrics,
    }


def print_metrics(metrics: dict) -> None:
    """Print metrics in a readable format."""
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    summary = metrics.get("summary", {})
    print(f"\nArticles analyzed: {summary.get('total_articles', 0)}")
    print(f"Total claims extracted: {summary.get('total_claims', 0)}")
    print(f"Total words: {summary.get('total_words', 0):,}")
    print(f"Claims per 100 words: {summary.get('claims_per_100_words', 0):.2f}")
    print(f"Avg claims per article: {summary.get('avg_claims_per_article', 0):.1f}")

    print("\n--- Evidence Type Distribution ---")
    for etype, count in sorted(metrics.get("evidence_distribution", {}).items(), key=lambda x: -x[1]):
        print(f"  {etype}: {count}")

    print("\n--- Claim Type Distribution ---")
    for ctype, count in sorted(metrics.get("claim_type_distribution", {}).items(), key=lambda x: -x[1]):
        print(f"  {ctype}: {count}")

    print("\n--- Source Quality ---")
    sq = metrics.get("source_quality", {})
    print(f"  Named source rate: {sq.get('named_source_rate', 0):.1%}")
    print(f"  Primary source rate: {sq.get('primary_source_rate', 0):.1%}")

    print("\n--- By Network ---")
    for network, data in metrics.get("by_network", {}).items():
        print(f"\n  {network}:")
        print(f"    Articles: {data.get('articles', 0)}")
        print(f"    Claims: {data.get('total_claims', 0)}")
        print(f"    Claims/100 words: {data.get('claims_per_100_words', 0):.2f}")
        print(f"    Evidence: {data.get('evidence_distribution', {})}")

    print("\n" + "=" * 60)
