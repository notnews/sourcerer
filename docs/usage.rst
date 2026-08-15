Usage
=====

Basic Usage
-----------

Extract claims from a news transcript:

.. code-block:: python

   from sourcerer.extract import extract_claims, get_client
   from sourcerer.evidence import classify_all_evidence

   client = get_client()

   text = """
   COOPER: The unemployment rate fell to 3.5% last month.
   SECRETARY YELLEN: We are seeing strong economic growth.
   """

   claims = extract_claims(text, client)
   claims = classify_all_evidence(claims, text, client)

   for claim in claims:
       print(f"Claim: {claim.text}")
       print(f"Claimer: {claim.claimer} ({claim.claimer_role.value})")
       for ev in claim.evidence:
           print(f"  Evidence: {ev.evidence_type.value}/{ev.evidence_subtype.value}")

Understanding Claimer Roles
---------------------------

The pipeline distinguishes between:

- **anchor/reporter**: News organization voice. We ask HOW they know.
- **guest/quoted_source**: Attributed claims. The evidence IS who said it.

Source Types for Attributed Claims
----------------------------------

For attributed claims (guest/quoted_source), sources are classified:

Elite/Institutional:
   - government_elected, government_appointed, government_agency, judicial
   - business_executive, business_analyst

Expert:
   - academic, think_tank, professional_expert

Civil Society:
   - ngo_advocacy, union_labor, religious_leader

Media:
   - journalist_other, media_personality

Ordinary:
   - eyewitness, affected_person, ordinary_citizen

Anonymous:
   - anonymous_official, anonymous_insider, anonymous_other

Generating Reports
------------------

Generate an HTML report:

.. code-block:: python

   from sourcerer.analyze import compute_metrics
   from sourcerer.report import generate_html_report
   from pathlib import Path

   metrics = compute_metrics(articles)
   html = generate_html_report(articles, metrics, Path("report.html"))

Export claims to CSV:

.. code-block:: python

   from sourcerer.report import export_claims_csv

   df = export_claims_csv(articles, Path("claims.csv"))
