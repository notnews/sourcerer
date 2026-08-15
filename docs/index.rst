Sourcerer
=========

Analyze epistemological support for claims in news articles.

Sourcerer extracts claims from news transcripts and articles, then classifies
the evidence supporting each claim. It distinguishes between:

- **Attributed claims**: Quotes from sources (guests, quoted sources) where
  the evidence is WHO said it
- **News org assertions**: Claims made by anchors/reporters where we ask
  HOW they know

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api

Installation
------------

.. code-block:: bash

   pip install sourcerer

Or install from source:

.. code-block:: bash

   git clone https://github.com/notnews/sourcerer.git
   cd sourcerer
   pip install -e .

Quick Start
-----------

.. code-block:: python

   from sourcerer.extract import extract_claims, get_client
   from sourcerer.evidence import classify_all_evidence

   client = get_client()
   text = '''
   COOPER: The CDC reported 500 new cases today.
   DR. FAUCI: We are seeing a significant uptick.
   '''
   claims = extract_claims(text, client)
   claims = classify_all_evidence(claims, text, client)

   for claim in claims:
       print(f"{claim.claimer}: {claim.text}")
       for ev in claim.evidence:
           print(f"  Evidence: {ev.evidence_type.value}")

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
