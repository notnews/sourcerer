Installation
============

Requirements
------------

- Python 3.11 or higher
- An Anthropic API key

Install from PyPI
-----------------

.. code-block:: bash

   pip install sourcerer

Install from Source
-------------------

.. code-block:: bash

   git clone https://github.com/notnews/sourcerer.git
   cd sourcerer
   pip install -e .

Development Installation
------------------------

To install with development dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

This includes:

- pytest for testing
- ruff for linting
- vulture for dead code detection
- pydoclint for docstring linting

Configuration
-------------

Set your Anthropic API key as an environment variable:

.. code-block:: bash

   export ANTHROPIC_API_KEY="your-api-key-here"
