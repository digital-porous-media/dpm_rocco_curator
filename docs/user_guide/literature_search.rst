Literature Search
===================

Find papers and publications related to a topic. This page covers the ``search_literature``
tool, backed by the Semantic Scholar API.

When to Use This
------------------

Route here for explicit requests for papers, publications, references, or citations — not
datasets ("find papers on X" vs. "find datasets about X", which is :doc:`dataset_discovery`).

What It Does
------------

``search_literature(query)`` (``src/assistant/tools.py``) calls
``LiteratureSearch.search_external_literature()`` (``src/assistant/literature_search.py``),
which queries the `Semantic Scholar Graph API <https://api.semanticscholar.org/graph/v1>`_
``/paper/search`` endpoint directly — there is no local publication index or FAISS store behind
this tool; every call is a live API request.

- **Authentication**: if ``SEMANTIC_SCHOLAR_API_KEY`` is set, requests are authenticated (higher
  rate limit); otherwise unauthenticated requests are used (shared, lower rate limit — fine for
  development). Get a key at https://www.semanticscholar.org/product/api.
- **Rate limiting**: a sleep-based throttle enforces at most 1 request/second per process,
  matching Semantic Scholar's authenticated rate limit. A 429 response triggers exponential
  backoff (5s, 10s, 20s) before giving up.
- **Fields returned**: title, authors, abstract, year, DOI, citation count, open-access PDF URL,
  and the Semantic Scholar page URL.

Results are labeled ``[semantic scholar]``. Each result includes up to 3 authors (with "et al."
for more), the DOI when available, and the abstract.

Because this is called directly as a standalone tool, its output is neither a verbatim tool nor
a self-contained tool in the conversation manager's classification (see :doc:`assistant`) — for
a single-tool turn its result is synthesized into the final response by the outer agent, which
is instructed to preserve DOIs and titles from the tool output rather than retype them from
memory.

Where Else Literature Search Is Used
---------------------------------------

The same ``LiteratureSearch`` class is also called internally, as a fallback, by:

- :doc:`domain_qa` and :doc:`workflow_guidance` — when no portal tutorial matches a question,
  a literature search result set is included as an alternative starting point for the user.

Example Queries
----------------

.. code-block:: text

   Find papers on micro-CT imaging of carbonate rocks
   Papers on water flooding in sandstone
   What research has been published on machine learning for permeability prediction?
   Find publications about pore-scale modeling

See Also
--------

- :doc:`dataset_discovery` — Finding datasets, not papers
- :doc:`domain_qa`, :doc:`workflow_guidance` — Where this tool is used as an internal fallback
- :doc:`assistant` — Overview of how all capabilities fit together
- ``src/assistant/literature_search.py`` — Full implementation
