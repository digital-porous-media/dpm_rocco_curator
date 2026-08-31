Preface
=======

Rocco is a domain-agnostic framework for curating, discovering, and explaining
research datasets. It combines rubric-based evaluation, retrieval-augmented
generation, and a graph-backed search layer behind a single web interface.

This manual collects the complete Rocco documentation into one volume: the user
guides for both modules, the developer guide, the operational runbooks, and the
generated code reference.

Who this manual is for
----------------------

Each part targets a different reader. You do not need to read the manual in
order.

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - If you are
     - Start at
     - Because
   * - A researcher curating a dataset description
     - Part I, then Part II
     - Part II covers the scoring rubric, document upload, and the enhancement
       loop.
   * - A researcher searching the catalog
     - Part I, then Part III
     - Part III covers every question type the assistant answers.
   * - An administrator deploying Rocco
     - Part IV, then Part VI
     - Part IV lists every setting; Part VI covers running Rocco as a service.
   * - A developer joining the project
     - Part V, then Appendix A
     - Part V covers setup and architecture; Appendix A is the class and
       function reference.

How this manual is organized
----------------------------

Parts I through III are task-oriented: they describe what Rocco does and how to
drive it. Part IV is a settings reference. Part V explains how the system is
built and how to change it. Part VI covers deployment, published benchmarks, and
release history. The appendices hold generated reference material: the Python
API, the Neo4j graph schema, the prompt files, the acceptance query suite, and
the license.

Some chapters in Part III describe internal behavior in more detail than a
typical user guide, naming the specific functions that make routing decisions.
That detail is deliberate. Rocco's answers depend on which tool handles a
question, and the chapters explain how a question is classified so you can tell
why an answer took the shape it did.

Conventions
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Convention
     - Meaning
   * - ``monospace``
     - A filename, command, environment variable, or code identifier.
   * - ``$ command``
     - A command you run in a shell. Do not type the ``$``.
   * - *Italic*
     - A term defined for the first time.
   * - :doc:`Cross-reference </front/preface>`
     - A link to another chapter. In the PDF, the page number follows in the
       table of contents.

File paths are relative to the repository root unless stated otherwise. Sample
API keys such as ``sk-proj-your-key-here`` are placeholders; replace them with
your own credentials, and keep those credentials out of version control.

Where to get updates
--------------------

This manual describes version 1.0.0. The documentation is maintained alongside
the source code, so the online version may be newer:

* Documentation: https://digital-porous-media.github.io/dpm_rocco_curator/
* Source code: https://github.com/digital-porous-media/dpm_rocco_curator
* Issue tracker: https://github.com/digital-porous-media/dpm_rocco_curator/issues
* Archived release: https://doi.org/10.5281/zenodo.20172375

To cite Rocco, see the citation entry in :doc:`/overview`.
