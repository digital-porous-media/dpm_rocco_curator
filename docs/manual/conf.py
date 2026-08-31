"""Sphinx configuration for the printable single-PDF manual.

This config is used only by ``scripts/build_manual.py``. It layers print-specific
overrides on top of the site config in ``docs/conf.py`` so the two builds cannot
drift: everything not overridden here is inherited.

The build runs against a *staged copy* of the documentation (see
``scripts/build_manual.py``), never against ``docs/`` itself.
"""

from pathlib import Path

# ``docs/manual/conf.py`` is copied into the staged tree as ``conf.py``, so the
# repo root is resolved from an env var the build script sets rather than from
# ``__file__``, which points into the staging directory at build time.
import os
import sys

REPO_ROOT = Path(os.environ["ROCCO_MANUAL_REPO_ROOT"]).resolve()

# Inherit the site configuration verbatim.
_base = (REPO_ROOT / "docs" / "conf.py").read_text(encoding="utf-8")
exec(compile(_base, str(REPO_ROOT / "docs" / "conf.py"), "exec"))  # noqa: S102

# autodoc imports ``src.*`` and the ``scripts/*`` modules by name. The scripts
# directory is not a package, so it goes on the path directly; every script is
# guarded by ``if __name__ == "__main__"`` and only calls ``load_dotenv()`` at
# module level, so importing them is side-effect free.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# -- Print-specific overrides ------------------------------------------------

master_doc = "index"

# Furo is a web theme: its sidebar, search box, and theme-toggle JavaScript are
# dead weight in a PDF. "basic" emits clean semantic HTML for WeasyPrint to style.
html_theme = "basic"
html_theme_options = {}
html_sidebars = {"**": []}
html_copy_source = False
html_show_sourcelink = False
html_use_index = False
html_static_path = ["_static"]
html_css_files = []  # print.css is applied by WeasyPrint, not linked from the HTML

# viewcode appends a full syntax-highlighted source listing for every documented
# module. In a single document that roughly doubles the page count, duplicates
# Appendix A, and embeds build-machine paths in the output.
extensions = [e for e in extensions if e != "sphinx.ext.viewcode"]  # noqa: F821

# Graphviz renders to SVG by default. WeasyPrint's SVG support does not handle
# Graphviz's HTML-table node labels, which every diagram in architecture.rst
# uses, so they print blank. PNG at 200 dpi renders correctly and stays sharp.
graphviz_output_format = "png"
graphviz_dot_args = ["-Gdpi=200", "-Gbgcolor=transparent"]

# Number parts, chapters, and sections consistently across the body text, the
# generated table of contents, and the PDF bookmark tree.
numfig = True

# Pre-existing warnings in the source docs, suppressed so the manual can build
# under -W (a dead cross-reference in a printed document cannot be fixed by the
# reader, so new warnings must fail the build).
#   - autodoc.import_object / ref.python: the graph_store dataclass attributes are
#     documented both by automodule and by their own attribute directives.
#   - misc.highlighting_failure: neo4j_schema.md:242 contains a Cypher snippet with
#     a ``$embedding`` parameter that Pygments' Cypher lexer rejects.
suppress_warnings = ["ref.python", "misc.highlighting_failure", "autodoc"]

# napoleon turns a Google-style "Attributes:" block into standalone
# ``.. attribute::`` directives, which duplicate the entries autodoc already
# emits for dataclass fields (DatasetProfileMatch). Rendering them as :ivar:
# fields on the class removes the collision.
napoleon_use_ivar = True

# Autodoc: public members only for the appendix.
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

# PDF-facing metadata. Set explicitly so nothing is inherited from the build
# machine (usernames, home directories, absolute paths).
project = "Rocco"
html_title = "Rocco Reference Manual"
