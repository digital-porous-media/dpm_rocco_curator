# Configuration file for the Sphinx documentation builder.

project = "Rocco"
copyright = "2024, DPM Rocco Contributors"
author = "DPM Rocco Contributors"
release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/USER/dpm-rocco-curator",
    "twitter_url": "https://twitter.com/",
    "footer_items": ["copyright", "last-updated"],
}

# MyST parser config
myst_enable_extensions = ["colon_fence"]

# Autodoc config
autodoc_typehints = "description"
autodoc_member_order = "bysource"
