# Configuration file for the Sphinx documentation builder.
import sys
from pathlib import Path

# Add project root to Python path for autodoc to import src as a module
sys.path.insert(0, str(Path(__file__).parent.parent))

project = "Rocco"
copyright = "2024, DPM Rocco Contributors"
author = "DPM Rocco Contributors"
release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_tabs.tabs",
    "sphinx.ext.graphviz",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_title = "Rocco"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#1a6b8a",
        "color-brand-content": "#1a6b8a",
        "color-highlighted-background": "#e8f4f8",
        "color-admonition-background": "rgba(26, 107, 138, 0.06)",
        "font-stack": "Inter, system-ui, -apple-system, sans-serif",
        "font-stack--monospace": "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5ec4df",
        "color-brand-content": "#5ec4df",
        "color-highlighted-background": "#1d3d4a",
        "color-admonition-background": "rgba(94, 196, 223, 0.08)",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/digital-porous-media/dpm_rocco_curator/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/digital-porous-media/dpm_rocco_curator",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                    0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01
                    1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95
                    0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27
                    2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82
                    1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2
                    0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
        {
            "name": "GitHub Issues",
            "url": "https://github.com/digital-porous-media/dpm_rocco_curator/issues",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path d="M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13zM0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8zm9 3a1 1 0 1 1-2 0 1 1 0 0 1 2 0zm-.25-6.25a.75.75 0 0 0-1.5 0v3.5a.75.75 0 0 0 1.5 0v-3.5z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# MyST parser config
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]

# Autodoc config
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# copybutton config
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
