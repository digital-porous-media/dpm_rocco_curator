import warnings
warnings.filterwarnings("ignore", message=r"Accessing `__path__`", module="transformers")

import streamlit as st
from src.assistant.assistant_ui import render_assistant_tab

st.set_page_config(page_title="Rocco Assistant – Dev", layout="wide")
st.title("General Assistant (dev)")

# ---------------------------------------------------------------------------
# Dev sidebar: LiteratureSearch direct test
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Dev Tools")
    with st.expander("Test LiteratureSearch", expanded=False):
        lit_query = st.text_input("Query", value="pore-scale imaging multiphase flow", key="lit_query")
        lit_n = st.number_input("Max results", min_value=1, max_value=20, value=5, key="lit_n")
        if st.button("Search Semantic Scholar", key="lit_search"):
            from src.assistant.literature_search import LiteratureSearch
            with st.spinner("Querying Semantic Scholar..."):
                results = LiteratureSearch().search_external_literature(lit_query, max_results=int(lit_n))
            if not results:
                st.warning("No results returned.")
            for p in results:
                st.markdown(f"**{p.title}** ({p.year})")
                st.caption(f"Authors: {', '.join(p.authors)} · Citations: {p.citation_count} · DOI: {p.doi}")
                if p.abstract:
                    st.markdown(p.abstract[:400] + ("…" if len(p.abstract or "") > 400 else ""))
                if p.pdf_url:
                    st.markdown(f"[PDF]({p.pdf_url})")
                st.divider()

render_assistant_tab()
