import streamlit as st
from src.assistant.assistant_ui import render_assistant_tab

st.set_page_config(page_title="Rocco Assistant – Dev", layout="wide")
st.title("General Assistant (dev)")
render_assistant_tab()
