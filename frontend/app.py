"""
Streamlit frontend for DocuVision AI
Multi-document RAG with citations, search, and visualizations.
"""

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="DocuVision AI - Document Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Styles
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            --dark-bg: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.1);
            --text-primary: #ffffff;
            --text-secondary: rgba(255, 255, 255, 0.8);
            --accent-color: #00f2fe;
            --success-color: #00ff88;
        }
        .main .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1300px; }
        #MainMenu, footer, header {visibility: hidden;}
        .main-header { text-align: center; padding: 2rem 0; margin-bottom: 2rem;}
        .main-title {
            font-size: 3.5rem; font-weight: 800;
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
            background-size: 300% 300%;
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            animation: gradientShift 8s ease-in-out infinite;
        }
        @keyframes gradientShift { 0%,100% {background-position:0% 50%;} 50% {background-position:100% 50%;}}
        .subtitle { font-size: 1.2rem; color: var(--text-secondary); margin-top: 0.5rem;}
        .description { color: rgba(255,255,255,0.65); max-width: 860px; margin: 0 auto; }
        .stButton > button {
            background: var(--accent-gradient); color: #000; border: none; border-radius: 12px;
            padding: 0.75rem 1.5rem; font-weight: 700; width: 100%;
        }
        .stButton > button:hover { background: var(--primary-gradient); transform: translateY(-1px); }
        .answer-box, .stCard {
            background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px; padding: 1.5rem; box-shadow: 0 20px 60px rgba(79,172,254,0.2);
        }
        .success-box {
            background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(79,172,254,0.1));
            border: 2px solid #00ff88; border-radius: 12px; padding: 1rem; margin: 0.5rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session state defaults
defaults = {
    "documents": [],
    "selected_docs": [],
    "answer": None,
    "sources": [],
    "matches": [],
    "keyword_stats": [],
    "similarity": {"labels": [], "matrix": []},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# Helpers
def check_api_health():
    try:
        resp = requests.get(f"{API_BASE_URL}/", timeout=3)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def refresh_documents():
    try:
        resp = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        resp.raise_for_status()
        st.session_state.documents = resp.json()
    except requests.RequestException as e:
        st.error(f"Could not load documents: {e}")


def upload_files(files):
    try:
        form = [("files", (f.name, f.getvalue(), f.type)) for f in files]
        resp = requests.post(f"{API_BASE_URL}/upload", files=form, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if data.get("success"):
            refresh_documents()
            st.success(f"Uploaded {len(files)} file(s) successfully.")
        else:
            st.error("Upload failed.")
    except requests.RequestException as e:
        st.error(f"Upload error: {e}")


def run_query(question, doc_ids, top_k=8):
    payload = {"question": question, "doc_ids": doc_ids, "top_k": top_k}
    resp = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()


def run_search(query, doc_ids, top_k=12):
    payload = {"query": query, "doc_ids": doc_ids, "top_k": top_k}
    resp = requests.post(f"{API_BASE_URL}/search", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_keyword_stats(doc_ids, top_n=12):
    payload = {"doc_ids": doc_ids, "top_n": top_n}
    resp = requests.post(f"{API_BASE_URL}/stats/keywords", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_similarity(doc_ids):
    payload = {"doc_ids": doc_ids}
    resp = requests.post(f"{API_BASE_URL}/stats/similarity", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


# Header
st.markdown(
    """
    <div class="main-header">
        <h1 class="main-title">🤖 DocuVision AI</h1>
        <p class="subtitle">Multi-document analysis, search, and visual insights</p>
        <p class="description">Upload multiple documents, ask questions with citations, search by keyword, and view document analytics.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Health check
if not check_api_health():
    st.error("⚠️ API server is not running. Start the backend: `uvicorn backend.main:app --reload`")
    st.stop()

# Load documents initially
if not st.session_state.documents:
    refresh_documents()

# Layout
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📤 Upload Documents")
    uploads = st.file_uploader(
        "Select one or more documents (PDF, DOC, DOCX, TXT)",
        type=["pdf", "txt", "doc", "docx"],
        accept_multiple_files=True,
    )
    if st.button("Upload & Process", use_container_width=True):
        if uploads:
            with st.spinner("Uploading and processing..."):
                upload_files(uploads)
        else:
            st.warning("Please choose at least one file.")

    st.subheader("📚 Document Library")
    if st.session_state.documents:
        docs_df = pd.DataFrame(st.session_state.documents)
        st.dataframe(
            docs_df[["doc_name", "pages", "chunks", "extension"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No documents uploaded yet.")

with col_right:
    st.subheader("💬 Ask a Question")
    selected = st.multiselect(
        "Choose documents to target (optional, defaults to all)",
        options=[d["doc_id"] for d in st.session_state.documents],
        format_func=lambda x: next(
            (d["doc_name"] for d in st.session_state.documents if d["doc_id"] == x), x
        ),
        default=st.session_state.selected_docs,
    )
    st.session_state.selected_docs = selected
    question = st.text_input("Question", placeholder="What do the documents say about ...?")
    top_k = st.slider("Top chunks to retrieve", 4, 15, 8)
    if st.button("🔍 Run Q&A", use_container_width=True):
        if not question:
            st.warning("Enter a question first.")
        else:
            with st.spinner("Querying with RAG..."):
                try:
                    result = run_query(question, selected, top_k)
                    st.session_state.answer = result.get("answer")
                    st.session_state.sources = result.get("sources", [])
                except requests.RequestException as e:
                    st.error(f"Query error: {e}")

# Results
if st.session_state.answer:
    st.markdown("### 🤖 Answer")
    st.markdown(
        f"""
        <div class="answer-box">
            <div class="answer-content">{st.session_state.answer}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.sources:
        st.markdown("**Sources**")
        for src in st.session_state.sources:
            st.markdown(
                f"- [{src.get('id')}] {src.get('doc_name')} — p.{src.get('page')} (chunk {src.get('chunk')})"
            )

# Search
st.markdown("---")
st.subheader("🔎 Keyword / Section Search")
search_col1, search_col2 = st.columns([3, 1])
with search_col1:
    search_query = st.text_input("Search terms", placeholder="Enter keyword or phrase...")
with search_col2:
    top_k_search = st.slider("Results", 3, 20, 8)
if st.button("Search Chunks", use_container_width=True):
    if not search_query:
        st.warning("Enter search terms.")
    else:
        with st.spinner("Searching..."):
            try:
                result = run_search(search_query, st.session_state.selected_docs, top_k_search)
                st.session_state.matches = result.get("matches", [])
            except requests.RequestException as e:
                st.error(f"Search error: {e}")

if st.session_state.matches:
    st.markdown("##### Matches")
    for m in st.session_state.matches:
        st.markdown(
            f"- **{m.get('doc_name')}** p.{m.get('page')} (chunk {m.get('chunk')}): {m.get('snippet')}"
        )

# Visualizations
st.markdown("---")
viz_col1, viz_col2 = st.columns([1, 1])

with viz_col1:
    st.subheader("📊 Keyword Frequency")
    if st.button("Refresh Keywords", use_container_width=True):
        with st.spinner("Computing keyword frequencies..."):
            try:
                stats = fetch_keyword_stats(st.session_state.selected_docs, top_n=12)
                st.session_state.keyword_stats = stats.get("keywords", [])
            except requests.RequestException as e:
                st.error(f"Keyword stats error: {e}")
    if st.session_state.keyword_stats:
        df = pd.DataFrame(st.session_state.keyword_stats)
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Count"),
                y=alt.Y("term:N", sort="-x", title="Keyword"),
                color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["term", "count"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Click 'Refresh Keywords' to see keyword frequencies.")

with viz_col2:
    st.subheader("🔥 Similarity Heatmap")
    if st.button("Refresh Similarity", use_container_width=True):
        with st.spinner("Computing document similarities..."):
            try:
                sim = fetch_similarity(st.session_state.selected_docs)
                st.session_state.similarity = sim
            except requests.RequestException as e:
                st.error(f"Similarity error: {e}")
    labels = st.session_state.similarity.get("labels", [])
    matrix = st.session_state.similarity.get("matrix", [])
    if labels and matrix:
        df = pd.DataFrame(matrix, columns=labels, index=labels)
        heat_df = (
            df.reset_index()
            .melt(id_vars="index", var_name="doc_b", value_name="similarity")
            .rename(columns={"index": "doc_a"})
        )
        heatmap = (
            alt.Chart(heat_df)
            .mark_rect()
            .encode(
                x=alt.X("doc_a:N", title="Document A"),
                y=alt.Y("doc_b:N", title="Document B"),
                color=alt.Color("similarity:Q", scale=alt.Scale(scheme="tealblues")),
                tooltip=["doc_a", "doc_b", alt.Tooltip("similarity:Q", format=".2f")],
            )
            .properties(height=320)
        )
        st.altair_chart(heatmap, use_container_width=True)
    else:
        st.info("Click 'Refresh Similarity' to see document similarities.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: rgba(255, 255, 255, 0.6); padding: 1rem 0;">
        <p>🔒 Uses only your uploaded documents. Citations included for transparency.</p>
        <p style="margin-top: 0.5rem;">© 2025 DocuVision AI. FastAPI + Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True,
)

