"""
potbot — Streamlit Application

Clean UI integrating with RAGPipeline and IngestionPipeline domain services.
"""

import os
import sys

# Ensure root is in PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import config
st.set_page_config(
    page_title="potbot — Internal Document Intelligence",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)
from domain.models import FeedbackRecord
from ingestion.pipeline import IngestionPipeline
from rag.pipeline import RAGPipeline
from app.database import PostgresDatabaseRepository

# Initialize services
@st.cache_resource
def get_db_repository():
    repo = PostgresDatabaseRepository()
    try:
        repo.init_db()
    except Exception as e:
        print(f"DB init deferred: {e}")
    return repo

@st.cache_resource
def get_rag_pipeline():
    return RAGPipeline(repository=get_db_repository())

db_repo = get_db_repository()
rag_pipeline = get_rag_pipeline()






# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        text-align: center;
        color: #a0a0b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .source-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 0.85rem;
    }

    .source-card .file-name {
        color: #667eea;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# --- State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🔒 potbot")
    st.markdown("---")

    # Document Ingestion
    st.markdown("### 📁 Document Ingestion")

    tab_files, tab_folder = st.tabs(["📤 Select Files", "📁 Folder Path"])

    with tab_files:
        uploaded_files = st.file_uploader(
            "Select document(s)",
            type=["pdf", "docx", "txt", "md", "csv"],
            accept_multiple_files=True,
            help="Select one or multiple documents from your computer to ingest",
        )
        recreate_idx_files = st.checkbox("Rebuild index from scratch", value=False, key="recreate_files")
        if st.button("🚀 Ingest Selected Files", use_container_width=True, key="btn_ingest_files"):
            if uploaded_files:
                with st.spinner(f"Ingesting {len(uploaded_files)} file(s)..."):
                    try:
                        pipeline = IngestionPipeline()
                        result = pipeline.run_uploaded_files(
                            uploaded_files=uploaded_files,
                            recreate_index=recreate_idx_files
                        )
                        st.success(
                            f"✅ Ingestion complete!\n\n"
                            f"- Documents loaded: **{result.get('doc_count', 0)}**\n"
                            f"- Chunks indexed: **{result.get('indexed_count', 0)}**"
                        )
                    except Exception as e:
                        st.error(f"❌ Ingestion failed: {str(e)}")
            else:
                st.warning("⚠️ Please select at least one document file to ingest.")

    with tab_folder:
        raw_folder_path = st.text_input(
            "Folder path",
            placeholder=r"e.g., D:\\agency\\books",
            help="Absolute path to local folder containing your documents",
        )
        # Normalize path: strip surrounding quotes/spaces, expand user, normalize separators
        clean_path = raw_folder_path.strip(" '\"") if raw_folder_path else ""
        if clean_path:
            # Convert to proper Windows path handling
            clean_path = os.path.normpath(clean_path)
            print(clean_path)
        recreate_idx_folder = st.checkbox("Rebuild index from scratch", value=False, key="recreate_folder")
        if st.button("🚀 Ingest Folder", use_container_width=True, key="btn_ingest_folder"):
            if clean_path and os.path.isdir(clean_path):
                with st.spinner("Ingesting documents from folder..."):
                    try:
                        pipeline = IngestionPipeline()
                        result = pipeline.run(folder_path=clean_path, recreate_index=recreate_idx_folder)
                        st.success(
                            f"✅ Ingestion complete!\n\n"
                            f"- Documents loaded: **{result.get('doc_count', 0)}**\n"
                            f"- Chunks indexed: **{result.get('indexed_count', 0)}**"
                        )
                    except Exception as e:
                        st.error(f"❌ Ingestion failed: {str(e)}")
            else:
                st.warning(f"⚠️ Folder not found or inaccessible: '{raw_folder_path}'. Try using the 'Select Files' tab to upload documents directly.")

    st.markdown("---")

    # Knowledge Base Stats
    st.markdown("### 📊 Knowledge Base")
    try:
        from ingestion.indexers import ElasticsearchVectorStore
        es_store = ElasticsearchVectorStore()
        stats = es_store.get_stats()
        if stats.get("exists"):
            st.metric("Indexed Chunks", stats.get("doc_count", 0))
        else:
            st.info("No index found. Ingest documents first.")
    except Exception:
        st.info("Connect to Elasticsearch to view stats")

    st.markdown("---")

    # Pipeline Settings
    st.markdown("### ⚙️ RAG Settings")
    retrieval_method = st.selectbox("Retrieval strategy", ["hybrid", "vector", "text"], index=0)
    use_reranking = st.checkbox("Enable re-ranking", value=True)
    use_query_rewrite = st.checkbox("Enable query rewriting", value=True)
    prompt_style = st.selectbox("Prompt style", ["detailed", "concise", "structured"], index=0)

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888; font-size: 0.8rem;'>"
        "potbot v1.0 • Local Embeddings<br>Groq LLM Engine"
        "</div>",
        unsafe_allow_html=True,
    )


# --- Main Header ---
st.markdown('<div class="main-header">🔒 potbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    "Enterprise Internal Document RAG — 100% Private Embeddings + Groq Intelligence"
    "</div>",
    unsafe_allow_html=True,
)


# --- Render Conversation History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<span class="file-name">📎 {src["file_name"]}</span>'
                        f'{" — Page " + str(src["page_number"]) if src.get("page_number") else ""}'
                        f'<br><small>{src["text"][:200]}...</small>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )


# --- Chat Input & Execution ---
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing context and generating answer..."):
            try:
                response = rag_pipeline.query(
                    user_query=prompt,
                    retrieval_method=retrieval_method,
                    use_reranking=use_reranking,
                    use_query_rewriting=use_query_rewrite,
                    prompt_style=prompt_style,
                )

                st.markdown(response.answer)

                sources = [
                    {
                        "file_name": doc.file_name,
                        "page_number": doc.page_number,
                        "text": doc.text,
                    }
                    for doc in response.retrieved_docs
                ]

                if sources:
                    with st.expander("📄 Sources", expanded=False):
                        for src in sources:
                            st.markdown(
                                f'<div class="source-card">'
                                f'<span class="file-name">📎 {src["file_name"]}</span>'
                                f'{" — Page " + str(src["page_number"]) if src.get("page_number") else ""}'
                                f'<br><small>{src["text"][:200]}...</small>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": sources,
                    "conversation_id": response.conversation_id,
                })

                st.caption(
                    f"⚡ {response.response_time_ms}ms • "
                    f"📊 {response.total_tokens} tokens • "
                    f"🔍 Strategy: {retrieval_method}"
                    f"{' + rerank' if use_reranking else ''}"
                    f"{' + rewrite' if use_query_rewrite else ''}"
                )

                # Feedback widget
                if response.conversation_id:
                    conv_id = response.conversation_id
                    c1, c2, _ = st.columns([1, 1, 6])
                    with c1:
                        if st.button("👍", key=f"up_{conv_id}"):
                            db_repo.save_feedback(FeedbackRecord(conversation_id=conv_id, sentiment="positive"))
                            st.toast("Feedback recorded: 👍")
                    with c2:
                        if st.button("👎", key=f"down_{conv_id}"):
                            db_repo.save_feedback(FeedbackRecord(conversation_id=conv_id, sentiment="negative"))
                            st.toast("Feedback recorded: 👎")

            except Exception as e:
                err_text = f"❌ Error: {str(e)}"
                st.error(err_text)
                st.session_state.messages.append({"role": "assistant", "content": err_text})
