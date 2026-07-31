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
    page_title="potbot",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)
from domain.models import FeedbackRecord
from ingestion.loaders import CompositeDocumentLoader
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
# Theme: Background (#0F1117), Sidebar (#14161D), Cards (#181B23), Borders (#262A35), 
# White text (#F5F5F5), Muted (#A1A1AA), Orange Accent (#F97316)
st.markdown("""
<style>
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp > header {background-color: transparent;}
    
    /* Global Theme & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #F5F5F5;
        background-color: #0F1117;
    }
    
    /* Backgrounds & Sidebar */
    .stApp {
        background-color: #0F1117;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #14161D;
        border-right: 1px solid #262A35;
        width: 300px !important;
    }
    
    /* Constrain main block width & center it */
    .main .block-container {
        max-width: 840px !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-top: 0.5rem !important;
        padding-bottom: 7rem !important;
        margin: 0 auto !important;
    }

    /* Cards */
    .sidebar-card {
        background-color: #181B23;
        border: 1px solid #262A35;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    
    /* Compact Header */
    .app-header {
        display: flex;
        align-items: center;
        gap: 12px;
        border-bottom: 1px solid #262A35;
        padding-bottom: 12px;
        margin-bottom: 16px;
    }
    .app-header-logo {
        background: #F97316;
        color: #181B23;
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 18px;
        font-weight: 600;
    }
    .app-header-title {
        font-size: 20px;
        font-weight: 600;
        margin: 0;
        color: #F5F5F5;
        line-height: 1.2;
    }
    .app-header-subtitle {
        font-size: 13px;
        color: #A1A1AA;
        margin: 0;
        line-height: 1.2;
    }
    
    /* Empty State style */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 20px 20px 32px;
        margin-top: 0;
        margin-bottom: 20px;
    }
    .empty-state-icon {
        font-size: 32px;
        color: #F97316;
        margin-bottom: 20px;
        background: rgba(249, 115, 22, 0.1);
        width: 60px;
        height: 60px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .empty-state-title {
        font-size: 22px;
        font-weight: 600;
        color: #F5F5F5;
        margin-bottom: 10px;
    }
    .empty-state-subtitle {
        font-size: 14px;
        color: #A1A1AA;
        max-width: 480px;
        line-height: 1.6;
        margin-bottom: 24px;
    }
    
    /* Target horizontal block after the suggestion marker */
    div:has(.suggestion-marker) + div div[data-testid="stHorizontalBlock"] button {
        background-color: #181B23 !important;
        border: 1px solid #262A35 !important;
        color: #F5F5F5 !important;
        border-radius: 8px !important;
        padding: 12px 14px !important;
        font-weight: 400 !important;
        text-align: left !important;
        font-size: 13px !important;
        transition: all 0.2s ease !important;
        height: auto !important;
        min-height: 64px !important;
        display: flex !important;
        align-items: center !important;
        box-shadow: none !important;
    }
    div:has(.suggestion-marker) + div div[data-testid="stHorizontalBlock"] button:hover {
        border-color: #F97316 !important;
        background-color: #1c202a !important;
        color: #F5F5F5 !important;
    }
    
    /* Target horizontal block after the feedback marker */
    div:has(.feedback-marker) + div div[data-testid="stHorizontalBlock"] button {
        background-color: #181B23 !important;
        border: 1px solid #262A35 !important;
        color: #A1A1AA !important;
        border-radius: 6px !important;
        padding: 4px 8px !important;
        font-size: 12px !important;
        min-height: unset !important;
        height: 32px !important;
        width: 32px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: none !important;
    }
    div:has(.feedback-marker) + div div[data-testid="stHorizontalBlock"] button:hover {
        border-color: #F97316 !important;
        color: #F97316 !important;
        background-color: rgba(249, 115, 22, 0.05) !important;
    }
    
    /* Source Cards inside Chat */
    .source-card {
        background-color: #181B23;
        border: 1px solid #262A35;
        border-radius: 6px;
        padding: 12px;
        margin-top: 8px;
        font-size: 0.85rem;
        color: #A1A1AA;
    }
    .source-card .file-name {
        color: #F97316;
        font-weight: 500;
        display: block;
        margin-bottom: 4px;
    }
    
    /* Premium Upload Area */
    [data-testid="stFileUploadDropzone"] {
        background-color: #181B23 !important;
        border: 1px dashed #262A35 !important;
        border-radius: 8px !important;
        padding: 24px 16px !important;
        transition: border-color 0.2s, background-color 0.2s;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #F97316 !important;
        background-color: #1c202a !important;
    }
    [data-testid="stFileUploadDropzone"] svg {
        fill: #F97316 !important;
    }
    [data-testid="stFileUploadDropzone"] div {
        color: #A1A1AA !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #181B23;
        border: 1px solid #262A35;
        color: #F5F5F5;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #F97316;
        background-color: #1c202a;
        color: #F97316;
    }
    
    /* Inputs */
    .stTextInput > div > div > input, 
    .stSelectbox > div > div > div {
        background-color: #0F1117;
        color: #F5F5F5;
        border: 1px solid #262A35;
        border-radius: 6px;
    }
    
    /* Chat Input styling */
    [data-testid="stChatInput"] {
        background-color: #181B23 !important;
        border: 1px solid #262A35 !important;
        border-radius: 8px !important;
        padding: 8px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #F97316 !important;
        box-shadow: 0 0 0 1px #F97316 !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #F5F5F5 !important;
        font-size: 15px !important;
        line-height: 1.5 !important;
    }

    /* Chat Messages styling with smooth fade-in */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stChatMessage {
        background-color: transparent;
        padding: 1rem 0;
        animation: fadeIn 0.3s ease-out forwards;
    }
    [data-testid="chatAvatarIcon-user"] {
        background-color: #262A35;
    }
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #F97316;
    }
    
    /* Sidebar stats layout */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        background-color: #181B23;
        border: 1px solid #262A35;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 20px;
    }
    .stat-item {
        text-align: center;
    }
    .stat-val {
        font-size: 15px;
        font-weight: 600;
        color: #F5F5F5;
        line-height: 1.2;
    }
    .stat-lbl {
        font-size: 10px;
        color: #A1A1AA;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        display: inline-block;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        border-bottom: 1px solid #262A35;
        padding: 0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 36px;
        white-space: nowrap;
        background-color: transparent;
        border: none;
        color: #A1A1AA;
        font-size: 13px;
        font-weight: 500;
        padding: 0 4px;
        transition: color 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #F5F5F5;
    }
    .stTabs [aria-selected="true"] {
        color: #F97316 !important;
        border-bottom: 2px solid #F97316 !important;
    }
    
    /* Expander overrides */
    .stExpander {
        background-color: #181B23 !important;
        border: 1px solid #262A35 !important;
        border-radius: 6px !important;
        margin-top: 8px !important;
    }
    .stExpander details {
        border: none !important;
        padding: 0 !important;
    }
    .stExpander details summary {
        padding: 8px 12px !important;
        color: #A1A1AA !important;
        font-size: 13px !important;
    }
    .stExpander details summary:hover {
        color: #F5F5F5 !important;
    }
    .stExpander details [data-testid="stExpanderDetails"] {
        padding: 0 12px 12px 12px !important;
        border-top: 1px solid #262A35 !important;
        background-color: #14161D !important;
    }
    
    hr {
        border-top: 1px solid #262A35;
    }
</style>
""", unsafe_allow_html=True)


# --- State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# --- Sidebar ---
with st.sidebar:
    st.markdown("""
        <div style='margin-bottom: 24px;'>
            <div style='font-size: 18px; font-weight: 600; color: #F5F5F5;'>potbot workspace</div>
            <div style='font-size: 12px; color: #A1A1AA;'>Local environment</div>
        </div>
    """, unsafe_allow_html=True)

    # Document Ingestion Section
    st.markdown("<div style='font-size: 11px; font-weight: 600; color: #A1A1AA; margin-bottom: 8px; text-transform: uppercase;'>Knowledge Base</div>", unsafe_allow_html=True)
    
    tab_files, tab_folder = st.tabs(["Upload", "Folder"])

    with tab_files:
        supported_types = CompositeDocumentLoader.get_supported_extensions_without_dot()
        uploaded_files = st.file_uploader(
            "Drop documents here",
            type=supported_types,
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        recreate_idx_files = st.checkbox("Rebuild index", value=False, key="recreate_files")
        if st.button("Ingest Files", use_container_width=True, key="btn_ingest_files"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    try:
                        pipeline = IngestionPipeline()
                        result = pipeline.run_uploaded_files(
                            uploaded_files=uploaded_files,
                            recreate_index=recreate_idx_files
                        )
                        st.success(f"Indexed {result.get('indexed_count', 0)} chunks")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please select a file.")

    with tab_folder:
        raw_folder_path = st.text_input(
            "Local folder path",
            placeholder=r"C:\documents",
            label_visibility="collapsed"
        )
        recreate_idx_folder = st.checkbox("Rebuild index", value=False, key="recreate_folder")
        if st.button("Ingest Folder", use_container_width=True, key="btn_ingest_folder"):
            clean_path = raw_folder_path.strip(" '\"") if raw_folder_path else ""
            if clean_path:
                clean_path = os.path.normpath(clean_path)
                if os.path.isdir(clean_path):
                    with st.spinner("Processing..."):
                        try:
                            pipeline = IngestionPipeline()
                            result = pipeline.run(folder_path=clean_path, recreate_index=recreate_idx_folder)
                            st.success(f"Indexed {result.get('indexed_count', 0)} chunks")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Invalid folder path.")

    st.write("") # Spacer

    # Knowledge Base Stats
    try:
        from ingestion.indexers import ElasticsearchVectorStore
        es_store = ElasticsearchVectorStore()
        es_client = es_store.es
        stats = es_store.get_stats()
        chunk_count = stats.get("doc_count", 0) if stats.get("exists") else 0
        
        unique_file_count = 0
        if stats.get("exists") and es_client:
            try:
                body = {
                    "size": 0,
                    "aggs": {
                        "unique_files": {
                            "cardinality": {
                                "field": "file_name"
                            }
                        }
                    }
                }
                res = es_client.search(index=es_store.index_name, body=body)
                unique_file_count = res["aggregations"]["unique_files"]["value"]
            except Exception:
                unique_file_count = 0
                
        if es_client and es_client.ping():
            status_text = "Online"
            status_color = "#10B981"
        else:
            status_text = "Offline"
            status_color = "#EF4444"
            
        st.markdown(f"""
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-val">{unique_file_count}</div>
                    <div class="stat-lbl">Documents</div>
                </div>
                <div class="stat-item">
                    <div class="stat-val">{chunk_count}</div>
                    <div class="stat-lbl">Chunks</div>
                </div>
                <div class="stat-item">
                    <div class="stat-val" style="color: {status_color}; display: flex; align-items: center; justify-content: center; gap: 6px;">
                        <span class="status-dot" style="background-color: {status_color};"></span>
                        {status_text}
                    </div>
                    <div class="stat-lbl">Status</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.markdown("""
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-val">0</div>
                    <div class="stat-lbl">Documents</div>
                </div>
                <div class="stat-item">
                    <div class="stat-val">0</div>
                    <div class="stat-lbl">Chunks</div>
                </div>
                <div class="stat-item">
                    <div class="stat-val" style="color: #EF4444; display: flex; align-items: center; justify-content: center; gap: 6px;">
                        <span class="status-dot" style="background-color: #EF4444;"></span>
                        Offline
                    </div>
                    <div class="stat-lbl">Status</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.write("") # Spacer

    # Pipeline Settings
    st.markdown("<div style='font-size: 11px; font-weight: 600; color: #A1A1AA; margin-bottom: 8px; margin-top: 16px; text-transform: uppercase;'>Settings</div>", unsafe_allow_html=True)
    retrieval_method = st.selectbox("Search Strategy", ["hybrid", "vector", "text"], index=0)
    
    c1, c2 = st.columns(2)
    with c1:
        use_reranking = st.checkbox("Rerank", value=True)
    with c2:
        use_query_rewrite = st.checkbox("Rewrite", value=True)
        
    prompt_style = st.selectbox("Output Style", ["detailed", "concise", "structured"], index=0)


# --- Main Header ---
st.markdown("""
<div class="app-header">
    <div class="app-header-logo">pb</div>
    <div>
        <h1 class="app-header-title">potbot</h1>
        <p class="app-header-subtitle">Internal Intelligence Platform</p>
    </div>
</div>
""", unsafe_allow_html=True)


# --- Render Empty State or Conversation History ---
if not st.session_state.messages:
    st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🔒</div>
            <h2 class="empty-state-title">Secure Enterprise RAG</h2>
            <p class="empty-state-subtitle">Ask questions across your internal documents. Fully private embeddings and execution ensure no data leaks.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='suggestion-marker'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 11px; color: #A1A1AA; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 12px;'>Suggested Prompts</p>", unsafe_allow_html=True)
    
    suggestions = [
        "Summarize the recent engineering guidelines.",
        "How do I setup a local database model?",
        "What are the main security protocols for internal documents?"
    ]
    
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    for i, suggestion in enumerate(suggestions):
        with cols[i]:
            if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                st.session_state.pending_prompt = suggestion
                st.rerun()
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<div class="source-card">'
                            f'<span class="file-name">{src["file_name"]}</span>'
                            f'{"Page " + str(src["page_number"]) + " • " if src.get("page_number") else ""}'
                            f'{src["text"][:200]}...'
                            f"</div>",
                            unsafe_allow_html=True,
                        )


# --- Chat Input & Execution ---
active_prompt = None

if st.session_state.pending_prompt:
    active_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

chat_input_val = st.chat_input("Message potbot...")
if chat_input_val:
    active_prompt = chat_input_val

if active_prompt:
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user"):
        st.markdown(active_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating..."):
            try:
                response = rag_pipeline.query(
                    user_query=active_prompt,
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
                    with st.expander("Sources", expanded=False):
                        for src in sources:
                            st.markdown(
                                f'<div class="source-card">'
                                f'<span class="file-name">{src["file_name"]}</span>'
                                f'{"Page " + str(src["page_number"]) + " • " if src.get("page_number") else ""}'
                                f'{src["text"][:200]}...'
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": sources,
                    "conversation_id": response.conversation_id,
                })

                st.markdown(
                    f"<div style='font-size: 12px; color: #A1A1AA; margin-top: 8px;'>"
                    f"Generated in {response.response_time_ms}ms • {response.total_tokens} tokens"
                    f"</div>",
                    unsafe_allow_html=True
                )

                # Feedback widget
                if response.conversation_id:
                    conv_id = response.conversation_id
                    st.markdown("<div class='feedback-marker'></div>", unsafe_allow_html=True)
                    c1, c2, _ = st.columns([1, 1, 10])
                    with c1:
                        if st.button("👍", key=f"up_{conv_id}"):
                            db_repo.save_feedback(FeedbackRecord(conversation_id=conv_id, sentiment="positive"))
                            st.toast("Feedback recorded: 👍")
                    with c2:
                        if st.button("👎", key=f"down_{conv_id}"):
                            db_repo.save_feedback(FeedbackRecord(conversation_id=conv_id, sentiment="negative"))
                            st.toast("Feedback recorded: 👎")

            except Exception as e:
                err_text = f"Error: {str(e)}"
                st.error(err_text)
                st.session_state.messages.append({"role": "assistant", "content": err_text})
                
    st.rerun()
