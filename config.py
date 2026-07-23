"""
potbot - Centralized configuration.

Loads settings from environment variables (.env file) with sensible defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# --- Groq LLM ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# --- Embeddings (local, sentence-transformers) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Elasticsearch ---
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
ELASTICSEARCH_INDEX = os.getenv("ELASTICSEARCH_INDEX", "potbot_documents")

# --- PostgreSQL ---
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "potbot")
POSTGRES_USER = os.getenv("POSTGRES_USER", "potbot")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "potbot_secret")

DATABASE_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# --- Retrieval ---
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))
