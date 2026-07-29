"""
potbot - Centralized configuration.

Loads settings from environment variables (.env file) with sensible defaults.
All ingestion tuning knobs are prefixed with INGESTION_ and can be set per
environment (laptop vs. server) without changing any source code.
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

# --- Ingestion Pipeline Parallelism ---
# ThreadPoolExecutor workers for I/O-bound file loading
INGESTION_LOADER_WORKERS = int(os.getenv("INGESTION_LOADER_WORKERS", "4"))
# ProcessPoolExecutor workers for CPU-bound text chunking
INGESTION_CHUNKER_WORKERS = int(os.getenv("INGESTION_CHUNKER_WORKERS", "2"))
# Number of chunks per embedding batch
INGESTION_EMBED_BATCH_SIZE = int(os.getenv("INGESTION_EMBED_BATCH_SIZE", "64"))
# Number of embedded chunks per Elasticsearch bulk request
INGESTION_INDEX_BULK_SIZE = int(os.getenv("INGESTION_INDEX_BULK_SIZE", "200"))
# Max items in the inter-stage queue (backpressure). 0 = unlimited
INGESTION_QUEUE_SIZE = int(os.getenv("INGESTION_QUEUE_SIZE", "500"))

# --- Ingestion Device ---
# Embedding compute device: "auto" | "cuda" | "mps" | "cpu"
INGESTION_DEVICE = os.getenv("INGESTION_DEVICE", "auto")

# --- Incremental Ingestion & Checkpointing ---
INGESTION_CHECKPOINT_PATH = os.getenv("INGESTION_CHECKPOINT_PATH", ".ingestion_state.db")

# --- Embedding Cache ---
EMBED_CACHE_ENABLED = os.getenv("EMBED_CACHE_ENABLED", "true").lower() == "true"
EMBED_CACHE_PATH = os.getenv("EMBED_CACHE_PATH", ".embed_cache.db")
# Max entries in LRU cache. <= 0 means unlimited.
EMBED_CACHE_MAX_ENTRIES = int(os.getenv("EMBED_CACHE_MAX_ENTRIES", "100000"))
