# potbot — Architecture & Design

A comprehensive guide to the system architecture, tech stack, design patterns, and how every component fits together.

---

## 1. What is potbot?

potbot is a **Retrieval-Augmented Generation (RAG)** application for querying internal company documents using natural language. It:

1. **Ingests** documents (PDF, DOCX, TXT, MD, CSV) → splits them into chunks → generates vector embeddings → stores them in Elasticsearch.
2. **Retrieves** the most relevant chunks when a user asks a question (using vector search, text search, or a hybrid of both).
3. **Generates** a natural-language answer by sending the retrieved context + the user's question to an LLM (Groq API).
4. **Tracks** every conversation and user feedback in PostgreSQL for monitoring and evaluation.

---

## 2. Tech Stack

| Layer              | Technology                           | Why this choice                                                                 |
|--------------------|--------------------------------------|---------------------------------------------------------------------------------|
| **Web UI**         | Streamlit                            | Rapid prototyping, built-in widgets, hot-reload, chat input support             |
| **LLM**           | Groq API (Llama 3.3 70B)            | Ultra-fast inference on open-weight models, free tier available                  |
| **Embeddings**     | sentence-transformers (`all-MiniLM-L6-v2`) | 100% local, no API key, 384-dim vectors, fast on CPU               |
| **Re-ranking**     | CrossEncoder (`ms-marco-MiniLM-L-6-v2`) | Local cross-encoder for high-accuracy relevance scoring             |
| **Vector Store**   | Elasticsearch 8.x                    | Supports both dense vector kNN and sparse BM25 in one engine — enables hybrid search |
| **Database**       | PostgreSQL 16                        | Reliable RDBMS for conversation logs, feedback, telemetry                       |
| **ORM**           | SQLAlchemy 2.0                       | Clean Repository pattern, session management, dialect-agnostic                  |
| **Monitoring**     | Grafana                              | Visual dashboards reading from PostgreSQL for latency, tokens, feedback trends  |
| **Doc Parsing**    | PyMuPDF, python-docx, csv, chardet   | Handles PDF, DOCX, TXT/MD, CSV with encoding detection                          |
| **Orchestration**  | Prefect (optional)                   | Pipeline orchestration for scheduled ingestion jobs                              |
| **Containerization** | Docker + Docker Compose            | One-command deployment of all 4 services                                         |

---

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                               │
│                     http://localhost:8501                            │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STREAMLIT APP (app/)                             │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  Sidebar UI  │    │   Chat UI    │    │   Session State       │  │
│  │ • Ingestion  │    │ • Input      │    │ • Message history     │  │
│  │ • Stats      │    │ • Responses  │    │ • Settings            │  │
│  │ • Settings   │    │ • Sources    │    │                       │  │
│  └──────┬───────┘    └──────┬───────┘    └───────────────────────┘  │
│         │                   │                                       │
└─────────┼───────────────────┼───────────────────────────────────────┘
          │                   │
          ▼                   ▼
┌──────────────────┐  ┌──────────────────────────────────────────────┐
│ INGESTION        │  │ RAG PIPELINE (rag/)                          │
│ PIPELINE         │  │                                              │
│ (ingestion/)     │  │  Query → Rewrite → Retrieve → Rerank →      │
│                  │  │  Build Prompt → LLM Generate → Response      │
│ Load → Chunk →   │  │                                              │
│ Embed → Index    │  └─────────┬────────────┬───────────┬───────────┘
│                  │            │            │           │
└────────┬─────────┘            │            │           │
         │                     │            │           │
         ▼                     ▼            ▼           ▼
┌──────────────────┐  ┌────────────┐ ┌────────────┐ ┌──────────┐
│  ELASTICSEARCH   │  │ SENTENCE   │ │   GROQ     │ │ POSTGRES │
│  (port 9200)     │  │ TRANSFORMER│ │   API      │ │(port 5432)│
│                  │  │  (local)   │ │ (cloud)    │ │          │
│ • Dense vectors  │  │            │ │            │ │ • Convos │
│ • BM25 text      │  │ • Embed    │ │ • Generate │ │ • Feedback│
│ • Hybrid search  │  │ • Rerank   │ │ • Rewrite  │ │ • Metrics│
└──────────────────┘  └────────────┘ └────────────┘ └──────────┘
```

---

## 4. Project Structure

```
llm-zoomcamp-project/
│
├── app/                          # Web UI layer
│   ├── streamlit_app.py          # Main Streamlit application
│   └── database.py               # PostgreSQL Repository (SQLAlchemy ORM)
│
├── domain/                       # Domain model layer (pure dataclasses)
│   └── models.py                 # Document, Chunk, SearchResult, RAGResponse, FeedbackRecord
│
├── ingestion/                    # Document ingestion pipeline
│   ├── loaders.py                # File format loaders (PDF, DOCX, TXT, CSV)
│   ├── document_loader.py        # (deprecated) old functional version
│   ├── chunkers.py               # Text splitting strategies
│   ├── chunker.py                # (deprecated) old functional version
│   ├── embedders.py              # Embedding generation (SentenceTransformer)
│   ├── embedder.py               # (deprecated) old functional version
│   ├── indexers.py               # Elasticsearch indexing & search
│   ├── indexer.py                # (deprecated) old functional version
│   └── pipeline.py               # Orchestrator: Load → Chunk → Embed → Index
│
├── rag/                          # RAG query pipeline
│   ├── query_rewriters.py        # LLM-based query expansion
│   ├── query_rewriter.py         # (deprecated) old functional version
│   ├── retrievers.py             # Search strategies (vector, text, hybrid + RRF)
│   ├── retriever.py              # (deprecated) old functional version
│   ├── rerankers.py              # Cross-encoder re-ranking
│   ├── reranker.py               # (deprecated) old functional version
│   ├── prompt_builders.py        # Prompt template construction
│   ├── prompt_builder.py         # (deprecated) old functional version
│   ├── llm_providers.py          # Groq LLM API client
│   ├── llm_client.py             # (deprecated) old functional version
│   └── pipeline.py               # Orchestrator: Rewrite → Retrieve → Rerank → Generate
│
├── evaluation/                   # Offline evaluation scripts
│   ├── ground_truth_generator.py # Synthetic Q&A generation from indexed chunks
│   ├── retrieval_eval.py         # Hit Rate & MRR across retrieval methods
│   └── llm_eval.py               # LLM-as-judge + cosine similarity scoring
│
├── monitoring/                   # Observability
│   └── grafana/                  # Grafana dashboard definitions & data source configs
│
├── scripts/                      # Utility scripts
│   └── generate_sample_documents.py  # Creates test documents
│
├── config.py                     # Centralized env-var configuration
├── docker-compose.yml            # Multi-service Docker deployment
├── Dockerfile                    # App container build
├── requirements.txt              # Python dependencies
└── docs/                         # Documentation (you are here)
```

---

## 5. Design Patterns Used

### 5.1 Strategy Pattern
Used extensively to make components swappable without changing calling code.

| Interface (ABC)       | Implementations                                            | Purpose                         |
|-----------------------|------------------------------------------------------------|---------------------------------|
| `BaseDocumentLoader`  | `PDFDocumentLoader`, `DocxDocumentLoader`, `TextDocumentLoader`, `CSVDocumentLoader` | Different file format parsers |
| `BaseChunker`         | `RecursiveCharacterChunker`, `MarkdownHeaderChunker`       | Different text splitting logic  |
| `BaseEmbedder`        | `SentenceTransformerEmbedder`                              | Embedding model abstraction     |
| `BaseVectorStore`     | `ElasticsearchVectorStore`                                 | Vector database abstraction     |
| `BaseSearchStrategy`  | `VectorSearchStrategy`, `TextSearchStrategy`, `HybridSearchStrategy` | Retrieval method selection |
| `BaseReranker`        | `CrossEncoderReranker`, `NoOpReranker`                     | Optional re-ranking             |
| `BaseQueryRewriter`   | `LLMQueryRewriter`, `NoOpQueryRewriter`                    | Optional query expansion        |
| `BasePromptBuilder`   | `TemplatePromptBuilder`                                    | Prompt style templates          |
| `BaseLLMProvider`     | `GroqLLMProvider`                                          | LLM API abstraction             |
| `BaseDatabaseRepository` | `PostgresDatabaseRepository`                            | Persistence abstraction         |

### 5.2 Composite Pattern
- `CompositeDocumentLoader` — holds a list of specialized loaders, delegates to the one that `can_load()` the file extension.
- `CompositeChunker` — routes Markdown files to `MarkdownHeaderChunker`, everything else to `RecursiveCharacterChunker`.

### 5.3 Factory Pattern
- `SearchStrategyFactory.get_strategy("hybrid")` — instantiates the correct strategy by name.

### 5.4 Repository Pattern
- `PostgresDatabaseRepository` — abstracts all database operations behind a clean interface (`save_conversation`, `save_feedback`, `get_recent_conversations`).

### 5.5 Facade Pattern
- `IngestionPipeline` — single `run()` method orchestrates 4 steps (load, chunk, embed, index).
- `RAGPipeline` — single `query()` method orchestrates 6 steps (strategy select, rewrite, retrieve, rerank, build prompt, generate).

### 5.6 Dependency Injection
Both pipelines accept optional component overrides in their constructors. If not provided, they create sensible defaults:

```python
class RAGPipeline:
    def __init__(
        self,
        search_strategy=None,   # inject custom strategy or use HybridSearchStrategy()
        reranker=None,          # inject custom reranker or use CrossEncoderReranker()
        ...
    ):
```

This makes the system **testable** (pass mocks) and **extensible** (swap components without changing the pipeline).

---

## 6. Data Flow Diagrams

### 6.1 Ingestion Flow (what happens when you click "Ingest")

```
User provides folder path or files
            │
            ▼
   ┌─────────────────────┐
   │ CompositeDocumentLoader │
   │                         │
   │ For each file:          │
   │  .pdf → PDFDocumentLoader (PyMuPDF, page-by-page)
   │  .docx → DocxDocumentLoader (python-docx)
   │  .txt/.md → TextDocumentLoader (chardet encoding detection)
   │  .csv → CSVDocumentLoader (csv.DictReader, row-by-row)
   │                         │
   │ Output: List[Document]  │  ← one Document per page (PDF) or per file (others)
   └──────────┬──────────────┘
              │
              ▼
   ┌─────────────────────┐
   │  CompositeChunker    │
   │                      │
   │  .md files → MarkdownHeaderChunker (split by ## headers, then recursive)
   │  others   → RecursiveCharacterChunker (split by ¶, \n, sentence, word)
   │                      │
   │  chunk_size=1000 chars, overlap=200 chars
   │                      │
   │  Output: List[Chunk]  │  ← each chunk gets a unique MD5 chunk_id
   └──────────┬───────────┘
              │
              ▼
   ┌─────────────────────────────┐
   │  SentenceTransformerEmbedder │
   │                              │
   │  Model: all-MiniLM-L6-v2    │
   │  Runs 100% locally (CPU)    │
   │  Output: 384-dimensional    │
   │  float vector per chunk     │
   │                              │
   │  chunk.embedding = [0.02, -0.13, ...]
   └──────────┬──────────────────┘
              │
              ▼
   ┌──────────────────────────────┐
   │  ElasticsearchVectorStore    │
   │                              │
   │  1. create_index() — creates │
   │     ES index with mapping:   │
   │     • text → BM25 analyzed   │
   │     • embedding → dense_vector (cosine, 384 dims)
   │     • file_name, chunk_id,   │
   │       doc_id, etc. → keyword │
   │                              │
   │  2. index_chunks() — bulk    │
   │     inserts via ES helpers   │
   │                              │
   │  Each chunk = 1 ES document  │
   └──────────────────────────────┘
```

### 6.2 Query Flow (what happens when you ask a question)

```
User types: "What is the vacation rollover limit?"
            │
            ▼
   ┌─────────────────────────┐
   │  1. Query Rewriting      │
   │     (LLMQueryRewriter)   │
   │                          │
   │  Calls Groq API:         │
   │  "Rewrite this query     │
   │   for better retrieval"  │
   │                          │
   │  "What is the vacation   │
   │   rollover limit?"       │
   │       ↓                  │
   │  "vacation PTO rollover  │
   │   day limit policy"      │
   └──────────┬──────────────┘
              │
              ▼
   ┌──────────────────────────┐
   │  2. Retrieval             │
   │     (HybridSearchStrategy)│
   │                           │
   │  Runs TWO searches in     │
   │  parallel against ES:     │
   │                           │
   │  Vector Search (kNN):     │
   │   • Embed query → 384-dim │
   │   • ES kNN similarity     │
   │   • Weight: 0.7           │
   │                           │
   │  Text Search (BM25):      │
   │   • Keyword matching      │
   │   • TF-IDF scoring        │
   │   • Weight: 0.3           │
   │                           │
   │  Reciprocal Rank Fusion:  │
   │   score = Σ weight/(rank+k)│
   │   Merges & deduplicates   │
   │                           │
   │  Output: top-10 results   │
   └──────────┬───────────────┘
              │
              ▼
   ┌──────────────────────────┐
   │  3. Re-ranking            │
   │     (CrossEncoderReranker)│
   │                           │
   │  Model: ms-marco-MiniLM   │
   │  Runs locally (CPU)       │
   │                           │
   │  Scores each (query, chunk)│
   │  pair for relevance       │
   │                           │
   │  Re-sorts by rerank_score │
   │  Output: top-3 results    │
   └──────────┬───────────────┘
              │
              ▼
   ┌──────────────────────────┐
   │  4. Prompt Building       │
   │     (TemplatePromptBuilder)│
   │                           │
   │  Formats retrieved chunks │
   │  into structured context: │
   │                           │
   │  "--- Context Doc 1 ---   │
   │   Source: vacation.md     │
   │   Content: Employees may  │
   │   roll over max 5 days..."│
   │                           │
   │  + System prompt (style-  │
   │    dependent: concise,    │
   │    detailed, structured)  │
   └──────────┬───────────────┘
              │
              ▼
   ┌──────────────────────────┐
   │  5. LLM Generation        │
   │     (GroqLLMProvider)     │
   │                           │
   │  API call to Groq cloud:  │
   │  Model: llama-3.3-70b     │
   │  Temperature: 0.1         │
   │  Max tokens: 1024         │
   │                           │
   │  Returns: answer text +   │
   │  token counts + latency   │
   └──────────┬───────────────┘
              │
              ▼
   ┌──────────────────────────┐
   │  6. Save to PostgreSQL    │
   │     (PostgresDatabaseRepo)│
   │                           │
   │  Persists: question,      │
   │  answer, sources, model,  │
   │  tokens, latency, flags   │
   │                           │
   │  Returns: conversation_id │
   │  (used for feedback 👍👎) │
   └──────────────────────────┘
```

---

## 7. Database Schema

PostgreSQL contains two tables, auto-created by SQLAlchemy on first startup:

### `conversations`

| Column               | Type        | Description                              |
|----------------------|-------------|------------------------------------------|
| `id`                 | SERIAL PK   | Auto-incrementing primary key            |
| `question`           | TEXT        | The user's original question             |
| `answer`             | TEXT        | The LLM-generated answer                 |
| `context`            | TEXT        | JSON string of retrieved source snippets |
| `model`              | VARCHAR(100)| LLM model used (e.g. `llama-3.3-70b-versatile`) |
| `prompt_style`       | VARCHAR(50) | `concise`, `detailed`, or `structured`   |
| `retrieval_method`   | VARCHAR(50) | `vector`, `text`, or `hybrid`            |
| `response_time_ms`   | INTEGER     | End-to-end LLM response time             |
| `prompt_tokens`      | INTEGER     | Tokens in the prompt sent to LLM         |
| `completion_tokens`  | INTEGER     | Tokens in the LLM response               |
| `total_tokens`       | INTEGER     | Total token usage                        |
| `reranking_used`     | BOOLEAN     | Whether re-ranking was enabled           |
| `query_rewriting_used` | BOOLEAN   | Whether query rewriting was enabled      |
| `created_at`         | TIMESTAMP   | UTC timestamp of the conversation        |

### `feedback`

| Column            | Type        | Description                          |
|-------------------|-------------|--------------------------------------|
| `id`              | SERIAL PK   | Auto-incrementing primary key        |
| `conversation_id` | INTEGER FK  | References `conversations.id`        |
| `sentiment`       | VARCHAR(20) | `positive` or `negative`             |
| `comment`         | TEXT        | Optional free-text comment           |
| `created_at`      | TIMESTAMP   | UTC timestamp of the feedback        |

---

## 8. Elasticsearch Index Schema

Index name: `potbot_documents` (configurable via `ELASTICSEARCH_INDEX`)

| Field          | ES Type        | Purpose                                        |
|----------------|----------------|-------------------------------------------------|
| `chunk_id`     | `keyword`      | Unique ID (MD5 of source_file + index)          |
| `doc_id`       | `keyword`      | Document-level ID (MD5 of source_file path)     |
| `text`         | `text`         | Chunk content, analyzed with BM25 for text search |
| `embedding`    | `dense_vector` | 384-dim cosine vector for kNN search            |
| `chunk_index`  | `integer`      | Position of this chunk within its source document |
| `source_file`  | `keyword`      | Absolute path to the original file              |
| `file_name`    | `keyword`      | Basename of the file (e.g. `policy.pdf`)        |
| `file_type`    | `keyword`      | Extension (`.pdf`, `.md`, `.txt`, `.csv`, `.docx`) |
| `page_number`  | `integer`      | Page number (PDFs only, null for others)        |
| `modified_date`| `date`         | Last-modified timestamp of the source file      |

Settings: 1 shard, 0 replicas (single-node dev setup).

---

## 9. Domain Models

All domain objects are **plain Python dataclasses** in `domain/models.py` with no framework dependencies:

| Model           | Role                                                              |
|-----------------|-------------------------------------------------------------------|
| `Document`      | Raw text extracted from a file (one per page or per file)         |
| `Chunk`         | A piece of a Document after splitting, with optional embedding    |
| `SearchResult`  | A retrieved chunk with relevance scores (BM25, kNN, RRF, rerank) |
| `RAGResponse`   | Complete response: answer + query + sources + metrics             |
| `FeedbackRecord`| User sentiment on a conversation                                 |

---

## 10. External Services & Their Roles

### Groq API (cloud — requires internet + API key)
- **Used for**: LLM text generation and query rewriting
- **Model**: `llama-3.3-70b-versatile` (open-weight, hosted by Groq for fast inference)
- **Rate limits**: Free tier ~30 requests/min; paid tier much higher
- **Data privacy**: Your document content is sent to Groq as part of the prompt context

### Hugging Face Hub (one-time download only)
- **Used for**: Downloading embedding and re-ranking model weights
- **Models downloaded**: `all-MiniLM-L6-v2` (~80 MB), `ms-marco-MiniLM-L-6-v2` (~90 MB)
- **Cache location**: `~/.cache/huggingface/hub/`
- **After first download**: Never contacts Hugging Face again — 100% offline inference
- **Rate limits**: None (it's a file download, not an API)

### Elasticsearch (local)
- **Used for**: Storing document chunks with both text and vector representations
- **Why not a dedicated vector DB (Pinecone, Weaviate)?**: ES supports both BM25 text search AND dense vector kNN in a single engine, enabling hybrid search without running two databases

### PostgreSQL (local)
- **Used for**: Conversation history, user feedback, telemetry metrics
- **Why not SQLite?**: PostgreSQL is the Grafana data source for real-time dashboards; also handles concurrent writes from multiple sessions
