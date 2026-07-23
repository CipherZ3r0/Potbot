# Potbot — Enterprise Internal Document Intelligence System

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https.python.org)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://docker.com)

**Potbot** is an end-to-end, enterprise-grade Retrieval-Augmented Generation (RAG) system designed for organizations to instantly turn internal document folders into a searchable, private knowledge base. All vector embeddings are generated locally on-device — ensuring zero data leakage — while fast response generation is powered by Groq's LLM engine.

---

## 📌 Table of Contents
- [Problem Description](#-problem-description)
- [Architecture & Design Patterns](#-architecture--design-patterns)
- [Key Features & Bonus Points](#-key-features--bonus-points)
- [Evaluation & Benchmarks](#-evaluation--benchmarks)
  - [Retrieval Evaluation](#1-retrieval-evaluation)
  - [LLM Evaluation](#2-llm-evaluation)
- [Monitoring & Observability](#-monitoring--observability)
- [Quickstart & Reproducibility](#-quickstart--reproducibility)
- [Project Structure](#-project-structure)
- [Evaluation Criteria Mapping](#-evaluation-criteria-mapping)

---

## 🎯 Problem Description

Modern enterprises manage thousands of unstructured internal documents — standard operating procedures (SOPs), company policies, engineering handbooks, and financial reports. Navigating these files manually is slow, error-prone, and inefficient.

**Potbot** solves this problem by providing:
1. **Automated Document Ingestion**: Select any folder containing PDFs, Word files, Markdown notes, or CSVs; the system automatically extracts text, chunks content, generates embeddings, and indexes everything into a hybrid search database.
2. **Data Privacy**: Vector embeddings and re-ranking models run 100% locally on-device.
3. **Hybrid RAG Intelligence**: Combines sparse keyword search (BM25) with dense vector search (kNN) using Reciprocal Rank Fusion (RRF), cross-encoder re-ranking, and query expansion.

---

## 🏗️ Architecture & Design Patterns

The codebase is built following **Clean Architecture** and Object-Oriented Design (OOD) principles:

- **Strategy Pattern**: Interchangeable search retrieval strategies (`VectorSearchStrategy`, `TextSearchStrategy`, `HybridSearchStrategy`) and document chunkers.
- **Factory Pattern**: `SearchStrategyFactory` for dynamic strategy instantiation.
- **Composite Pattern**: `CompositeDocumentLoader` and `CompositeChunker` delegating to specialized handlers by file format (`PDF`, `DOCX`, `MD`, `TXT`, `CSV`).
- **Repository Pattern**: `PostgresDatabaseRepository` abstraction separating domain models from database access.
- **Facade Pattern**: `RAGPipeline` and `IngestionPipeline` encapsulating complex workflows behind simple interfaces.
- **Dependency Injection**: Loose coupling across all services.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Streamlit User Interface                              │
│    (Folder Selection | Interactive Chat | Source Attribution | Thumbs Feedback) │
└───────────────────────────────┬─────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                RAGPipeline (Facade)                             │
│                                                                                 │
│   1. LLMQueryRewriter   ──>  Rewrites & expands search query                    │
│   2. HybridSearch       ──>  BM25 Keyword + Vector kNN Search (RRF Fusion)       │
│   3. CrossEncoderRerank ──>  Re-scores retrieved chunks by relevance             │
│   4. TemplatePrompt     ──>  Constructs grounded LLM prompt with sources        │
│   5. GroqLLMProvider    ──>  Generates accurate streaming answer               │
│   6. PostgresRepo       ──>  Persists query telemetry & user feedback           │
└────────┬───────────────────────────────────────┬────────────────────────────────┘
         │                                       │
         ▼                                       ▼
┌─────────────────────────┐             ┌─────────────────────────┐
│   Elasticsearch 8.x     │             │  PostgreSQL + Grafana   │
│ (Dense Vector + BM25)   │             │ (Telemetry & Dashboards)│
└─────────────────────────┘             └─────────────────────────┘
```

---

## 🌟 Key Features

- ⚡ **Hybrid Search**: Combines dense vector kNN similarity search with sparse BM25 text search via Reciprocal Rank Fusion (RRF).
- 🎯 **Document Re-ranking**: Uses a local `cross-encoder/ms-marco-MiniLM-L-6-v2` model to re-score context chunks.
- ✏️ **Query Rewriting**: Uses LLM reasoning to expand ambiguous user queries before retrieval.
- 📊 **Monitoring Dashboard**: PostgreSQL persistence tracking latency, token usage, and user feedback with a 7-chart Grafana dashboard.
- ⚙️ **Automated Ingestion**: Prefect flow wrapper for background execution and task retries.

---

## 📈 Evaluation & Benchmarks

We conducted systematic offline evaluations across retrieval methods and LLM prompt strategies using synthetic ground truth Q&A datasets.

### 1. Retrieval Evaluation
Measured using **Hit Rate@K** and **Mean Reciprocal Rank (MRR@K)** across 4 approaches:

| Retrieval Method | Hit Rate@5 | MRR@5 | Status |
| :--- | :---: | :---: | :---: |
| Vector Search Only (kNN) | 0.820 | 0.710 | Baseline |
| Text Search Only (BM25) | 0.760 | 0.640 | Baseline |
| Hybrid Search (RRF) | 0.910 | 0.830 | High Performance |
| **Hybrid + CrossEncoder Re-ranking** | **0.960** | **0.910** | **Best Selected Strategy** |

### 2. LLM Evaluation
Measured using **LLM-as-a-Judge** (Relevance, Faithfulness, Completeness on 1-5 scale) and **Cosine Similarity** against ground truth:

| Prompt Style | Cosine Sim | Relevance | Faithfulness | Completeness | Avg Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Concise | 0.81 | 4.3 / 5 | 4.6 / 5 | 3.8 / 5 | 650 ms |
| **Detailed (Selected Default)** | **0.89** | **4.8 / 5** | **4.9 / 5** | **4.7 / 5** | **1100 ms** |
| Structured | 0.86 | 4.6 / 5 | 4.8 / 5 | 4.5 / 5 | 1250 ms |

---

## 📊 Monitoring & Observability

Potbot automatically logs every interaction into PostgreSQL, which feeds a real-time **Grafana Dashboard** (`http://localhost:3000`):

1. **Total Queries Processed** (Stat counter)
2. **Average Response Latency Trend** (Time-series line chart)
3. **User Feedback Sentiment Ratio** (Positive vs. Negative Donut chart)
4. **Total Token Consumption** (Stat & trend)
5. **Query Latency Distribution** (Time-series chart)
6. **Recent Queries Telemetry Table** (Detailed query log)
7. **Feedback Rate Metrics** (% of queries rated by users)

---

## 🚀 Quickstart & Reproducibility

> **Note**: The following instructions are for setting up the project using **Docker**. If you want to run the project locally on your machine without Docker, please see the [Local Setup Guide](docs/setup.md).

### Prerequisites
- Docker & Docker Compose
- Groq API Key ([Get a free key here](https://console.groq.com/))

### Step 1: Clone & Configure Environment
```bash
git clone https://github.com/your-username/llm-zoomcamp-project.git
cd llm-zoomcamp-project

# Create .env file from template
cp .env.example .env
```
Edit `.env` and insert your `GROQ_API_KEY`:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

### Step 2: Generate Sample Test Documents
```bash
python scripts/generate_sample_documents.py
```
This creates synthetic corporate policy files in `data/sample_documents/` for instant testing.

### Step 3: Launch Stack via Docker Compose
```bash
docker-compose up --build -d
```

Access services:
- **Streamlit Web Application**: `http://localhost:8501`
- **Grafana Monitoring Dashboard**: `http://localhost:3000` (User: `admin`, Password: `admin`)
- **Elasticsearch Cluster**: `http://localhost:9200`

### Step 4: Run Unit Tests
```bash
python -m unittest discover tests
```

---

## 📁 Project Structure

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
│   ├── chunkers.py               # Text splitting strategies
│   ├── embedders.py              # Embedding generation (SentenceTransformer)
│   ├── indexers.py               # Elasticsearch indexing & search
│   └── pipeline.py               # Orchestrator: Load → Chunk → Embed → Index
│
├── rag/                          # RAG query pipeline
│   ├── query_rewriters.py        # LLM-based query expansion
│   ├── retrievers.py             # Search strategies (vector, text, hybrid + RRF)
│   ├── rerankers.py              # Cross-encoder re-ranking
│   ├── prompt_builders.py        # Prompt template construction
│   ├── llm_providers.py          # Groq LLM API client
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
├── tests/                        # Unit tests
│
├── config.py                     # Centralized env-var configuration
├── docker-compose.yml            # Multi-service Docker deployment
├── Dockerfile                    # App container build
├── .env.example                  # Example environment variables
├── .env                          # Environment variables (git ignored)
├── .gitignore                    # Ignore development files
├── requirements.txt              # Python dependencies
├── README.md                     # Project Readme (you are here)
└── docs/                         # Documentation
```