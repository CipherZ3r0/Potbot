# Potbot End-to-End Walkthrough

This guide walks you through a complete, end-to-end demonstration of the Potbot system, from generating sample data and indexing documents to querying the RAG system and monitoring performance in Grafana.

---

## 1. Initial Setup and Sample Data Generation

Before using the application, ensure all services are running via Docker. Then, you can generate sample corporate documents to test the system.

```bash
# Generate sample documents
python scripts/generate_sample_documents.py
```

These documents are saved in the `data/sample_documents/` folder.

<!-- [Insert screenshot of terminal output from data generation] -->
<br>

---

## 2. Accessing the Streamlit Application

The main interface for Potbot is built using Streamlit. Open your browser and navigate to `http://localhost:8501`.

You will see the main chat interface on the right and a sidebar for configuration and document ingestion on the left.

<!-- [Insert screenshot of Streamlit home screen / empty chat] -->
<br>

---

## 3. Ingesting Documents into the Vector Database

Before we can chat with our documents, we need to ingest them. Potbot extracts text, splits it into semantic chunks, generates embeddings locally, and stores them in Elasticsearch.

1. In the left sidebar, under **Document Ingestion**, locate the input field for the folder path.
2. Enter `/app/data/sample_documents` (the internal Docker path for the generated files).
3. Click the **🚀 Ingest Documents** button.
4. Wait for the success notification indicating all files were parsed and indexed.

<!-- [Insert screenshot of the sidebar with successful ingestion message] -->
<br>

---

## 4. Querying the Assistant

Now that the knowledge base is populated, you can ask questions based on the ingested documents. Potbot uses a Hybrid Search approach (BM25 + Vector kNN) combined with cross-encoder re-ranking to find the most relevant chunks.

Try asking a question related to the sample documents, such as:
> *"What is our policy on vacation rollover days?"*

<!-- [Insert screenshot of user query and the assistant's response in Streamlit] -->
<br>

---

## 5. Exploring Sources and Grounding

Potbot ensures all answers are traceable back to internal documents. 
Underneath the assistant's response, you will find an expandable **Sources** section.

1. Click on the **Sources** expander to see the exact chunks of text retrieved from the database.
2. Note the file name and relevance scores that contributed to the answer.

<!-- [Insert screenshot showing the expanded Sources tab with document excerpts] -->
<br>

---

## 6. Providing Feedback

Potbot captures user feedback to evaluate answer quality over time.
Under every response, there are **👍** and **👎** buttons.

1. Click the **👍 (Thumbs Up)** button if the answer was helpful and accurate.
2. The UI will confirm that your feedback has been recorded in the PostgreSQL telemetry database.

<!-- [Insert screenshot showing the thumbs up/down buttons and feedback confirmation] -->
<br>

---

## 7. Monitoring Telemetry in Grafana

All queries, latencies, token consumption, and user feedback are persisted to PostgreSQL and visualized in real-time using Grafana.

1. Open Grafana by navigating to `http://localhost:3000`.
2. Log in with the default credentials (`admin` / `admin`).
3. Open the **Potbot Monitoring Dashboard** from the dashboards menu.

<!-- [Insert screenshot of the Grafana dashboard showing query metrics and charts] -->
<br>

### Key Grafana Panels:
- **Total Queries & Tokens**: Keep track of usage and LLM costs.
- **Latency Distribution**: Monitor how fast the RAG pipeline is generating responses.
- **User Feedback Sentiment**: A donut chart showing the ratio of positive to negative feedback.
- **Recent Queries Log**: A raw table showing exact user inputs, latency, and the feedback provided.

<!-- [Insert screenshot zooming in on the Recent Queries log table] -->
<br>

---

## 8. Offline Evaluation (Optional)

Potbot includes scripts to formally evaluate retrieval metrics (Hit Rate & MRR) and generation quality (LLM-as-a-judge).

Run the evaluation suite in your terminal:
```bash
python evaluation/retrieval_eval.py data/ground_truth.json
python evaluation/llm_eval.py data/ground_truth.json
```

<!-- [Insert screenshot of terminal output showing MRR, Hit Rate, and LLM-as-a-judge scores] -->
<br>
