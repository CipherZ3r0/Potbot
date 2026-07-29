# potbot Usage Guide

Walkthrough of common workflows in potbot.

## 1. Generating Sample Documents
To generate a set of test corporate policy documents:
```bash
python scripts/generate_sample_documents.py
```
This generates documents in `data/sample_documents/`:
- `company_vacation_policy.md`
- `engineering_oncall_sop.md`
- `it_security_policy.txt`
- `department_budget_2026.csv`

---

## 2. Ingesting Documents via Streamlit UI
1. Open Streamlit UI (`http://localhost:8501`).
2. In the sidebar under **Document Ingestion**, enter the absolute path to your document folder (e.g., `data/sample_documents` or `/app/data/sample_documents`).
3. Click **🚀 Ingest Documents**.
4. The system will scan, chunk, embed, and index all documents into Elasticsearch.
5. **Incremental Runs**: If you re-ingest the exact same folder later, the system will instantly skip unchanged files using a SQLite state checkpoint. Identical text snippets across files will bypass heavy ML inference using the LRU embedding cache, dramatically accelerating the ingestion process.

---

## 3. Querying & Interacting
1. Type a question in the chat input (e.g., *"What is the vacation rollover limit?"* or *"What is the P1 alert response SLA?"*).
2. The assistant will return an answer with:
   - Grounded context from source documents
   - Expandable **Sources** section displaying original file name, page, and text snippet
   - Latency and token consumption metrics
3. Click **👍** or **👎** to rate the answer quality.

---

## 4. Running Offline Evaluations
To run retrieval and LLM evaluation scripts:

```bash
# 1. Generate synthetic ground truth Q&A pairs from indexed docs
python evaluation/ground_truth_generator.py data/ground_truth.json

# 2. Evaluate retrieval methods (Hit Rate & MRR)
python evaluation/retrieval_eval.py data/ground_truth.json

# 3. Evaluate LLM generation quality across prompt styles
python evaluation/llm_eval.py data/ground_truth.json
```

---

## 5. Viewing Telemetry in Grafana
1. Open Grafana (`http://localhost:3000`).
2. Log in with `admin` / `admin`.
3. Navigate to **Dashboards** → **potbot Monitoring Dashboard**.
4. Monitor real-time latency, token usage, query logs, and user feedback sentiment.
