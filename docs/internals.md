# potbot — Deep Dive & Internals

This document explains the "why" and "how" behind potbot's core logic, algorithms, and observed behaviors.

---

## 1. Document Parsing & Chunking

### Why Chunking Matters
LLMs have a finite context window (e.g., 8k or 128k tokens). You cannot feed a 500-page corporate manual into a single prompt. Furthermore, retrieving a 500-page document doesn't help the LLM pinpoint the exact answer. 

We break documents down into **chunks** (default 1000 characters with 200 characters overlap).
- **Overlap** prevents cutting a crucial sentence in half.
- **Chunk Size** balances between having enough context to make sense, and being small enough to retrieve highly relevant specific snippets.

### How it Works (CompositeChunker)
The system adapts how it chunks based on the file type:
- **Markdown (`.md`)**: Uses `MarkdownHeaderChunker`. It uses regex (`^#{1,6}`) to split the document logically by headers (e.g., keeping "Section 2" together). If a section is still too large, it falls back to character splitting.
- **Other text/PDFs**: Uses `RecursiveCharacterChunker`. It tries to split gracefully on paragraphs (`\n\n`), then lines (`\n`), then sentences (`. `), and finally words.

---

## 2. The Search Stack: Hybrid + Re-ranking

potbot uses a multi-stage retrieval pipeline to achieve maximum accuracy.

### Stage 1: Hybrid Retrieval (Elasticsearch)
Vector search is great at understanding *meaning*, but terrible at exact keyword matching (e.g., searching for a specific product ID like `AX-992B`). Text search is the exact opposite.

potbot does both simultaneously:
1. **Vector Search (kNN)**: Embeds the user's query into a 384-dimensional vector and asks Elasticsearch for the closest chunks using cosine similarity.
2. **Text Search (BM25)**: Asks Elasticsearch for chunks containing exact keywords from the query.

**Merging the Results (RRF):**
The `HybridSearchStrategy` uses **Reciprocal Rank Fusion (RRF)** to combine the two lists. 
If a chunk is Rank #1 in Vector and Rank #10 in Text, its RRF score is:
`Score = (VectorWeight / (1 + 60)) + (TextWeight / (10 + 60))`
This mathematically guarantees that chunks performing well in *both* semantic and keyword searches bubble to the top.

### Stage 2: Cross-Encoder Re-ranking
The initial embedding model (`all-MiniLM-L6-v2`) is a **Bi-encoder**. It embeds the query and the document separately. This is fast, but misses fine-grained interactions between words.

The re-ranking model (`ms-marco-MiniLM-L-6-v2`) is a **Cross-encoder**. It takes the query AND the document together as a single input and outputs a highly accurate relevance score (0.0 to 1.0).
- **Why not use this everywhere?** It is too slow to run across thousands of documents.
- **The Solution**: We use Elasticsearch to quickly grab the top 10 candidates, then use the slow/accurate Cross-encoder to re-sort those 10 candidates to pick the absolute best 3 to send to the LLM.

---

## 3. Query Rewriting

Users often ask lazy follow-up questions like:
> "What is the policy?" 

If you search for that exact string, the vector database will return random policies.
potbot intercepts the question and sends it to the LLM with a hidden `SYSTEM_PROMPT` (inside `LLMQueryRewriter`). The LLM rewrites the query to make it search-friendly (e.g., expanding abbreviations, adding context) *before* the database search happens.

---

## 4. Addressing System Behaviors

### Q: Why do "Indexed Chunks" seem to increase asynchronously?
You may notice the "Indexed Chunks" metric in the sidebar updating slightly after ingestion, or appearing to increase as you use the app. 

1. **Questions do NOT index new chunks**: Asking a question in the chat does not add new documents to Elasticsearch. The RAG pipeline only reads from the index.
2. **Elasticsearch Refresh Interval**: When the `IngestionPipeline` bulk-inserts chunks into Elasticsearch, the data is written to disk but is not instantly searchable. Elasticsearch operates in "near real-time", refreshing its index segments periodically (usually every 1 second). 
3. **Metric Polling**: The Streamlit sidebar calls `es.indices.stats()` which reads the count of primary documents. If an ingestion job just finished, Elasticsearch might still be merging segments in the background, causing the exact document count to fluctuate or catch up over a few seconds.

### Q: Where are my conversations saved?
When the RAG pipeline finishes generating an answer, it asynchronously fires `repository.save_conversation()`. This saves the query, the AI's answer, the retrieved context, and token usage into the local PostgreSQL database (`conversations` table). 

When you click the 👍 or 👎 buttons on a message, a `FeedbackRecord` is written to the `feedback` table linked to that specific conversation ID. This allows administrators to use Grafana to track exactly which documents are producing bad answers.
