FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for build & document parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch: avoids ~2.2 GB NVIDIA/CUDA wheels (container runs without GPU).
# satisfies sentence-transformers' torch>=1.11 requirement (resolved in the layer below).
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.13.0+cpu

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download sentence-transformer embedding model & cross-encoder model to container cache
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy application code
COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
