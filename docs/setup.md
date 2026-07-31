# potbot Setup Guide

Detailed instructions for installing, configuring, and running potbot — both locally (without Docker) and with Docker Compose.

## 📑 Table of Contents

- [🛠️ System Requirements](#️-system-requirements)
- [🔑 Environment Configuration](#-environment-configuration)
- [💻 Local Development Setup (Without Docker)](#-local-development-setup-without-docker)
  - [Step 1 — Python virtual environment](#step-1--python-virtual-environment)
  - [Step 2 — Install Python dependencies](#step-2--install-python-dependencies)
  - [Step 3 — Install & Run Elasticsearch Locally](#step-3--install--run-elasticsearch-locally)
  - [Step 4 — Install & Run PostgreSQL Locally](#step-4--install--run-postgresql-locally)
  - [Step 5 — Update `.env`](#step-5--update-env-for-local-development)
  - [Step 6 — Run the application](#step-6--run-the-application)
  - [Step 7 — Run unit tests](#step-7--run-unit-tests-optional)
- [🐳 Docker Deployment (All Services)](#-docker-deployment-all-services)
  - [Prerequisites](#prerequisites)
  - [Step 1 — Set Docker host values](#step-1--set-docker-host-values-in-env)
  - [Step 2 — Build and start containers](#step-2--build-and-start-all-containers)
  - [Step 3 — Verify services](#step-3--verify-services-are-healthy)
  - [Step 4 — View logs](#step-4--view-logs)
  - [Step 5 — Access the application](#step-5--access-the-application)
  - [Step 6 — Stop all services](#step-6--stop-all-services)
- [🤗 About Hugging Face & the Embedding Model](#-about-hugging-face--the-embedding-model)
  - [How it works](#how-it-works)
  - [Are there usage limits?](#are-there-usage-limits)
  - [Working in an air-gapped / offline environment](#working-in-an-air-gapped--offline-environment)

## 🛠️ System Requirements & Version Compatibility

| Requirement     | Supported Versions | Recommended | Notes |
|-----------------|-------------------|-------------|-------|
| OS              | Linux, macOS, Windows 10/11 | Linux / macOS | Tested on Linux x86_64 |
| Python          | `3.10.x`, `3.11.x`, `3.12.x` | `3.11.x` | Primary production target: Python 3.11 |
| Streamlit       | `1.35.0` – `1.45.x` | `1.45.1` | Configured with `fileWatcherType = "none"` in `.streamlit/config.toml` |
| PyTorch / Torch | `2.0.0` – `2.6.x` | `2.13.0` (or `2.x`) | CPU / CUDA / Apple MPS supported |
| RAM             | 8 GB minimum | 16 GB | Required for parallel embedding batching |
| Disk            | 5 GB free | 10 GB free | Vector cache & Docker images |
| Docker          | Docker Desktop 4.x / Engine 24+ | Latest | Multi-container setup via Docker Compose |


---

## 🔑 Environment Configuration

Create a `.env` file in the project root by copying the example:

```bash
cp .env.example .env          # Linux / macOS / Git Bash
copy .env.example .env        # Windows CMD
```

Open `.env` and set your values. The most important variable is your **Groq API key**:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### Ingestion Tuning (Optional)
You can tune the ingestion pipeline's performance and memory usage by adding these optional variables to your `.env`:

```env
# Concurrency tuning
INGESTION_LOADER_WORKERS=4           # Number of threads for reading files
INGESTION_CHUNKER_WORKERS=2          # Number of CPU processes for parsing text

# Batched execution
INGESTION_EMBED_BATCH_SIZE=32        # Number of chunks to embed at once
INGESTION_BULK_INDEX_SIZE=500        # Number of chunks to send to Elasticsearch at once

# Hardware & Caching
INGESTION_DEVICE=auto                # auto, cuda, mps, cpu
INGESTION_INCREMENTAL=true           # Skip unchanged files using sqlite hashes
INGESTION_EMBED_CACHE_ENABLED=true   # Cache vector embeddings in sqlite
```

### Local vs Docker host values

| Variable             | Local Development      | Docker Deployment           |
|----------------------|------------------------|-----------------------------|
| `ELASTICSEARCH_HOST` | `http://localhost:9200` | `http://elasticsearch:9200` |
| `POSTGRES_HOST`      | `localhost`             | `postgres`                  |

> **Important**: If you switch between local and Docker modes, update these two values in `.env` accordingly.

---

## 💻 Local Development Setup (Without Docker)

Use this approach when you want to run everything directly on your machine without Docker.

### Step 1 — Python virtual environment

```bash
# Create the venv
python -m venv .venv

# Activate it
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
```

### Step 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

---

### Step 3 — Install & Run Elasticsearch Locally

Elasticsearch is the vector / text search engine that stores and retrieves document chunks.

#### Option A — Download the zip (no Docker needed)

1. **Download** Elasticsearch 8.18.0 (must match the project's client version):
   - Windows: https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.18.0-windows-x86_64.zip
   - Linux:   https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.18.0-linux-x86_64.tar.gz
   - macOS:   https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.18.0-darwin-x86_64.tar.gz

2. **Extract** the archive to a folder, e.g. `C:\elasticsearch` or `~/elasticsearch`.

3. **Disable security** (required for local dev — the app connects without authentication).
   Open `config/elasticsearch.yml` inside the extracted folder and add/modify these lines:

   ```yaml
   xpack.security.enabled: false
   xpack.security.http.ssl.enabled: false
   discovery.type: single-node
   ```

4. **Start Elasticsearch**:

   ```bash
   # Windows (PowerShell):
   C:\elasticsearch\bin\elasticsearch.bat

   # Linux / macOS:
   ~/elasticsearch/bin/elasticsearch
   ```

   > Leave this terminal open — Elasticsearch runs in the foreground by default.

5. **Verify it's running** — open a new terminal:

   ```bash
   curl http://localhost:9200
   ```

   You should see a JSON response with `"cluster_name"`, `"version"`, etc. Example:

   ```json
   {
     "name" : "your-pc",
     "cluster_name" : "elasticsearch",
     "cluster_uuid" : "...",
     "version" : { "number" : "8.18.0", ... },
     "tagline" : "You Know, for Search"
   }
   ```

#### Option B — Run Elasticsearch via Docker (but keep the app outside Docker)

If you have Docker but want to run only the app locally:

```bash
docker run -d --name es-local \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  docker.elastic.co/elasticsearch/elasticsearch:8.18.0
```

Windows PowerShell (use backticks instead of backslashes):

```powershell
docker run -d --name es-local `
  -p 9200:9200 -p 9300:9300 `
  -e "discovery.type=single-node" `
  -e "xpack.security.enabled=false" `
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" `
  docker.elastic.co/elasticsearch/elasticsearch:8.18.0
```

Verify: `curl http://localhost:9200`

#### Troubleshooting Elasticsearch

| Symptom | Fix |
|---------|-----|
| `could not translate host name "elasticsearch"` | Your `.env` has `ELASTICSEARCH_HOST=http://elasticsearch:9200`. Change it to `http://localhost:9200`. |
| `Connection refused on port 9200` | Elasticsearch is not running. Start it with the commands above. |
| `Elasticsearch ping failed` | Check that security is disabled (`xpack.security.enabled: false`). ES 8.x enables security by default. |
| `Java heap space` error | Increase heap: set `ES_JAVA_OPTS=-Xms1g -Xmx1g` (in `jvm.options` or docker env). |
| Windows firewall popup | Click **Allow access** for private networks. |

---

### Step 4 — Install & Run PostgreSQL Locally

PostgreSQL stores conversation history, feedback, and telemetry data. The app will still run without it (you'll see a warning), but feedback recording and Grafana dashboards will not work.

#### Option A — Native installer (no Docker)

1. **Download** PostgreSQL 16 from https://www.postgresql.org/download/
   - Windows: Use the interactive installer from EDB (https://www.enterprisedb.com/downloads/postgres-postgresql-downloads).
   - Linux (Debian/Ubuntu): `sudo apt install postgresql-16`
   - macOS: `brew install postgresql@16`

2. **Start the service**:

   ```bash
   # Windows: it starts automatically after install.
   # Check via Services (Win+R → services.msc → look for "postgresql-x64-16").

   # Linux:
   sudo systemctl start postgresql
   sudo systemctl enable postgresql

   # macOS (Homebrew):
   brew services start postgresql@16
   ```

3. **Create the database and user** matching your `.env` values:

   ```bash
   # Connect as the postgres superuser:
   # Windows (use SQL Shell / psql from Start Menu) or:
   psql -U postgres

   # Then run:
   CREATE USER potbot WITH PASSWORD 'potbot_secret';
   CREATE DATABASE potbot OWNER potbot;
   GRANT ALL PRIVILEGES ON DATABASE potbot TO potbot;
   \q
   ```

4. **Verify the connection**:

   ```bash
   psql -h localhost -U potbot -d potbot
   # Enter password: potbot_secret
   # You should get the psql prompt. Type \q to exit.
   ```

#### Option B — Run PostgreSQL via Docker (but keep the app outside Docker)

```bash
docker run -d --name pg-local \
  -p 5432:5432 \
  -e POSTGRES_DB=potbot \
  -e POSTGRES_USER=potbot \
  -e POSTGRES_PASSWORD=potbot_secret \
  postgres:16-alpine
```

Windows PowerShell:

```powershell
docker run -d --name pg-local `
  -p 5432:5432 `
  -e POSTGRES_DB=potbot `
  -e POSTGRES_USER=potbot `
  -e POSTGRES_PASSWORD=potbot_secret `
  postgres:16-alpine
```

Verify:

```bash
psql -h localhost -U potbot -d potbot
# password: potbot_secret
```

#### Troubleshooting PostgreSQL

| Symptom | Fix |
|---------|-----|
| `could not translate host name "postgres"` | Your `.env` has `POSTGRES_HOST=postgres`. Change it to `localhost`. |
| `connection refused on port 5432` | PostgreSQL is not running. Start it with the commands above. |
| `password authentication failed` | The user/password in `.env` doesn't match what's in PostgreSQL. Re-run the `CREATE USER` command. |
| `database "potbot" does not exist` | Run `CREATE DATABASE potbot OWNER potbot;` in psql. |

---

### Step 5 — Update `.env` for local development

Make sure your `.env` has these values (not Docker service names):

```env
ELASTICSEARCH_HOST=http://localhost:9200
POSTGRES_HOST=localhost
```

---

### Step 6 — Run the application

```bash
# Make sure your venv is activated, then:
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

### Step 7 — Run unit tests (optional)

```bash
python -m unittest discover tests
```

---

## 🐳 Docker Deployment (All Services)

Use this when you want to run **everything** (Elasticsearch, PostgreSQL, Grafana, and the app) inside Docker.

### Prerequisites

- Docker Desktop installed and running
- At least 4 GB RAM allocated to Docker (Settings → Resources → Memory)

### Step 1 — Set Docker host values in `.env`

```env
ELASTICSEARCH_HOST=http://elasticsearch:9200
POSTGRES_HOST=postgres
```

> Inside Docker Compose, services reference each other by **service name** (e.g. `elasticsearch`, `postgres`), not `localhost`.

### Step 2 — Build and start all containers

```bash
docker-compose up --build -d
```

This starts four services:

| Service          | Port  | Description                    |
|------------------|-------|--------------------------------|
| `elasticsearch`  | 9200  | Vector + text search engine    |
| `postgres`       | 5432  | Feedback & telemetry database  |
| `grafana`        | 3000  | Monitoring dashboards          |
| `streamlit-app`  | 8501  | The potbot application      |

> **Note**: The ingestion pipeline uses two local SQLite files (`.embed_cache.db` and `.ingest_state.db`) for caching and incremental runs. These are stored locally in the project root and are mounted or written directly by the application. They are safely ignored by git.

### Step 3 — Verify services are healthy

```bash
docker-compose ps
```

All services should show `Up (healthy)` or `Up`.

### Step 4 — View logs

```bash
# All services:
docker-compose logs -f

# Just the app:
docker-compose logs -f streamlit-app
```

### Step 5 — Access the application

| Service     | URL                     | Credentials            |
|-------------|-------------------------|------------------------|
| potbot   | http://localhost:8501    | —                      |
| Grafana     | http://localhost:3000    | `admin` / `admin`      |
| Elasticsearch | http://localhost:9200 | —                      |

### Step 6 — Stop all services

```bash
# Stop containers (preserves data volumes):
docker-compose down

# Stop and DELETE all data (fresh start):
docker-compose down -v
```

### Troubleshooting & Framework Notes

| Symptom / Issue | Cause | Fix / Resolution |
|-----------------|-------|------------------|
| Streamlit crash: `RuntimeError: Tried to instantiate class '__path__._path'` | PyTorch 2.x modules in `sys.modules` break Streamlit's default file watcher inspection. | **Permanent Fix**: `.streamlit/config.toml` is configured with `[server] fileWatcherType = "none"`. |
| Postgres authentication error (`FATAL: password authentication failed`) | `.env` credentials don't match `docker-compose.yml`. | Standardize `.env` values (`POSTGRES_USER=potbot`, `POSTGRES_DB=potbot`, `POSTGRES_PASSWORD=potbot_secret`). |
| Container keeps restarting | Memory exhaustion on large embeddings. | Increase Docker RAM allocation to 4 GB+ or reduce `INGESTION_EMBED_BATCH_SIZE` in `.env`. |
| `elasticsearch` health check fails | Elasticsearch JVM warmup takes 30–60s. | Wait for container initialization and verify with `docker-compose ps`. |
| Port already in use | Host port conflict on 9200, 5432, or 8501. | Stop conflicting host process or adjust host port mapping in `docker-compose.yml`. |


---

## 🤗 About Hugging Face & the Embedding Model

potbot uses the `sentence-transformers/all-MiniLM-L6-v2` model for generating document embeddings.

### How it works

- On **first run**, the model weights (~80 MB) are downloaded from Hugging Face and **cached locally** in `~/.cache/huggingface/`.
- On **subsequent runs**, the cached copy is used — **no internet connection needed**.
- All inference runs **100% locally on your CPU/GPU**. No data is sent to Hugging Face servers.

### Are there usage limits?

**No.** Downloading the model is a simple file download (like cloning a git repo). There are:

- ❌ No API rate limits
- ❌ No per-request charges
- ❌ No authentication required (for public models)
- ❌ No telemetry or data sharing

Hugging Face is only contacted **once** to download the model files. After that, everything is offline.

### Working in an air-gapped / offline environment

If your machine cannot reach `huggingface.co` at all:

1. Download the model on a machine with internet:
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
   model.save("./models/all-MiniLM-L6-v2")
   ```

2. Copy the `models/all-MiniLM-L6-v2` folder to the target machine.

3. Update `EMBEDDING_MODEL` in `.env`:
   ```env
   EMBEDDING_MODEL=./models/all-MiniLM-L6-v2
   ```
