# Indie Game Feedback Analytics Engine
 
An asynchronous backend engine that ingests thousands of unstructured player reviews (Steam/Discord), embeds them into high-dimensional vector space, clusters related feedback without manual labeling, and performs sub-40ms semantic search.
 
---
 
## What It Does
 
1. **Streams & Vectorizes**: Ingests raw feedback batches asynchronously and converts review text into 384-dimensional semantic vectors using `SentenceTransformers` (`all-MiniLM-L6-v2`).
2. **Stores with HNSW Indexing**: Persists vector embeddings into PostgreSQL via `pgvector` with an HNSW graph index configured for cosine similarity (`vector_cosine_ops`).
3. **Semantic Search (<40ms)**: Finds relevant bug reports and player sentiment instantly using cosine distance (`<=>`) without relying on exact keyword matching.
4. **Unsupervised Discovery**: Combines UMAP dimensionality reduction and HDBSCAN density clustering to separate conversational noise and isolate clusters of related gameplay balance and bug complaints.
---
 
## Tech Stack
 
- **Backend**: FastAPI, Python 3.12, `asyncpg`
- **Database**: PostgreSQL 16 with `pgvector` (Dockerized)
- **Machine Learning**: `sentence-transformers`, `umap-learn`, `hdbscan`
---
 
## Quickstart
 
### Prerequisites
 
- Docker & Docker Compose
- Python 3.11 or 3.12
### 1. Start the Vector Database
 
Launch the pre-configured PostgreSQL + `pgvector` container:
 
```bash
docker compose up -d
```
 
### 2. Set Up the Python Environment
 
```bash
python3.12 -m venv .venv
source .venv/bin/activate
 
pip install --upgrade pip
pip install -r requirements.txt
```
 
### 3. Start the API Server
 
```bash
uvicorn main:app --reload
```
 
The server will run on [http://127.0.0.1:8000](http://127.0.0.1:8000). You can inspect the interactive OpenAPI documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
 
### 4. Run the Verification Pipeline
 
In a second terminal window, run the test script to ingest sample feedback, test semantic search latency, and execute unsupervised clustering:
 
```bash
source .venv/bin/activate
python test_pipeline.py
```
 
---
 
## API Endpoints
 
- `POST /api/v1/ingest`: Batch ingest feedback records and insert generated vectors into PostgreSQL.
- `GET /api/v1/search?query=...&limit=5`: Retrieve top-k semantically closest reviews with benchmarked execution latency.
- `POST /api/v1/cluster?min_cluster_size=2`: Run UMAP + HDBSCAN clustering over stored feedback and prune noise.
---
 
## Roadmap & Next Steps
 
This repository is currently the standalone backend analytics engine. The next milestones to build this into a full end-to-end product include:
 
- [ ] **LLM Ticket Synthesizer**: Feed discovered HDBSCAN clusters to an LLM with strict Pydantic schemas to automatically draft bug tickets with reproduction steps and severity levels.
- [ ] **Next.js Kanban Dashboard**: Build a user interface with draggable Kanban columns to organize synthesized bug tickets and view raw supporting feedback.
- [ ] **Large File Upload Worker**: Add background workers and streaming parsers (`ijson` / `aiocsv`) to handle 50,000+ row CSV and JSON exports with flat `O(1)` memory usage.
- [ ] **Webhook Ingestion**: Add automated webhooks for live Discord feedback channels and scheduled Steam review scrapers.
 
