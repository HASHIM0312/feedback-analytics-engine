import json
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import asyncpg
import hdbscan
import numpy as np
import umap
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

DATABASE_URL = "postgresql://postgres:password@localhost:5432/feedback_analytics"

model: Optional[SentenceTransformer] = None
db_pool: Optional[asyncpg.Pool] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, db_pool
    # 384-dimensional embeddings, local CPU-friendly execution
    model = SentenceTransformer("all-MiniLM-L6-v2")
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=5, max_size=20)
    yield
    await db_pool.close()


app = FastAPI(title="Indie Game Feedback Analytics Engine", lifespan=lifespan)


class FeedbackItem(BaseModel):
    source: str
    content: str


class IngestBatchRequest(BaseModel):
    items: List[FeedbackItem]


# ---------------------------------------------------------------------
# Bullet Point 1: Asynchronous Ingestion & Batch Vectorization
# ---------------------------------------------------------------------
@app.post("/api/v1/ingest")
async def ingest_feedback(payload: IngestBatchRequest):
    if not payload.items:
        raise HTTPException(status_code=400, detail="Batch is empty.")

    start_time = time.perf_counter()
    texts = [item.content for item in payload.items]

    # Vectorize in mini-batches to optimize CPU throughput
    embeddings = model.encode(texts, convert_to_numpy=True, batch_size=64)

    records = [
        {"source": item.source, "content": item.content, "embedding": embeddings[i].tolist()}
        for i, item in enumerate(payload.items)
    ]

    # Bulk insert using Postgres JSON casting to avoid N individual INSERT statements
    insert_sql = """
    INSERT INTO feedback (source, content, embedding)
    SELECT r.source, r.content, r.embedding::vector
    FROM json_to_recordset($1::json) AS r(source text, content text, embedding real[])
    """

    async with db_pool.acquire() as conn:
        await conn.execute(insert_sql, json.dumps(records))

    duration_ms = (time.perf_counter() - start_time) * 1000

    return {
        "status": "success",
        "inserted_count": len(records),
        "pipeline_latency_ms": round(duration_ms, 2)
    }


# ---------------------------------------------------------------------
# Bullet Point 3: HNSW Cosine Search (Sub-40ms semantic retrieval)
# ---------------------------------------------------------------------
@app.get("/api/v1/search")
async def search_feedback(
    query: str = Query(..., description="Semantic search query"),
    limit: int = 5
):
    start_time = time.perf_counter()

    # Vectorize search query into 384 dimensions
    query_vector = model.encode([query], convert_to_numpy=True)[0].tolist()

    # pgvector '<=>' operator invokes HNSW cosine graph traversal
    sql = """
    SELECT 
        id, 
        source, 
        content, 
        1 - (embedding <=> $1::vector) AS similarity
    FROM feedback
    ORDER BY embedding <=> $1::vector ASC
    LIMIT $2;
    """

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(sql, str(query_vector), limit)

    latency_ms = (time.perf_counter() - start_time) * 1000

    return {
        "query": query,
        "search_latency_ms": round(latency_ms, 2),
        "results": [
            {
                "id": r["id"],
                "source": r["source"],
                "content": r["content"],
                "similarity": round(float(r["similarity"]), 4)
            }
            for r in rows
        ]
    }


import json

# ---------------------------------------------------------------------
# Bullet Point 2: UMAP Reduction + HDBSCAN Unsupervised Clustering
# ---------------------------------------------------------------------
@app.post("/api/v1/cluster")
async def cluster_feedback(min_cluster_size: int = 2):
    # 1. Cast embedding to text so PostgreSQL outputs a standard string representation
    sql = "SELECT id, content, embedding::text AS embedding FROM feedback;"
    
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(sql)

    if len(rows) < min_cluster_size + 2:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {min_cluster_size + 2} samples for clustering. Found {len(rows)}."
        )

    ids = [r["id"] for r in rows]
    texts = [r["content"] for r in rows]

    # 2. Parse vectors using json.loads -> guarantees a clean (N, 384) float32 matrix
    vectors = np.array([json.loads(r["embedding"]) for r in rows], dtype=np.float32)

    # 3. UMAP Dimensionality Reduction
    n_samples = len(vectors)
    if n_samples >= 15:
        reducer = umap.UMAP(
            n_neighbors=15,
            n_components=5,
            metric="cosine",
            random_state=42
        )
        reduced_embeddings = reducer.fit_transform(vectors)
    else:
        # Small-batch safety: init="random" prevents spectral failure on tiny test sets
        reducer = umap.UMAP(
            n_neighbors=max(2, min(5, n_samples - 1)),
            n_components=min(2, n_samples - 2),
            metric="cosine",
            init="random",
            random_state=42
        )
        reduced_embeddings = reducer.fit_transform(vectors)

    # 4. HDBSCAN Density Clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom"
    )
    labels = clusterer.fit_predict(reduced_embeddings)

    clusters = {}
    noise_count = 0

    for doc_id, text, label in zip(ids, texts, labels):
        label_int = int(label)
        if label_int == -1:
            noise_count += 1
            continue
        if label_int not in clusters:
            clusters[label_int] = []
        clusters[label_int].append({"id": doc_id, "content": text})

    return {
        "total_records": len(ids),
        "noise_records_filtered": noise_count,
        "actionable_clusters_discovered": len(clusters),
        "clusters": [
            {
                "cluster_id": cid,
                "cluster_size": len(items),
                "sample_feedback": items[:3]
            }
            for cid, items in clusters.items()
        ]
    }