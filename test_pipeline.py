import time
import httpx

BASE_URL = "http://127.0.0.1:8000"

FEEDBACK_DATA = [
    # Crash / Technical Bug Cluster
    {"source": "steam", "content": "Fatal game crash occurs immediately on the Stage 3 boss transition."},
    {"source": "discord", "content": "Desktop freeze when Stage 3 boss enters phase two."},
    {"source": "steam", "content": "Game freezes completely during stage 3 boss fight, requires task manager kill."},
    {"source": "discord", "content": "Level 3 boss causes an instant memory leak crash to desktop."},

    # Balance / Combat Cluster
    {"source": "steam", "content": "Swords attack animation is far too slow compared to daggers."},
    {"source": "discord", "content": "Two-handed sword attack speed needs an immediate buff in PvP."},
    {"source": "steam", "content": "Melee sword balance feels sluggish and underpowered."},

    # Noise / Unrelated Chatter
    {"source": "steam", "content": "The soundtrack is an absolute masterpiece."},
    {"source": "discord", "content": "Anyone know what time the update drops tomorrow?"}
]


def run_verification():
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        print("[1/3] Testing Batch Ingestion Endpoint...")
        res = client.post("/api/v1/ingest", json={"items": FEEDBACK_DATA})
        print("Ingestion Result:", res.json())
        assert res.status_code == 200

        print("\n[2/3] Testing HNSW Cosine Search Latency...")
        query = "boss freezing my PC"
        res = client.get(f"/api/v1/search?query={query}&limit=3")
        search_data = res.json()
        print(f"Query: '{query}'")
        print(f"Retrieval Latency: {search_data['search_latency_ms']} ms")
        for match in search_data["results"]:
            print(f" -> Similarity {match['similarity']}: {match['content']}")
        assert res.status_code == 200

        print("\n[3/3] Testing UMAP + HDBSCAN Unsupervised Clustering...")
        print("\n[3/3] Testing UMAP + HDBSCAN Unsupervised Clustering...")
        res = client.post("/api/v1/cluster?min_cluster_size=2")
        if res.status_code != 200:
            print("Cluster endpoint error:", res.status_code, res.text)
            return
        cluster_data = res.json()
        print(f"Total Records: {cluster_data['total_records']}")
        print(f"Noise Pruned: {cluster_data['noise_records_filtered']}")
        print(f"Clusters Discovered: {cluster_data['actionable_clusters_discovered']}")
        for c in cluster_data["clusters"]:
            print(f"\nCluster ID {c['cluster_id']} (Size: {c['cluster_size']}):")
            for sample in c["sample_feedback"]:
                print(f"   - {sample['content']}")
        assert res.status_code == 200

    print("\n✓ Pipeline verification completed successfully.")


if __name__ == "__main__":
    run_verification()