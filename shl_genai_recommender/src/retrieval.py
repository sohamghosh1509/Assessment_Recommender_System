import os
import faiss
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# ------------------------------
# 1. Setup
# ------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-large"
INDEX_FILE = "models/faiss_index.bin"
ASSESSMENT_FILE = "data/processed/assessments_with_embeddings.csv"

# Load FAISS index and assessment metadata
index = faiss.read_index(INDEX_FILE)
assessments_df = pd.read_csv(ASSESSMENT_FILE)

print(f"Loaded FAISS index with {index.ntotal} vectors.")
print(f"Loaded {len(assessments_df)} assessments with metadata.")


# ------------------------------
# 2. Helper: Generate query embedding
# ------------------------------
def get_query_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)


# ------------------------------
# 3. Retrieve top-K recommendations
# ------------------------------
def recommend_assessments(query: str, top_k: int = 10):
    """Return top-K most relevant SHL assessments for a given query."""
    query_emb = get_query_embedding(query)

    # Perform similarity search (lower L2 = more similar)
    distances, indices = index.search(query_emb, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        row = assessments_df.iloc[idx]
        similarity = 1 / (1 + float(distances[0][rank]))  # normalize to 0–1

        results.append({
            "rank": rank + 1,
            "assessment_name": row.get("title", "Unknown"),
            "url": row["url"],
            # "description": row.get("description", "No description"),
            "adaptive_support": row.get("adaptive_support", "No"),
            "remote_support": row.get("remote_support", "Yes"),
            "duration": int(row.get("duration", 60)),
            "test_type": row.get("test_type", "General Cognitive"),
            "similarity_score": round(similarity, 4)
        })
    return results


# ------------------------------
# 4. Quick test
# ------------------------------
if __name__ == "__main__":
    query = "I am hiring for an analyst and wants applications to screen using Cognitive and personality tests, what options are available within 45 mins."
    recs = recommend_assessments(query, top_k=8)

    print("\nRecommendations:\n")
    for r in recs:
        print(f"{r['rank']}. {r['assessment_name']} -> {r['duration']} "
              f"({r['test_type']}, sim={r['similarity_score']})")
