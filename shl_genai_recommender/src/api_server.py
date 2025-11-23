from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI(title="SHL Assessment Recommendation API")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# Paths
EMBEDDING_MODEL = "text-embedding-3-large"
INDEX_PATH = "models/faiss_index.bin"
ASSESSMENT_FILE = "data/processed/assessments_with_embeddings.csv"

# Load resources
index = faiss.read_index(INDEX_PATH)
assessments_df = pd.read_csv(ASSESSMENT_FILE)

print(f"API loaded: {len(assessments_df)} assessments, {index.ntotal} vectors.")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

def get_embedding(text: str) -> np.ndarray:
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    emb = np.array(response.data[0].embedding, dtype=np.float32)
    return emb.reshape(1, -1)

def retrieve_recommendations(query: str, top_k: int = 10):
    """Search FAISS index for top-K relevant SHL assessments."""
    query_emb = get_embedding(query)
    distances, indices = index.search(query_emb, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        row = assessments_df.iloc[idx]
        similarity = 1 / (1 + float(distances[0][rank]))  # normalize 0–1

        # Build each result entry
        rec = {
            "url": row["url"],
            "name": str(row.get("title", "Unknown Assessment")),
            "adaptive_support": str(row.get("adaptive_support", "No")),
            # "description": str(row.get("description", "No description available")),
            "duration": int(row.get("duration", 60)),
            "remote_support": str(row.get("remote_support", "Yes")),
            "test_type": [t.strip() for t in str(row.get("test_type", "General Cognitive")).split(",")],
        }
        results.append(rec)

    return results

@app.get("/health")
def health_check():
    """Simple status check."""
    return {"status": "ok"}


@app.post("/recommend")
def recommend(request: QueryRequest):
    """
    SHL Assessment Recommendation Endpoint.
    Returns results in the exact schema required by SHL evaluators.
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        recommendations = retrieve_recommendations(request.query, request.top_k)
        if not recommendations:
            raise HTTPException(status_code=404, detail="No recommendations found.")

        return {"recommended_assessments": recommendations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# ------------------------------
# 6. Run (development only)
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)