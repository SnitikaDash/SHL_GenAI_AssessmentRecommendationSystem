import os
import logging
import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load pickled files
try:
    with open("df.pkl", "rb") as f:
        df = pickle.load(f)
    with open("X.pkl", "rb") as f:
        X = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    logger.info("✅ Pickle files loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading pickle files: {e}")
    raise RuntimeError("Failed to load pickle files.")

# Health check route
@app.get("/health")
def health():
    return {"status": "ok"}

# Input model for /recommend
class QueryModel(BaseModel):
    query: str

# Recommendation endpoint
@app.post("/recommend")
def recommend(payload: QueryModel):
    query = payload.query
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, X).flatten()
    top_indices = scores.argsort()[::-1][:10]
    results = []

    for idx in top_indices:
        item = df.iloc[idx]
        results.append({
            "Assessment Name": item.get("title"),
            "URL": item.get("url"),
            "Test Type": item.get("test_type"),
            "Duration": item.get("duration_minutes"),
            "Supports Remote Testing": item.get("remote_testing_support"),
            "Adaptive/IRT": item.get("adaptive_irt_support"),
            "Similarity Score": float(scores[idx])
        })

    return {"results": results}
