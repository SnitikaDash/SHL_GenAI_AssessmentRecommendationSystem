from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data and models
df = pickle.load(open("df.pkl", "rb"))
X = pickle.load(open("X.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Input model
class QueryModel(BaseModel):
    query: str

# POST /recommend endpoint
@app.post("/recommend")
def recommend(query_model: QueryModel):
    query = query_model.query
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, X).flatten()
    top_indices = scores.argsort()[::-1][:10]

    results = []

    for idx in top_indices:
        item = df.iloc[idx]
        results.append({
            "url": item.get("URL"),
            "adaptive_support": item.get("Adaptive/IRT", "No"),
            "description": item.get("Assessment Name", "N/A"),
            "duration": item.get("Duration", 0),
            "remote_support": item.get("Supports Remote Testing", "No"),
            "test_type": item.get("Test Type") if isinstance(item.get("Test Type"), list) else [item.get("Test Type")]
        })

    return {"recommended_assessments": results}
