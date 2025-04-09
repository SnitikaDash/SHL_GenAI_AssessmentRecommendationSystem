from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# CORS settings
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

# Health check
@app.get("/health")
def health():
    return {"status": "ok"}

# Define input model for /recommend
class QueryModel(BaseModel):
    query: str

# Recommend route using JSON body
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
            "Assessment Name": item.get("Assessment Name"),
            "URL": item.get("URL"),
            "Test Type": item.get("Test Type"),
            "Duration": item.get("Duration"),
            "Supports Remote Testing": item.get("Supports Remote Testing"),
            "Adaptive/IRT": item.get("Adaptive/IRT"),
            "Similarity Score": float(scores[idx])
        })

    return {"results": results}
