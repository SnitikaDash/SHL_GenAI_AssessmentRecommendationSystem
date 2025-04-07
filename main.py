from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
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

@app.get("/")
def read_root():
    return {"message": "SHL Recommendation API running!"}

@app.get("/recommend")
def recommend(query: str = Query(..., description="Job description or query text")):
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
