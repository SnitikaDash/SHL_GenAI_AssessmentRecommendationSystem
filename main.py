from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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

# ✅ Load the CSV file
df = pd.read_csv("shl_sample_100_assessments.csv")

# ✅ Combine title and description into a single text field for vectorization
df["combined_text"] = df["title"].fillna('') + " " + df["description"].fillna('')

# ✅ Create vectorizer and transform combined text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["combined_text"])

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
