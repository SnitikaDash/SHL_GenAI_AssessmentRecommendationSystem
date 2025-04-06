from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# Load pre-saved data and model
df = pickle.load(open("df.pkl", "rb"))
X = pickle.load(open("X.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Create the FastAPI app
app = FastAPI()

# Define input model
class QueryInput(BaseModel):
    query: str
    top_n: int = 10

@app.post("/recommend")
def recommend_assessments(input: QueryInput):
    query_vec = vectorizer.transform([input.query])
    similarity_scores = np.dot(query_vec, X.T).toarray()[0]
    top_indices = similarity_scores.argsort()[-input.top_n:][::-1]
    results = df.iloc[top_indices].copy()
    results["similarity_score"] = similarity_scores[top_indices]
    return results[[
        "Assessment Name", "URL", "Remote Testing Support", 
        "Adaptive/IRT Support", "Duration", "Test Type", "similarity_score"
    ]].to_dict(orient="records")
