from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

app = FastAPI()

# ✅ CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Logger for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Try to load the CSV file
try:
    df = pd.read_csv("shl_sample_100_assessments.csv", encoding="utf-8")
    logger.info("✅ CSV loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading CSV: {e}")
    df = pd.DataFrame()  # fallback to empty DataFrame

# ✅ Proceed only if df is not empty
if not df.empty:
    df["combined_text"] = df["title"].fillna('') + " " + df["description"].fillna('')
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["combined_text"])
else:
    vectorizer = None
    X = None

# ✅ Health check route
@app.get("/health")
def health():
    return {"status": "ok"}

# ✅ Input model for /recommend
class QueryModel(BaseModel):
    query: str

# ✅ Recommendation endpoint
@app.post("/recommend")
def recommend(payload: QueryModel):
    if df.empty or vectorizer is None or X is None:
        return {"results": [], "error": "CSV file failed to load or is empty."}

    query = payload.query
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, X).flatten()
    top_indices = scores.argsort()[::-1][:10]
    results = []

    for idx in top_indices:
        item = df.iloc[idx]
        results.append({
            "Assessment Name": item.get("title", ""),
            "URL": item.get("url", ""),
            "Test Type": item.get("test_type", ""),
            "Duration": item.get("duration_minutes", ""),
            "Supports Remote Testing": item.get("remote_testing_support", ""),
            "Adaptive/IRT": item.get("adaptive_irt_support", ""),
            "Similarity Score": float(scores[idx])
        })

    return {"results": results}
