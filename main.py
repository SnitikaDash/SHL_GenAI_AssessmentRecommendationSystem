from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check route
@app.get("/health")
def health():
    return {"status": "ok"}

# Input model for /recommend
class QueryModel(BaseModel):
    query: str

# Dummy data
dummy_assessments = [
    {
        "Assessment Name": "Cognitive Ability Test",
        "URL": "https://shl.com/cognitive",
        "Test Type": "Cognitive",
        "Duration": 30,
        "Supports Remote Testing": True,
        "Adaptive/IRT": True,
        "Similarity Score": 0.95
    },
    {
        "Assessment Name": "Personality Test",
        "URL": "https://shl.com/personality",
        "Test Type": "Personality",
        "Duration": 25,
        "Supports Remote Testing": True,
        "Adaptive/IRT": False,
        "Similarity Score": 0.88
    },
    # Add more dummy items if needed
]

# Recommendation endpoint
@app.post("/recommend")
def recommend(payload: QueryModel):
    return {"results": dummy_assessments}
