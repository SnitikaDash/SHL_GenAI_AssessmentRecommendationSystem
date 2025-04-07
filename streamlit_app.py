import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

# Stylish CSS for background + transparent container
st.markdown("""
    <style>
    /* Full-page background */
    body {
        background-image: url('https://images.unsplash.com/photo-1522075469751-3a6694fb2f61');
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Main content container */
    .reportview-container .main {
        background-color: rgba(255, 255, 255, 0.88);
        padding: 2rem;
        border-radius: 15px;
        margin-top: 40px;
    }

    /* Heading colors */
    h1, h2, h3 {
        color: #0b2545;
    }

    /* Button styling */
    .stButton button {
        background-color: #1a73e8;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("📘 SHL Assessment Recommendation System")
st.markdown("Enter a job description or hiring need, and get the most relevant SHL assessments.")

# Load saved files
try:
    df = pickle.load(open("df.pkl", "rb"))
    X = pickle.load(open("X.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    st.success("✅ Assessment data loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load files: {e}")
    st.stop()

# Input
query = st.text_area("📝 Enter Job Description / Hiring Requirement:")

# Recommendation logic
def get_recommendations(query, df, X, vectorizer):
    query_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, X).flatten()
    top_indices = similarity_scores.argsort()[::-1]
    top_indices = [i for i in top_indices if i < len(df)][:10]
    recommendations = df.iloc[top_indices].copy()
    recommendations["Similarity Score"] = similarity_scores[top_indices]
    return recommendations

# Button and output
if st.button("🔎 Recommend Assessments"):
    if not query.strip():
        st.warning("Please enter a valid job description.")
    else:
        try:
            recommendations = get_recommendations(query, df, X, vectorizer)
            if recommendations.empty:
                st.info("No relevant assessments found. Please try a different query.")
            else:
                st.subheader("🎯 Top Recommended Assessments")
                for _, row in recommendations.iterrows():
                    st.markdown(f"### [{row.get('Assessment Name', 'Unnamed Assessment')}]({row.get('URL', '#')})")
                    st.write(f"**Test Type:** {row.get('Test Type', 'N/A')}")
                    st.write(f"**Duration:** {row.get('Duration', 'N/A')}")
                    st.write(f"**Supports Remote Testing:** {row.get('Supports Remote Testing', 'N/A')}")
                    st.write(f"**Adaptive/IRT:** {row.get('Adaptive/IRT', 'N/A')}")
                    st.progress(min(row.get("Similarity Score", 0), 1.0))
                    st.markdown("---")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
