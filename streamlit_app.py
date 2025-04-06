import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load the data
df = pd.read_csv("shl_assessments.csv")

# Combine text fields to vectorize
df["combined"] = df["title"].fillna('') + " " + df["description"].fillna('')

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df["combined"])

def get_recommendations(query, df, X, vectorizer, top_n=5):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, X).flatten()
    
    if similarity.max() == 0:
        return pd.DataFrame()  # No similar assessments found

    top_indices = similarity.argsort()[::-1][:top_n]
    recommendations = df.iloc[top_indices].copy()
    return recommendations

# Streamlit UI
st.title("🔍 SHL Assessment Recommendation System")
st.markdown("Enter a job description or hiring need, and get the most relevant SHL assessments.")

query = st.text_area("📝 Enter Job Description / Hiring Requirement:")

if query:
    recommendations = get_recommendations(query, df, X, vectorizer)
    
    if recommendations.empty:
        st.warning("❌ No matching assessments found. Try a different query.")
    else:
        st.markdown("## 🎯 Top Recommended Assessments")
        for _, row in recommendations.iterrows():
            title = row.get("title", "Unnamed Assessment")
            url = row.get("url", "#")
            duration = row.get("duration_minutes", "N/A")
            remote = row.get("remote_testing_support", "N/A")
            adaptive = row.get("adaptive_irt_support", "N/A")
            test_type = row.get("test_type", "N/A")

            st.markdown(f"### [{title}]({url})")
            st.markdown(f"- **Test Type**: {test_type}")
            st.markdown(f"- **Duration**: {duration} minutes")
            st.markdown(f"- **Supports Remote Testing**: {remote}")
            st.markdown(f"- **Adaptive/IRT**: {adaptive}")
            st.markdown("---")
