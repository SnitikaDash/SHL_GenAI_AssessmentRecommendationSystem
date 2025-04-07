import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained files
df = pickle.load(open("df.pkl", "rb"))
X = pickle.load(open("X.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

st.title("🔍 SHL Assessment Recommendation System")
st.markdown("Enter a job description or hiring need, and get the most relevant SHL assessments.")

query = st.text_area("📝 Enter Job Description / Hiring Requirement:")

def get_recommendations(query, df, X, vectorizer):
    query_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, X).flatten()

    # Get only valid top indices that exist in df
    top_indices = similarity_scores.argsort()[::-1]
    top_indices = [i for i in top_indices if i < len(df)][:10]

    recommendations = df.iloc[top_indices].copy()
    recommendations["Similarity Score"] = similarity_scores[top_indices]
    return recommendations

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
