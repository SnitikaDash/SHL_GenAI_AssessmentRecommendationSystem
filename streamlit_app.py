import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained files
df = pickle.load(open("df.pkl", "rb"))
X = pickle.load(open("X.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Streamlit App
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

st.title("🔍 SHL Assessment Recommendation System")
st.markdown("Enter a job description or hiring need, and get the most relevant SHL assessments.")

# Input text box
query = st.text_area("📝 Enter Job Description / Hiring Requirement:")

def get_recommendations(query, df, X, vectorizer):
    if query.strip() == "":
        st.warning("Please enter a valid job description.")
    else:
        # Vectorize the input query
        query_vec = vectorizer.transform([query])

        # Compute cosine similarity
        similarity_scores = cosine_similarity(query_vec, X).flatten()
        top_indices = similarity_scores.argsort()[-10:][::-1]  # Top 10

        # Prepare results
        recommendations = df.iloc[top_indices].copy()
        recommendations["Similarity Score"] = similarity_scores[top_indices]

        return recommendations

# Show results when the button is clicked
if st.button("🔎 Recommend Assessments"):
    recommendations = get_recommendations(query, df, X, vectorizer)
    
    if recommendations.empty:
        st.warning("No relevant assessments found.")
    else:
        st.subheader("🎯 Top Recommended Assessments")
        for idx, row in recommendations.iterrows():
            st.markdown(f"### [{row['Assessment Name']}]({row['URL']})")
            st.write(f"**Test Type:** {row.get('Test Type', 'N/A')}")
            st.write(f"**Duration:** {row.get('Duration', 'N/A')}")
            st.write(f"**Supports Remote Testing:** {row.get('Supports Remote Testing', 'N/A')}")
            st.write(f"**Adaptive/IRT:** {row.get('Adaptive/IRT', 'N/A')}")
            st.progress(min(row['Similarity Score'], 1.0))
            st.markdown("---")
