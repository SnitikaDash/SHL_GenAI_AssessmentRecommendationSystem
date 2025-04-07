import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Set page config and background
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

# Set a light background image using HTML/CSS
page_bg_img = '''
<style>
body {
    background-image: url("https://images.unsplash.com/photo-1522075469751-3a6694fb2f61");
    background-size: cover;
    background-attachment: fixed;
    background-repeat: no-repeat;
}
.main {
    background-color: rgba(255, 255, 255, 0.8);
    padding: 2rem;
    border-radius: 12px;
}
h1, h2, h3 {
    color: #1a237e;
}
</style>
<div class="main">
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

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

# User input
query = st.text_area("📝 Enter Job Description / Hiring Requirement:")

def get_recommendations(query, df, X, vectorizer):
    query_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, X).flatten()
    top_indices = similarity_scores.argsort()[::-1]
    top_indices = [i for i in top_indices if i < len(df)][:10]
    recommendations = df.iloc[top_indices].copy()
    recommendations["Similarity Score"] = similarity_scores[top_indices]
    return recommendations

# Recommendation button
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

# Close the main container div
st.markdown("</div>", unsafe_allow_html=True)
