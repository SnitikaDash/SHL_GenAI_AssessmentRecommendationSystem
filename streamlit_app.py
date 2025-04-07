import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Set Streamlit page config
st.set_page_config(
    page_title="SHL Assessment Recommender",
    layout="centered",
    page_icon="🔍"
)

# App Title
st.title("🔍 SHL Assessment Recommendation System")
st.markdown("Welcome! Enter a job description or hiring need, and get the most relevant SHL assessments tailored to your query.")

st.divider()

# Load Pre-trained Files
try:
    df = pickle.load(open("df.pkl", "rb"))
    X = pickle.load(open("X.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    st.success("✅ Assessment data loaded successfully.")
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# Input Section
st.markdown("### 📝 Enter Job Description or Hiring Requirement:")
query = st.text_area("Describe the role, skills, or needs for which you want assessment recommendations:")

# Recommendation Logic
def get_recommendations(query, df, X, vectorizer):
    query_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, X).flatten()

    top_indices = similarity_scores.argsort()[::-1]
    top_indices = [i for i in top_indices if i < len(df)][:10]

    recommendations = df.iloc[top_indices].copy()
    recommendations["Similarity Score"] = similarity_scores[top_indices]
    return recommendations

# Button to Trigger Recommendations
if st.button("🔎 Recommend Assessments"):
    if not query.strip():
        st.warning("⚠️ Please enter a valid job description.")
    else:
        try:
            recommendations = get_recommendations(query, df, X, vectorizer)
            if recommendations.empty:
                st.info("No relevant assessments found. Try a different query.")
            else:
                st.subheader("🎯 Top Recommended Assessments")
                for _, row in recommendations.iterrows():
                    st.markdown(f"### [{row.get('Assessment Name', 'Unnamed Assessment')}]({row.get('URL', '#')})")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"🧪 **Test Type:** {row.get('Test Type', 'N/A')}")
                        st.markdown(f"⏱️ **Duration:** {row.get('Duration', 'N/A')} minutes")
                    with col2:
                        st.markdown(f"🌐 **Remote Testing:** {row.get('Supports Remote Testing', 'N/A')}")
                        st.markdown(f"📊 **Adaptive/IRT:** {row.get('Adaptive/IRT', 'N/A')}")
                    st.progress(min(row.get("Similarity Score", 0), 1.0))
                    st.markdown("---")
        except Exception as e:
            st.error(f"❌ Something went wrong while generating recommendations: {e}")
