import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained files
df = pickle.load(open("df.pkl", "rb"))
X = pickle.load(open("X.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="SHL Assessment Recommender", layout="centered")

# Custom background and glassmorphism effect
st.markdown("""
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1606760227091-124d7a5404dc?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .glass-box {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        color: #000;
        max-width: 800px;
        margin: 2rem auto;
    }
    .stTextArea textarea {
        border-radius: 0.5rem;
        font-size: 1rem;
    }
    .stButton>button {
        border-radius: 0.5rem;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        font-size: 16px;
        padding: 0.6rem 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='glass-box'>", unsafe_allow_html=True)

st.title("🔍 SHL Assessment Recommendation System")
st.markdown("Enter your **job description or hiring need**, and we’ll recommend the most relevant SHL assessments.")

query = st.text_area("📝 Job Description / Hiring Requirement", height=180, placeholder="e.g., Looking for a software engineer with strong analytical thinking...")

def get_recommendations(query, df, X, vectorizer):
    query_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, X).flatten()

    top_indices = similarity_scores.argsort()[::-1]
    top_indices = [i for i in top_indices if i < len(df)][:10]

    recommendations = df.iloc[top_indices].copy()
    recommendations["Similarity Score"] = similarity_scores[top_indices]
    return recommendations

if st.button("🚀 Recommend Assessments"):
    if not query.strip():
        st.warning("⚠️ Please enter a valid job description.")
    else:
        try:
            recommendations = get_recommendations(query, df, X, vectorizer)
            if recommendations.empty:
                st.info("No relevant assessments found. Try rewording your query.")
            else:
                st.subheader("🎯 Top SHL Assessment Matches")
                for _, row in recommendations.iterrows():
                    st.markdown(f"### 🔗 [{row.get('Assessment Name', 'Unnamed Assessment')}]({row.get('URL', '#')})")
                    st.markdown(f"- **🧪 Test Type:** {row.get('Test Type', 'N/A')}")
                    st.markdown(f"- **⏱ Duration:** {row.get('Duration', 'N/A')} mins")
                    st.markdown(f"- **💻 Remote Testing:** {row.get('Supports Remote Testing', 'N/A')}")
                    st.markdown(f"- **🧠 Adaptive/IRT:** {row.get('Adaptive/IRT', 'N/A')}")
                    st.progress(min(row.get("Similarity Score", 0), 1.0))
                    st.markdown("---")
        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")

st.markdown("</div>", unsafe_allow_html=True)
