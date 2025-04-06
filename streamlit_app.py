import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data and vectorizer
df = pd.read_pickle("df.pkl")
X = pickle.load(open("X.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Show columns for debugging (you can remove this line later)
st.write("🧾 Columns in DataFrame:", df.columns.tolist())

# Function to get recommendations
def get_recommendations(query, df, X, vectorizer, top_n=5):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, X).flatten()
    st.write("Similarity Scores:", similarity)

    top_indices = similarity.argsort()[::-1][:top_n]
    st.write("Top Indices:", top_indices)

    # Avoid index errors
    top_indices = [i for i in top_indices if i < len(df)]
    recommendations = df.iloc[top_indices].copy()
    return recommendations

# Streamlit UI
st.title("🔍 SHL Assessment Recommendation System")
st.markdown("Enter a job description or hiring need, and get the most relevant SHL assessments.")

query = st.text_area("📝 Enter Job Description / Hiring Requirement:")

if query:
    recommendations = get_recommendations(query, df, X, vectorizer)

    st.markdown("## 🎯 Top Recommended Assessments")
    for _, row in recommendations.iterrows():
        name = row.get('Assessment Name', 'Unnamed Assessment')
        url = row.get('URL', '#')
        test_type = row.get('Test Type', 'N/A')
        duration = row.get('Duration', 'N/A')
        remote = row.get('Remote Testing Support', 'N/A')
        adaptive = row.get('Adaptive/IRT', 'N/A')

        st.markdown(f"### [{name}]({url})")
        st.write(f"**Test Type:** {test_type}")
        st.write(f"**Duration:** {duration}")
        st.write(f"**Supports Remote Testing:** {remote}")
        st.write(f"**Adaptive/IRT:** {adaptive}")
        st.markdown("---")

st.write("🧾 Columns in DataFrame:", df.columns.tolist())

