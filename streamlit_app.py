import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set Streamlit page config
st.set_page_config(page_title="SHL Assessment Recommendation System", layout="centered")

# Load CSV
df = pd.read_csv("shl_assessments.csv")

# Safely combine title and description
df["title"] = df["title"].fillna("").astype(str)
df["description"] = df["description"].fillna("").astype(str)
df["combined"] = (df["title"] + " " + df["description"]).str.strip()

# Drop rows where combined text is empty
df = df[df["combined"].str.strip() != ""]

# Vectorize
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["combined"])

# Streamlit UI
st.title("🔍 SHL Assessment Recommendation System")
st.write("Enter a job description or hiring need, and get the most relevant SHL assessments.")

query = st.text_area("📝 Enter Job Description / Hiring Requirement:")

# Recommendation logic
def get_recommendations(query, df, X, vectorizer, top_n=5):
    if not query.strip():
        return pd.DataFrame()
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, X).flatten()
    top_indices = similarity.argsort()[::-1][:top_n]
    return df.iloc[top_indices].copy()

# Handle submission
if st.button("🔍 Get Recommendations"):
    results = get_recommendations(query, df, X, vectorizer)

    if results.empty:
        st.warning("Please enter a valid job description.")
    else:
        st.markdown("## 🎯 Top Recommended Assessments")
        for _, row in results.iterrows():
            st.markdown(f"### [{row['title']}]({row['url']})" if pd.notna(row['url']) else f"### {row['title']}")
            st.markdown(f"**Test Type:** {row['test_type'] if pd.notna(row['test_type']) else 'N/A'}")
            st.markdown(f"**Duration:** {str(row['duration_minutes']) + ' mins' if pd.notna(row['duration_minutes']) else 'N/A'}")
            st.markdown(f"**Supports Remote Testing:** {row['remote_testing_support'] if pd.notna(row['remote_testing_support']) else 'N/A'}")
            st.markdown(f"**Adaptive/IRT:** {row['adaptive_irt_support'] if pd.notna(row['adaptive_irt_support']) else 'N/A'}")
            st.markdown("---")
