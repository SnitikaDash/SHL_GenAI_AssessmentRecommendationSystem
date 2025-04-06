import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SHL Assessment Recommender", layout="centered")

st.title("🔍 SHL Assessment Recommendation System")
st.write("Enter a job description or hiring need to get recommended SHL assessments.")

# Load CSV
try:
    df = pd.read_csv("shl_assessments.csv")
    st.success("✅ CSV loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load CSV: {e}")
    st.stop()

# Show column names
st.write("**CSV Columns:**", df.columns.tolist())

# Clean up missing values
df["title"] = df["title"].fillna("").astype(str)
df["description"] = df["description"].fillna("").astype(str)
df["combined"] = (df["title"] + " " + df["description"]).str.strip()

# Show first few combined rows for debugging
st.write("**First 5 Combined Entries:**", df["combined"].head())

# Remove empty combined rows
df = df[df["combined"].str.strip() != ""]

# Final check before vectorizing
if df["combined"].empty:
    st.error("❌ All 'combined' text fields are empty. Please check your CSV content.")
    st.stop()

# Vectorization
try:
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["combined"])
    st.success("✅ TF-IDF vectorization successful.")
except Exception as e:
    st.error(f"❌ Vectorization failed: {e}")
    st.stop()

# Input
query = st.text_area("📝 Enter Job Description / Hiring Requirement:")

def get_recommendations(query, df, X, vectorizer, top_n=5):
    if not query.strip():
        return pd.DataFrame()
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, X).flatten()
    top_indices = similarity.argsort()[::-1][:top_n]
    return df.iloc[top_indices].copy()

# Button
if st.button("🔍 Get Recommendations"):
    results = get_recommendations(query, df, X, vectorizer)
    if results.empty:
        st.warning("No results found.")
    else:
        st.markdown("## 🎯 Top Recommended Assessments")
        for _, row in results.iterrows():
            st.markdown(f"### [{row['title']}]({row['url']})" if pd.notna(row['url']) else f"### {row['title']}")
            st.markdown(f"**Test Type:** {row.get('test_type', 'N/A')}")
            st.markdown(f"**Duration:** {row.get('duration_minutes', 'N/A')} mins")
            st.markdown(f"**Supports Remote Testing:** {row.get('remote_testing_support', 'N/A')}")
            st.markdown(f"**Adaptive/IRT:** {row.get('adaptive_irt_support', 'N/A')}")
            st.markdown("---")
