import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(page_title="SHL Assessment Recommendation System", layout="centered")

# Load CSV file
df = pd.read_csv("shl_assessments.csv")

# Combine title and description for vectorization
df["combined"] = df["title"].fillna('') + " " + df["description"].fillna('')

# Vectorize the combined column
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df["combined"])

# App Title
st.title("🔍 SHL Assessment Recommendation System")
st.write("Enter a job description or hiring need, and get the most relevant SHL assessments.")

# User Input
query = st.text_area("📝 Enter Job Description / Hiring Requirement:")

# Function to get recommendations
def get_recommendations(query, df, X, vectorizer, top_n=5):
    if not query.strip():
        return pd.DataFrame()
    
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, X).flatten()
    top_indices = similarity.argsort()[::-1][:top_n]
    recommendations = df.iloc[top_indices].copy()
    return recommendations

# Recommend and Display
if st.button("🔍 Get Recommendations"):
    results = get_recommendations(query, df, X, vectorizer)

    if results.empty:
        st.warning("Please enter a valid job description.")
    else:
        st.markdown("## 🎯 Top Recommended Assessments")
        for _, row in results.iterrows():
            st.markdown(f"### [{row['title']}]({row['url']})")
            st.markdown(f"**Test Type:** {row['test_type'] if pd.notna(row['test_type']) else 'N/A'}  ")
            st.markdown(f"**Duration:** {str(row['duration_minutes']) + ' mins' if pd.notna(row['duration_minutes']) else 'N/A'}  ")
            st.markdown(f"**Supports Remote Testing:** {row['remote_testing_support'] if pd.notna(row['remote_testing_support']) else 'N/A'}  ")
            st.markdown(f"**Adaptive/IRT:** {row['adaptive_irt_support'] if pd.notna(row['adaptive_irt_support']) else 'N/A'}  ")
            st.markdown("---")
