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

# Print column names to debug
st.write("Columns in DataFrame:", df.columns)

# Input text box
query = st.text_area("📝 Enter Job Description / Hiring Requirement:")

def get_recommendations(query, df, X, vectorizer):
    if query.strip() == "":
        st.warning("Please enter a valid job description.")
        return pd.DataFrame()  # Return an empty DataFrame if no query is entered
    
    # Vectorize the input query
    query_vec = vectorizer.transform([query])

    # Compute cosine similarity
    similarity_scores = cosine_similarity(query_vec, X).flatten()
    top_indices = similarity_scores.argsort()[-10:][::-1]  # Top 10
    
    # Debugging: Print the similarity scores and top indices
    st.write("Similarity Scores:", similarity_scores)
    st.write("Top Indices:", top_indices)

    # Prepare results
    recommendations = df.iloc[top_indices].copy()
    recommendations["Similarity Score"] = similarity_scores[top_indices]
    
    if recommendations.empty:
        st.write("No recommendations found.")
    else:
        st.subheader("🎯 Top Recommended Assessments")
        for idx, row in recommendations.iterrows():
            # Check if the necessary columns exist
            assessment_name = row.get('Assessment Name', 'N/A')
            url = row.get('URL', 'N/A')
            test_type = row.get('Test Type', 'N/A')
            duration = row.get('Duration', 'N/A')
            remote_testing = row.get('Supports Remote Testing', 'N/A')
            adaptive_irt = row.get('Adaptive/IRT', 'N/A')

            # Display the recommendation
            st.markdown(f"### [{assessment_name}]({url})")
            st.write(f"**Test Type:** {test_type}")
            st.write(f"**Duration:** {duration}")
            st.write(f"**Supports Remote Testing:** {remote_testing}")
            st.write(f"**Adaptive/IRT:** {adaptive_irt}")
            st.progress(min(row['Similarity Score'], 1.0))
            st.markdown("---")

# Get recommendations when button is pressed
if st.button("🔎 Recommend Assessments"):
    recommendations = get_recommendations(query, df, X, vectorizer)
