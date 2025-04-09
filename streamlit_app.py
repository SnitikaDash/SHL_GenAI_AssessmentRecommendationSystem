import streamlit as st
import requests  # ✅ Required for backend API call

# Page setup
st.set_page_config(page_title="SHL Assessment Recommender", layout="centered")
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
        }
        .stTextArea textarea {
            border-radius: 8px;
            font-size: 16px;
        }
        .stButton>button {
            border-radius: 8px;
            background-color: #4CAF50;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)

# Title
st.title("🔍 SHL Assessment Recommendation System")
st.write("Enter a **job description or hiring requirement**, and get the most relevant SHL assessments instantly.")

# Input box
query = st.text_area("📝 Enter Job Description or Hiring Need", height=180, placeholder="e.g., We're hiring a data analyst with strong problem-solving skills...")

# ✅ Backend URL
backend_url = "https://shlgenaiassessmentrecommendati-production.up.railway.app/recommend"

if st.button("🔎 Recommend Assessments"):
    if not query.strip():
        st.warning("⚠️ Please enter a valid job description.")
    else:
        try:
            response = requests.post(backend_url, json={"query": query})

            if response.status_code == 200:
                recommendations = response.json().get("results", [])
                if not recommendations:
                    st.info("No relevant assessments found. Try a different query.")
                else:
                    st.subheader("🎯 Top Recommended Assessments")
                    for row in recommendations:
                        st.markdown(f"### 🔗 [{row.get('Assessment Name', 'Unnamed Assessment')}]({row.get('URL', '#')})")
                        st.markdown(f"- **Duration**: {row.get('Duration', 'N/A')}  \n"
                                    f"- **Remote Test**: {row.get('Remote Test Support', 'No')}  \n"
                                    f"- **Adaptive/IRT**: {row.get('Adaptive/IRT', 'No')}  \n"
                                    f"- **Type**: {row.get('Test Type', 'N/A')}")
            else:
                st.error(f"🚫 Error: {response.status_code} - Could not fetch recommendations.")
        except Exception as e:
            st.error(f"Something went wrong while contacting the backend: {e}")

st.markdown("</div>", unsafe_allow_html=True)
