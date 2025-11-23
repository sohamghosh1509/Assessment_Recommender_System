import streamlit as st
import requests
import pandas as pd

# ------------------------------
# 1. Configuration
# ------------------------------
# Change this URL to your deployed API if hosted on Render, Railway, etc.
API_BASE_URL = "http://127.0.0.1:8000"
# API_BASE_URL = "https://shl-genai-api.onrender.com"


st.set_page_config(page_title="SHL Assessment Recommendation", page_icon="🧠", layout="wide")
st.title("🧠 SHL Assessment Recommendation System")
st.markdown(
    """
    Enter a **job description** or **natural language query** below.  
    The system will recommend relevant SHL assessments using your deployed GenAI API.
    """
)

# ------------------------------
# 2. Input Section
# ------------------------------
query = st.text_area(
    "Enter Job Description or Query:",
    placeholder="e.g. Hiring a Java developer with strong communication skills",
    height=200,
)

top_k = st.slider("Number of Recommendations", 1, 10, 5)
submit = st.button("Get Recommendations")

# ------------------------------
# 3. When user clicks "Get Recommendations"
# ------------------------------
if submit:
    if not query.strip():
        st.warning("Please enter a job description or query first.")
    else:
        with st.spinner("Fetching recommendations from API..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/recommend",
                    json={"query": query, "top_k": top_k},
                    timeout=60,
                )
                if response.status_code == 200:
                    data = response.json()
                    recs = data.get("recommended_assessments", [])

                    if len(recs) == 0:
                        st.info("No recommendations found for this query.")
                    else:
                        # Convert to DataFrame for clean table display
                        df = pd.DataFrame(recs)
                        df.rename(
                            columns={
                                "url": "Assessment URL",
                                "name": "Name",
                                "adaptive_support": "Adaptive",
                                "description": "Description",
                                "duration": "Duration (min)",
                                "remote_support": "Remote",
                                "test_type": "Test Type",
                            },
                            inplace=True,
                        )

                        # Show results in table
                        st.success(f"Found {len(df)} recommendations!")
                        st.dataframe(df, use_container_width=True)

                        # Optionally show clickable links
                        st.markdown("### Assessment Links")
                        for _, row in df.iterrows():
                            st.markdown(f"- [{row['Name']}]({row['Assessment URL']})")

                else:
                    st.error(f"API returned error: {response.status_code}")
                    st.json(response.json())

            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to API: {str(e)}")

# ------------------------------
# 4. Footer
# ------------------------------
st.markdown("---")
st.caption("Built for SHL GenAI Internship Assignment • Powered by FastAPI + Streamlit")
