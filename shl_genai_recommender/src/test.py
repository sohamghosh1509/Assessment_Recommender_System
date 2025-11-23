import pandas as pd
import requests

# ------------------------------
# Configuration
# ------------------------------
API_URL = "http://127.0.0.1:8000/recommend"  # local FastAPI endpoint
TEST_FILE = "data/processed/test_clean.csv"
OUTPUT_FILE = "outputs/predictions.csv"

# ------------------------------
# Load test queries
# ------------------------------
test_df = pd.read_csv(TEST_FILE)

rows = []
for i, query in enumerate(test_df["Query_clean"], start=1):
    print(f"Processing query {i}/{len(test_df)}...")
    try:
        response = requests.post(API_URL, json={"query": query, "top_k": 5}, timeout=60)
        if response.status_code == 200:
            data = response.json()
            for r in data.get("recommended_assessments", []):
                rows.append({
                    "Query": query,
                    "Assessment_url": r["url"]
                })
        else:
            print(f"⚠️ API error {response.status_code} for query {i}")
    except Exception as e:
        print(f"Error for query {i}: {e}")

# ------------------------------
# Save predictions.csv
# ------------------------------
if rows:
    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)
    print(f"predictions.csv saved at {OUTPUT_FILE}")
else:
    print("No data collected. Check API connectivity.")
