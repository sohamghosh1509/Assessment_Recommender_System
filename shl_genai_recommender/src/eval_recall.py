import pandas as pd
import requests

# Local API endpoint
API_URL = "http://127.0.0.1:8000/recommend"

# Load labeled data
train_df = pd.read_csv("data/processed/train_clean.csv")

def recall_at_k(true_items, predicted_items, k=10):
    predicted_top_k = predicted_items[:k]
    intersect = len(set(true_items) & set(predicted_top_k))
    return intersect / len(set(true_items)) if true_items else 0

recalls = []

for i, row in enumerate(train_df.groupby("Query_clean"), start=1):
    query, group = row
    true_urls = group["Assessment_url"].tolist()

    print(f"🔍 Evaluating query {i}/{len(train_df['Query_clean'].unique())}")
    res = requests.post(API_URL, json={"query": query, "top_k": 10}).json()
    pred_urls = [r["url"] for r in res["recommended_assessments"]]

    recall = recall_at_k(true_urls, pred_urls, k=10)
    recalls.append(recall)

mean_recall_10 = sum(recalls) / len(recalls)
print(f"\nMean Recall@10 on training data: {mean_recall_10:.3f}")
