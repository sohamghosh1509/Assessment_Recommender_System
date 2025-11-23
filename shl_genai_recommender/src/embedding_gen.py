import os
import time
import requests
import pandas as pd
import numpy as np
import faiss
from bs4 import BeautifulSoup
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# ------------------------------
# 1. Setup
# ------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ASSESSMENT_FILE = "data/processed/unique_assessments.csv"
OUTPUT_FILE = "data/processed/assessments_with_embeddings.csv"
FAISS_INDEX_FILE = "models/faiss_index.bin"
os.makedirs("models", exist_ok=True)

assessments_df = pd.read_csv(ASSESSMENT_FILE)


# ------------------------------
# 2. Enhanced Scraper
# ------------------------------
def scrape_shl_assessment(url: str) -> dict:
    """Scrape SHL assessment metadata with extended fields."""
    try:
        res = requests.get(url, timeout=12)
        if res.status_code != 200:
            return {"url": url}

        soup = BeautifulSoup(res.text, "html.parser")

        # --- Title ---
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else None

        # --- Description ---
        desc_tag = soup.find("p")
        desc = desc_tag.get_text(strip=True) if desc_tag else ""

        # --- Duration ---
        duration = 60
        duration_tag = soup.find(string=lambda x: x and "minute" in x.lower())
        if duration_tag:
            try:
                duration = int("".join([c for c in duration_tag if c.isdigit()]) or 60)
            except:
                duration = 60

        # --- Adaptive & Remote Support ---
        text_content = soup.get_text(" ").lower()
        adaptive_support = "yes" if "adaptive" in text_content else "no"
        remote_support = "yes" if "remote" in text_content or "online" in text_content else "no"

        # --- Test Type ---
        # test_type = []
        # if "personality" in text_content or "behavior" in text_content:
        #     test_type.append("Personality & Behavior")
        # if "technical" in text_content or "skill" in text_content or "knowledge" in text_content:
        #     test_type.append("Knowledge & Skills")
        # if not test_type:
        #     test_type = ["General Cognitive"]
        test_type = ["Other"]
        try:
            prompt = f"""Classify the following SHL assessment into one or more of these types:
            ['Knowledge & Skills', 'Personality & Behavior', 'Cognitive Ability', 'Other'].
            Assessment name: {title}
            Description: {desc}
            Return only a comma-separated list of categories."""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            raw_output = response.choices[0].message.content.strip()
            test_type = [t.strip() for t in raw_output.replace("[", "").replace("]", "").split(",") if t.strip()]
        
        except Exception as e:
            print(f"⚠️ GPT classification failed for {title}: {e}")
            test_type = ["Other"]

        return {
            "url": url,
            "title": title,
            "description": desc,
            "duration": duration,
            "adaptive_support": adaptive_support.title(),
            "remote_support": remote_support.title(),
            "test_type": ", ".join(test_type),
        }
        

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return {"url": url}


# ------------------------------
# 3. Run Scraper
# ------------------------------
print("Scraping SHL assessment metadata (enhanced)...")
scraped_data = [scrape_shl_assessment(url) for url in tqdm(assessments_df["Assessment_url"])]
scraped_df = pd.DataFrame(scraped_data)


# ------------------------------
# 4. Generate Embeddings
# ------------------------------
def get_embedding(text: str):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.zeros(3072, dtype=np.float32)
    for _ in range(3):
        try:
            resp = client.embeddings.create(input=text, model="text-embedding-3-large")
            return np.array(resp.data[0].embedding, dtype=np.float32)
        except Exception as e:
            print(f"⚠️ Retry after error: {e}")
            time.sleep(2)
    return np.zeros(3072, dtype=np.float32)


print("Generating embeddings ...")
scraped_df["combined_text"] = scraped_df.apply(
    lambda x: f"{x['title']} {x['description']} {x['test_type']}", axis=1
)

embeddings = []
for text in tqdm(scraped_df["combined_text"], desc="Embedding generation"):
    embeddings.append(get_embedding(text))

embeddings_array = np.vstack(embeddings)

# ------------------------------
# 5. Build FAISS Index
# ------------------------------
dim = embeddings_array.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings_array)
faiss.write_index(index, FAISS_INDEX_FILE)
print(f"FAISS index saved to {FAISS_INDEX_FILE}")

# ------------------------------
# 6. Save Enriched Dataset
# ------------------------------
scraped_df["embedding_dim"] = [len(e) for e in embeddings]
scraped_df.to_csv(OUTPUT_FILE, index=False)
print(f"Enriched dataset saved to {OUTPUT_FILE}")
print(f"Total assessments processed: {len(scraped_df)}")
