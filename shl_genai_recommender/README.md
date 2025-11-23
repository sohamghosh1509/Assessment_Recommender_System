## Overview

This project implements an intelligent Assessment Recommendation System that takes a job description or query and returns relevant SHL assessments from the SHL product catalog.
The system includes:

FastAPI backend (LLM-powered retrieval engine)
FAISS vector search index for fast similarity retrieval
Streamlit frontend for interactive use

## Features

Input: free-text query or job description
Output: top 1–10 SHL assessments with:

Assessment URL
Name
Description
Duration
Adaptive support
Remote support
Test type (Knowledge & Skills / Personality & Behavior)

Uses OpenAI embeddings + FAISS for similarity search
Deployed backend + frontend

## Project Structure
shl_genai_recommender/
├── data/
│   ├── raw/                      # Provided dataset
│   └── processed/                # Cleaned + enriched data
│
├── models/
│   └── faiss_index.bin           # FAISS vector index
│
├── src/
│   ├── data_prep.py              # Step 1: Clean dataset
│   ├── embedding_generation.py   # Step 2: Scrape + embed + build FAISS
│   ├── retrieval_engine.py       # Step 3: Top-K semantic search
│   ├── api_server.py             # Step 4: FastAPI service
│   └── frontend_app.py           # Step 5: Streamlit UI
│
├── outputs/
│   ├── predictions.csv           # SHL submission file
│   └── approach_document.pdf     # 2-page writeup
│
├── requirements.txt
└── README.md
