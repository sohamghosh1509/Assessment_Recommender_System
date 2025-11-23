## Overview

This project implements an intelligent Assessment Recommendation System that takes a job description or query and returns relevant assessments from the product catalog.

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

