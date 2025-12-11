## Hands-on Demo

You can try the Assessment Recommendation System live without installing anything.

---

### **1. Web UI (Streamlit Frontend)**  
Interact with the recommendation system directly in your browser:

**Streamlit App:**  
https://sohamghosh1509-assessment-re-shl-genai-recommendersrcapp-qxsum6.streamlit.app/

You can:
- Paste a Job Description or natural language query  
- Choose how many assessments you want (Top-K)  
- See recommended assessments with descriptions, duration, test type, etc.

---

### **2. REST API (FastAPI Backend)**  
The backend is deployed on Render and exposes two endpoints: GET /health and POST /recommend

**Render URL:**  
[https://shl-genai-api.onrender.com
](https://dashboard.render.com/web/srv-d4hienjuibrs73dkurv0/deploys/dep-d4hil98dl3ps73cgdb10?r=2025-11-23%4015%3A38%3A18%7E2025-11-23%4015%3A42%3A35)
