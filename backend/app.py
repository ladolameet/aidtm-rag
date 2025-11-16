from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import os
import requests
from dotenv import load_dotenv

# ================================================
# Load Environment
# ================================================
load_dotenv("google_key.env")
API_KEY = os.getenv("GOOGLE_API_KEY")

EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedText"
GEN_URL   = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent"

# ================================================
# Load Handbook + Precomputed Embeddings
# ================================================
df = pd.read_csv("handbook_final.csv")   # page | chunk
texts = [f"Page {row['page']}: {row['chunk']}" for _, row in df.iterrows()]

# 🔧 Load small numpy file of precomputed embeddings (<15MB)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(docs, normalize_embeddings=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(np.array(embeddings))

print("Ready with:", len(texts), "chunks")

# ================================================
# FastAPI Setup
# ================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

# ================================================
# Helper — Gemini Embedding API
# ================================================
def get_embedding(text):
    url = f"{EMBED_URL}?key={API_KEY}"
    body = { "text": text }

    res = requests.post(url, json=body)
    emb = res.json()["embedding"]["value"]
    return np.array(emb)

# ================================================
# Vector Search (Cosine Similarity)
# ================================================
def search_similar(query, top_k=4):
    q_emb = get_embedding(query)
    sims = embeddings @ q_emb / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb)
    )
    idxs = sims.argsort()[-top_k:][::-1]
    return [texts[i] for i in idxs]

# ================================================
# RAG + Gemini Answer
# ================================================
def rag_answer(query):
    chunks = search_similar(query)

    context = "\n".join(chunks)

    prompt = f"""
Use the following handbook context first to answer the question.
If context is missing info, use your general knowledge.

Context:
{context}

Question:
{query}

Answer clearly and correctly.
"""

    url = f"{GEN_URL}?key={API_KEY}"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }

    res = requests.post(url, json=body)
    try:
        return res.json()["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return "⚠ Gemini API Error"

# ================================================
# API Endpoint
# ================================================
@app.post("/chat")
async def chat(data: Query):
    answer = rag_answer(data.query)
    return {"answer": answer}