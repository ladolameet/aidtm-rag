from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
import requests
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# 🔐 Load API Key
# ============================================================
load_dotenv("google_key.env")
API_KEY = os.getenv("GOOGLE_API_KEY")

ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1/models/"
    "gemini-2.0-flash:generateContent"
)
TEMPERATURE = 0.5

# ============================================================
# 🔧 CHANGED — Lazy load CSV instead of loading at startup
# ============================================================
df = None
docs = None

def load_handbook():
    global df, docs
    if df is None:   # only loads once
        df = pd.read_csv("handbook_final.csv")
        docs = [f"Page {row['page']}: {row['chunk']}" for _, row in df.iterrows()]
    return df, docs

# ============================================================
# 🔧 CHANGED — Lazy load embedding model
# ============================================================
embed_model = None

def load_model():
    global embed_model
    if embed_model is None:
        embed_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    return embed_model

# ============================================================
# 🔧 CHANGED — Lazy FAISS creation (saves 200–300MB RAM)
# ============================================================
index = None

def load_faiss():
    global index
    if index is None:
        _, docs = load_handbook()
        model = load_model()

        print("🔧 Building embeddings (first request only)...")
        embeddings = model.encode(docs, normalize_embeddings=True)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(np.array(embeddings))

        print("📚 FAISS Index Ready:", index.ntotal, "chunks loaded")

    return index

# ============================================================
# 🚀 FastAPI Setup + CORS
# ============================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.options("/chat")
async def chat_options():
    return {"status": "ok"}

# ============================================================
# 📩 Input Model
# ============================================================
class Query(BaseModel):
    query: str

# ============================================================
# 🤖 RAG + Gemini Function
# ============================================================
def rag_query_gemini(query, k=4):
    df, docs = load_handbook()          # 🔧 loads only when needed
    model = load_model()                # 🔧 loads only when needed
    index = load_faiss()                # 🔧 loads only when needed

    # 1️⃣ Encode user query
    q_emb = model.encode([query], normalize_embeddings=True)

    # 2️⃣ Search FAISS
    scores, idxs = index.search(np.array(q_emb), k)
    retrieved_chunks = [docs[i] for i in idxs[0]]

    context = "\n".join(retrieved_chunks)

    prompt = f"""
You are an assistant that uses the AIDTM Student Handbook as your primary reference.

Rules:
1. If the handbook context contains the answer, use that information first.
2. If the handbook does NOT contain the answer, then clearly answer using your general knowledge or other reliable information.
3. Do NOT say things like "I cannot answer" or "I can only use the handbook."
4. Always give the user a helpful answer, even when the handbook has no information.
5. If the handbook information is incomplete, combine it with your general knowledge.

Handbook Context:
{context}

User Question:
{query}

Give a clear and correct answer following the rules above.
"""

    # 4️⃣ Call Gemini API
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": TEMPERATURE}
    }

    url = f"{ENDPOINT}?key={API_KEY}"
    response = requests.post(url, headers=headers, json=body)

    if not response.ok:
        return "⚠ API Error", retrieved_chunks

    data = response.json()

    try:
        answer = data["candidates"][0]["content"]["parts"][0]["text"]
    except:
        answer = "⚠ No valid response from Gemini."

    return answer, retrieved_chunks

# ============================================================
# 🌐 CHAT Endpoint (POST + GET)
# ============================================================
@app.api_route("/chat", methods=["GET", "POST"])
async def chat_api(request: Request):
    # GET request support
    if request.method == "GET":
        query = request.query_params.get("query", "")
        if not query:
            return {"error": "Query parameter is missing."}
    else:
        body = await request.json()
        query = body.get("query", "")

    answer, retrieved = rag_query_gemini(query)
    return {"answer": answer, "retrieved": retrieved}
