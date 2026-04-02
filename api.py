import os
import logging
import secrets
import re
import asyncio
from typing import List, Tuple
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# -----------------------------
# Logger
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-L1.1")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Signaturesi Neo L1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ change in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Clients
# -----------------------------
SUPABASE: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# Models (cheap → expensive)
# -----------------------------
MODELS = {
    "fast": "llama-3.1-8b-instant",
    "reason": "openai/gpt-oss-20b",
    "power": "openai/gpt-oss-safeguard-20b"
}

# -----------------------------
# System prompt
# -----------------------------
SYSTEM_PROMPT = """Mode: Neo-Concise.
- Simple → direct answer
- Complex → short reasoning
- Search → only when needed
Max 60 tokens. No fluff."""

# -----------------------------
# Pydantic
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

# -----------------------------
# Smart Query Classifier
# -----------------------------
def classify_query(text: str) -> str:
    text = text.lower()

    if len(text) < 20:
        return "fast"

    if any(x in text for x in ["latest", "today", "2026", "news", "price"]):
        return "search"

    if any(x in text for x in ["why", "how", "explain", "logic"]):
        return "reason"

    return "fast"

# -----------------------------
# Extract best answer
# -----------------------------
def extract_best_content(message_obj) -> str:
    content = getattr(message_obj, 'content', "") or ""
    if content.strip():
        return content.strip()

    reasoning = getattr(message_obj, 'reasoning', "")
    if reasoning:
        lines = [l.strip() for l in reasoning.split('\n') if len(l.strip()) > 10]
        return lines[-1] if lines else reasoning[:200]

    return "No response"

# -----------------------------
# AI Call
# -----------------------------
async def call_ai(messages: List[dict], mode: str):

    model = MODELS["fast"]
    tools = None
    reasoning = None
    max_tokens = 300

    if mode == "reason":
        model = MODELS["reason"]
        reasoning = "medium"
        max_tokens = 600

    elif mode == "search":
        model = MODELS["power"]
        reasoning = "low"
        tools = [{"type": "browser_search"}]
        max_tokens = 800

    try:
        completion = await asyncio.wait_for(
            asyncio.to_thread(
                GROQ.chat.completions.create,
                model=model,
                messages=messages,
                temperature=0.7,
                max_completion_tokens=max_tokens,
                reasoning_effort=reasoning,
                tools=tools,
                stream=False
            ),
            timeout=12
        )

        return completion, model

    except Exception as e:
        logger.error(f"AI Error: {e}")
        raise HTTPException(500, "AI request failed")

# -----------------------------
# Health
# -----------------------------
@app.get("/")
def health():
    return {"status": "online", "version": "Neo L1.1"}

# -----------------------------
# Chat Endpoint
# -----------------------------
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):

    # 🔐 Auth
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key")

    api_key = authorization.replace("Bearer ", "")

    user = SUPABASE.table("users") \
        .select("token_balance") \
        .eq("api_key", api_key) \
        .maybe_single() \
        .execute()

    if not user.data:
        raise HTTPException(401, "Invalid API key")

    balance = user.data["token_balance"]
    if balance <= 0:
        raise HTTPException(402, "Insufficient balance")

    # 🧠 Prepare
    user_text = payload.messages[-1]["content"]
    mode = classify_query(user_text)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + payload.messages

    # 🤖 AI Call
    ai_response, used_model = await call_ai(messages, mode)

    # 🧾 Extract
    final_answer = extract_best_content(ai_response.choices[0].message)

    # 💰 Token usage (IMPORTANT FIX)
    tokens_used = ai_response.usage.total_tokens
    new_balance = max(0, balance - tokens_used)

    # 💾 Update DB (safe)
    await asyncio.to_thread(
        SUPABASE.table("users")
        .update({"token_balance": new_balance})
        .eq("api_key", api_key)
        .execute
    )

    return {
        "id": f"neo_{secrets.token_hex(6)}",
        "message": final_answer,
        "mode": mode,
        "usage": {
            "total_tokens": tokens_used
        },
        "model_engine": "Neo-L1.1",
        "internal_model": used_model
    }
