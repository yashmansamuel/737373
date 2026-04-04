import os
import logging
import secrets
import asyncio
import json
from typing import List
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# -----------------------------
# 1. Setup & Config
# -----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-L1.0-Core")

# Check environment variables
required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing required environment variable: {var}")

app = FastAPI(title="Neo L1.0 Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Single model for Neo
MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"

# -----------------------------
# System Prompt
# -----------------------------
SYSTEM_PROMPT = """Identity: Neo L1.0.
Role: High-Density Reasoning Engine.
STRICT OUTPUT RULES:
1. Return ONLY valid JSON: {"final_answer": "...", "reasoning": "..."}.
2. Keep reasoning concise, ≤50 words.
3. Escape all quotes and newlines properly."""

# -----------------------------
# Pydantic Models
# -----------------------------
class ChatRequest(BaseModel):
    messages: List[dict]

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

# -----------------------------
# Knowledge Engine
# -----------------------------
def get_neo_knowledge(query: str, max_lines: int = 3) -> str:
    try:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge.txt")
        if not os.path.exists(file_path):
            return ""
        words = [w.lower() for w in query.split() if len(w) > 3]
        matches = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if any(w in line.lower() for w in words):
                    matches.append(line.strip())
                if len(matches) >= max_lines:
                    break
        return "\n".join(matches)
    except Exception as e:
        logger.error(f"Knowledge retrieval failed: {e}")
        return ""

# -----------------------------
# Token Utilities
# -----------------------------
async def deduct_tokens(api_key: str, tokens_used: int) -> int:
    def db_update():
        res = SUPABASE.table("users").update({
            "token_balance": f"token_balance - {tokens_used}"
        }).eq("api_key", api_key).gte("token_balance", tokens_used).execute()
        return res.data
    result = await asyncio.to_thread(db_update)
    if not result:
        raise HTTPException(402, "Token limit reached. Please top-up.")
    new_balance = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute().data["token_balance"]
    return new_balance

def get_user(api_key: str):
    return SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()

# -----------------------------
# API Routes
# -----------------------------
@app.get("/")
async def root():
    return {
        "company": "signaturesi.com",
        "engine": "Neo L1.0",
        "status": "running"
    }

@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    user = get_user(api_key)
    if not user.data:
        return {"api_key": api_key, "balance": 0}
    return {"api_key": api_key, "balance": user.data.get("token_balance", 0)}

@app.post("/v1/user/new-key")
def create_key():
    api_key = "sig-" + secrets.token_hex(16)
    SUPABASE.table("users").insert({"api_key": api_key, "token_balance": 1_000_000}).execute()
    return {"api_key": api_key, "plan": "Starter", "token_balance": 1_000_000}

@app.post("/v1/user/top-up")
def top_up(api_key: str, tokens: int = 1_000_000):
    price_per_1M = 12
    price = (tokens / 1_000_000) * price_per_1M
    SUPABASE.table("users").update({
        "token_balance": f"token_balance + {tokens}"
    }).eq("api_key", api_key).execute()
    new_balance = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute().data["token_balance"]
    return {"new_balance": new_balance, "tokens_added": tokens, "price_usd": price}

@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key")
    api_key = authorization.replace("Bearer ", "")
    user = get_user(api_key)
    if not user.data:
        raise HTTPException(401, "User not found")
    balance = user.data["token_balance"]
    if balance <= 0:
        raise HTTPException(402, "No tokens left")

    user_msg = payload.messages[-1].get("content", "") if payload.messages else ""
    context = get_neo_knowledge(user_msg)

    final_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context:
        final_messages.append({"role": "system", "content": f"Local Context:\n{context}"})
    final_messages.extend(payload.messages)

    try:
        response = GROQ.chat.completions.create(
            model=MODEL_ID,
            messages=final_messages,
            temperature=0.5,
            max_tokens=2000
        )

        raw_content = response.choices[0].message.content
        tokens_used = getattr(response.usage, "total_tokens", 0)
        new_balance = await deduct_tokens(api_key, tokens_used)

        # Safe JSON parse
        try:
            data = json.loads(raw_content)
        except:
            data = {"final_answer": raw_content, "reasoning": "Fallback: JSON parse failed."}

        return {
            "company": "signaturesi.com",
            "final_answer": data.get("final_answer"),
            "reasoning": data.get("reasoning"),
            "usage": {"total_tokens": tokens_used},
            "model": "Neo L1.0",
            "internal_engine": MODEL_ID,
            "balance": new_balance
        }

    except Exception as e:
        logger.exception("Engine call failed:")
        raise HTTPException(503, f"Engine error: {str(e)}")
