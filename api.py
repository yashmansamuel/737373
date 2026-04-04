import os
import logging
import secrets
import asyncio
from typing import List
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# -----------------------------
# Setup
# -----------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-L1.0-Core")

app = FastAPI(title="Neo L1.0 Engine")

# Supabase client
SUPABASE: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# GROQ client
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Only company model
MODEL_NAME = "openai/gpt-oss-120b"

# Weaponized Prompt
SYSTEM_PROMPT = """Identity: Neo L1.0. Deployment: Jan 1, 2026.
Style: High-Density Reasoning. No filler. Max 2000 tokens.
Use 'Local Context' for all recent facts."""

# -----------------------------
# Models
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

# -----------------------------
# Helpers
# -----------------------------
def extract_content(msg):
    return getattr(msg, "content", "") or "No response"

def get_user(api_key: str):
    return SUPABASE.table("users") \
        .select("token_balance") \
        .eq("api_key", api_key) \
        .maybe_single() \
        .execute()

async def update_balance_async(api_key: str, new_balance: int):
    await asyncio.to_thread(
        SUPABASE.table("users")
        .update({"token_balance": new_balance})
        .eq("api_key", api_key)
        .execute
    )

# -----------------------------
# Root JSON Status
# -----------------------------
@app.get("/")
def root():
    return {"status": "running"}

# -----------------------------
# Chat Endpoint (Company Model Only)
# -----------------------------
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

    final_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    final_messages.extend(payload.messages)

    try:
        response = GROQ.chat.completions.create(
            model=MODEL_NAME,
            messages=final_messages,
            temperature=0.6,
            max_completion_tokens=2000
        )

        reply = extract_content(response.choices[0].message)
        tokens_used = response.usage.total_tokens
        new_balance = max(0, balance - tokens_used)

        asyncio.create_task(update_balance_async(api_key, new_balance))

        return {
            "message": reply,
            "usage": {"total_tokens": tokens_used},
            "model": "Neo L1.0",
            "internal_engine": MODEL_NAME,
            "balance": new_balance
        }

    except Exception as e:
        logger.error(f"{MODEL_NAME} failed: {e}")
        raise HTTPException(503, "Company model failed")
