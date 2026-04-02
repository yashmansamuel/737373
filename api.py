import os
import logging
import secrets
import re
import asyncio
from typing import List, Dict, Any

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clients
try:
    supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    logger.error(f"Initialization Error: {e}")

# Configuration
SYSTEM_PROMPT = (
    "Mode: Concise Thinker. Logic: Essential only. "
    "Reasoning: Be brief, no repetition. Output: Max 3 short bullets or 2 sentences. "
    "Knowledge cutoff: July 2025."
)

# Models ordered by reliability for better switching
GROQ_MODELS = [
    "llama-3.3-70b-versatile", 
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b"
]

def extract_answer(reasoning: str) -> str:
    if not reasoning: return ""
    lines = [l.strip() for l in reasoning.split('\n') if l.strip()]
    return lines[-1] if len(lines[-1]) > 10 else reasoning[:200]

async def call_groq_silent_fallback(messages: List[Dict[str, str]]):
    """Bina user ko bataye models switch karne ka logic"""
    for model in GROQ_MODELS:
        for attempt in range(2): # Har model ko 2 baar try karein
            try:
                logger.info(f"Attempting {model}...")
                completion = groq_client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=0.6,
                    max_completion_tokens=1024,
                    reasoning_effort="low", # Token saving
                    stream=False
                )
                return completion, model
            except Exception as e:
                logger.warning(f"{model} failed. Retrying/Switching... Error: {e}")
                await asyncio.sleep(0.5)
                continue
    return None, None

@app.post("/v1/chat/completions")
async def chat_proxy(request: Request, authorization: str = Header(None)):
    if not authorization: raise HTTPException(401)
    api_key = authorization.split(" ")[1]
    body = await request.json()

    # 1. Balance Check
    user = supabase.table("users").select("token_balance").eq("api_key", api_key).execute()
    if not user.data or user.data[0]["token_balance"] <= 0:
        raise HTTPException(402, "Insufficient Balance")

    # 2. AI Call with Silent Fallback
    payload = [{"role": "system", "content": SYSTEM_PROMPT}] + body.get("messages", [])
    ai_res, used_model = await call_groq_silent_fallback(payload)

    if not ai_res:
        raise HTTPException(503, "All engines busy. Please try again.")

    # 3. Data Extraction
    msg = ai_res.choices[0].message
    content = msg.content or ""
    reasoning = getattr(msg, 'reasoning', "")

    if not content.strip() and reasoning:
        content = extract_answer(reasoning)

    # 4. Token Accounting
    tokens_used = ai_res.usage.completion_tokens
    new_bal = max(0, user.data[0]["token_balance"] - tokens_used)
    supabase.table("users").update({"token_balance": new_bal}).eq("api_key", api_key).execute()

    return {
        "message": content.strip(),
        "reasoning": reasoning.strip(),
        "usage": {"completion_tokens": tokens_used},
        "model": used_model
    }

@app.get("/v1/user/balance")
async def balance(api_key: str):
    res = supabase.table("users").select("token_balance").eq("api_key", api_key).execute()
    return {"balance": res.data[0]["token_balance"] if res.data else 0}
