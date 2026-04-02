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

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Signaturesi Neo L1.0 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clients
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    groq_client = Groq()
except Exception as e:
    logger.error(f"Init Error: {e}")
    raise RuntimeError("Infra failure")

# Optimized System Prompt for Token Saving
SYSTEM_PROMPT = (
    "Mode: Concise Thinker. Logic: Essential only. "
    "Reasoning: Be brief, avoid repetition. "
    "Output: Max 2-3 sentences or short bullets. No filler. "
    "Knowledge cutoff: July 2025."
)

GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]

def extract_answer_from_reasoning(reasoning: str) -> str:
    if not reasoning: return ""
    patterns = [
        r"(?:So|Therefore|Thus),?\s*(?:we )?answer:?\s*(.+?)(?:\n\n|$)",
        r"Final (?:answer|output):?\s*(.+?)(?:\n\n|$)",
        r"Output:?\s*(.+?)(?:\n\n|$)",
    ]
    for pat in patterns:
        m = re.search(pat, reasoning, re.IGNORECASE | re.DOTALL)
        if m: return m.group(1).strip()
    return reasoning.split('\n')[-1] if len(reasoning.split('\n')[-1]) > 10 else reasoning[:200]

async def call_groq_with_fallback(messages: List[Dict[str, str]]):
    for model in GROQ_MODELS:
        try:
            completion = groq_client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=0.6,
                max_completion_tokens=1024, # Cap tokens to save cost
                reasoning_effort="low",      # CONTROL: Low effort = less tokens
                stream=False
            )
            return completion, model
        except Exception as e:
            logger.warning(f"Model {model} failed: {e}")
            continue
    raise HTTPException(500, "AI engines exhausted")

@app.get("/v1/user/balance")
def get_balance(api_key: str):
    resp = supabase.table("users").select("token_balance").eq("api_key", api_key).execute()
    if not resp.data: raise HTTPException(404, "Invalid Key")
    return {"balance": resp.data[0]["token_balance"]}

@app.post("/v1/chat/completions")
async def chat_proxy(request: Request, authorization: str = Header(None)):
    if not authorization: raise HTTPException(401)
    user_api_key = authorization.split(" ")[1]
    body = await request.json()
    
    # Balance check
    resp = supabase.table("users").select("token_balance").eq("api_key", user_api_key).execute()
    if not resp.data or resp.data[0]["token_balance"] <= 0:
        raise HTTPException(402, "No balance")

    # AI Call
    payload = [{"role": "system", "content": SYSTEM_PROMPT}] + body.get("messages", [])
    ai_res, used_model = await call_groq_with_fallback(payload)

    msg_obj = ai_res.choices[0].message
    raw_content = msg_obj.content or ""
    raw_reasoning = getattr(msg_obj, 'reasoning', "")

    if not raw_content.strip() and raw_reasoning:
        raw_content = extract_answer_from_reasoning(raw_reasoning)

    # Token update
    tokens_used = ai_res.usage.completion_tokens
    new_bal = max(0, resp.data[0]["token_balance"] - tokens_used)
    supabase.table("users").update({"token_balance": new_bal}).eq("api_key", user_api_key).execute()

    return {
        "message": raw_content.strip(),
        "reasoning": raw_reasoning.strip(),
        "usage": {"completion_tokens": tokens_used},
        "model": used_model
    }
