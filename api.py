import os
import re
import logging
import secrets
import asyncio
import time
from typing import List, Dict, Optional

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# --- Init ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-L1-Auto")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Clients ---
try:
    supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    logger.error(f"Setup Error: {e}")
    raise RuntimeError("Infra Fail")

# --- Optimized Models List (Stability Priority) ---
GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # Sabse zyada stable aur fast
    "llama-3.1-8b-instant",      # Fast backup
    "openai/gpt-oss-120b",       # High IQ (lekin low rate limit)
]

NEO_IDENTITY = (
    "You are Neo L1.0. An autonomous reasoning engine. "
    "Internal Monologue: Sharp, expert human thinking. "
    "Use browser_search for current info. Keep it natural and concise (2-3 sentences)."
)

# --- Core Logic ---
def extract_final_answer(reasoning: str) -> str:
    if not reasoning: return ""
    patterns = [r"(?:Final answer|Conclusion|Output):?\s*(.+?)(?:\n\n|$)", r"(?:Therefore|So|Thus),?\s*(.+?)(?:\n\n|$)"]
    for pat in patterns:
        match = re.search(pat, reasoning, re.IGNORECASE | re.DOTALL)
        if match: return match.group(1).strip()
    lines = [l.strip() for l in reasoning.split('\n') if len(l.strip()) > 20]
    return lines[-1] if lines else reasoning[:250].strip()

async def call_groq_with_auto_retry(messages: List[Dict]):
    """
    Automatic Rate-Limit Handling (Exponential Backoff)
    """
    last_error = ""
    for model in GROQ_MODELS:
        # Har model par 3 baar koshish (total 9 tries)
        for attempt in range(3):
            try:
                response = groq_client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=0.8,
                    max_completion_tokens=1500,
                    reasoning_effort="medium" if "gpt-oss" in model else None,
                    tools=[{"type": "browser_search"}],
                    stream=False
                )
                return response, model
            except Exception as e:
                err_str = str(e).lower()
                last_error = str(e)
                
                # Agar Rate Limit (429) aye toh wait karo
                if "rate_limit" in err_str or "429" in err_str:
                    wait_time = (attempt + 1) * 2 # 2s, 4s, 6s wait
                    logger.warning(f"Rate limited on {model}. Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    # Agar koi aur error hai toh next model par jao
                    logger.warning(f"Model {model} failed: {e}")
                    break 
                    
    raise HTTPException(503, f"All neural nodes busy. Last error: {last_error}")

# --- API Endpoints ---

@app.post("/v1/chat/completions")
async def chat_proxy(request: Request, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Key missing")
    
    user_key = authorization.replace("Bearer ", "").strip()
    body = await request.json()
    
    # 1. Check User
    user_res = supabase.table("users").select("token_balance").eq("api_key", user_key).execute()
    if not user_res.data: raise HTTPException(401, "Invalid Key")
    
    balance = user_res.data[0]["token_balance"]
    if balance <= 0: raise HTTPException(402, "Out of tokens")

    # 2. AI Call with Auto-Retry
    payload = [{"role": "system", "content": NEO_IDENTITY}] + body.get("messages", [])
    ai_raw, used_model = await call_groq_with_auto_retry(payload)

    # 3. Process
    msg_obj = ai_raw.choices[0].message
    content = msg_obj.content or ""
    reasoning = getattr(msg_obj, 'reasoning', "") or ""

    if not content.strip() and reasoning:
        content = extract_final_answer(reasoning)

    # 4. Bill
    tokens_used = ai_raw.usage.completion_tokens
    new_balance = max(0, balance - tokens_used)
    supabase.table("users").update({"token_balance": new_balance}).eq("api_key", user_key).execute()

    return {
        "message": content.strip(),
        "reasoning": reasoning.strip(),
        "usage": {"billed": tokens_used, "remaining": new_balance},
        "model_info": {"engine": used_model}
    }

# Baaki endpoints (new-key, balance) purane wale hi rahenge.
