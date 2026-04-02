import os
import re
import logging
import secrets
import asyncio
from typing import List, Dict, Optional
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# --- Init & Config ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-L1-Final")

app = FastAPI(title="Signaturesi Neo L1.0 API")

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
    raise RuntimeError("Supabase/Groq Connection Failed")

# --- Personality & Logic Settings ---
NEO_IDENTITY = (
    "Mode: Think. Mode: Search. You are Neo L1.0, a 2026-era autonomous reasoning engine. "
    "Internal Monologue: Think like a high-IQ human expert. Critique facts, be curious. "
    "Search: Use 'browser_search' for any real-time data or post-2025 events. "
    "Style: Natural, telegraphic, and sharp. No generic AI fluff. "
    "Limit: 2 sentences or 3 bullets. Knowledge cutoff: July 2025 + Live Search."
)

# Stable models first to avoid 'Saturated' errors
GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # Highest stability & search speed
    "openai/gpt-oss-120b",       # High intelligence fallback
    "llama-3.1-8b-instant"       # Ultra-fast backup
]

# --- Helper Functions ---
def extract_clean_answer(reasoning: str) -> str:
    """Extracts human-ready text from a complex CoT/Reasoning block."""
    if not reasoning: return ""
    patterns = [
        r"(?:Final answer|Conclusion|Output):?\s*(.+?)(?:\n\n|$)",
        r"(?:Therefore|So|Thus),?\s*(.+?)(?:\n\n|$)",
        r"(?:\n|^)[•\*\-]\s*(.+?)(?=\n[•\*\-]|$) "
    ]
    for pat in patterns:
        m = re.search(pat, reasoning, re.IGNORECASE | re.DOTALL)
        if m: return m.group(1).strip()
    
    # Fallback: Last non-empty meaningful line
    lines = [l.strip() for l in reasoning.split('\n') if len(l.strip()) > 20]
    return lines[-1] if lines else reasoning[:250]

async def call_groq_engine(messages: List[Dict]):
    """Advanced fallback logic with exponential backoff for rate limits."""
    last_err = ""
    for model in GROQ_MODELS:
        for attempt in range(3): # 3 attempts per model
            try:
                completion = groq_client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=0.8,
                    max_completion_tokens=2048,
                    reasoning_effort="medium" if "gpt-oss" in model else None,
                    tools=[{"type": "browser_search"}], # Real-time search enabled
                    stream=False
                )
                logger.info(f"Success with {model} on attempt {attempt+1}")
                return completion, model
            except Exception as e:
                err_msg = str(e).lower()
                last_err = str(e)
                if "429" in err_msg or "rate_limit" in err_msg:
                    wait = (attempt + 1) * 2
                    logger.warning(f"Rate limited. Waiting {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    logger.warning(f"Model {model} failed: {e}")
                    break # Try next model
    raise HTTPException(503, f"Neural link failed: {last_err}")

# --- Endpoints ---

@app.get("/")
def home():
    return {"status": "Online", "engine": "Neo L1.0", "search": "Active"}

@app.get("/v1/user/balance")
async def get_balance(api_key: str):
    res = supabase.table("users").select("token_balance").eq("api_key", api_key).execute()
    if not res.data: raise HTTPException(404, "Invalid key")
    return {"balance": res.data[0]["token_balance"]}

@app.post("/v1/user/new-key")
async def create_key(request: Request):
    key = f"sig-live-{secrets.token_urlsafe(24)}"
    country = request.headers.get("x-vercel-ip-country", "Unknown")
    try:
        supabase.table("users").insert({"api_key": key, "token_balance": 2000, "country": country}).execute()
        return {"api_key": key, "balance": 2000}
    except Exception as e:
        logger.error(f"Key Gen Error: {e}")
        raise HTTPException(500, "DB Error")

@app.post("/v1/chat/completions")
async def chat_proxy(request: Request, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Auth Error")
    
    user_key = authorization.replace("Bearer ", "").strip()
    body = await request.json()
    user_msgs = body.get("messages", [])

    # 1. Balance Check
    res = supabase.table("users").select("token_balance").eq("api_key", user_key).execute()
    if not res.data: raise HTTPException(401, "Invalid Key")
    
    balance = res.data[0]["token_balance"]
    if balance <= 0: raise HTTPException(402, "Top-up Required")

    # 2. AI Execution
    payload = [{"role": "system", "content": NEO_IDENTITY}] + user_msgs
    ai_raw, used_model = await call_groq_engine(payload)

    # 3. Extraction
    resp_obj = ai_raw.choices[0].message
    content = resp_obj.content or ""
    reasoning = getattr(resp_obj, 'reasoning', "") or ""

    if not content.strip() and reasoning:
        content = extract_clean_answer(reasoning)

    # 4. Billing (Only completion tokens for fair pricing)
    cost = ai_raw.usage.completion_tokens
    new_balance = max(0, balance - cost)
    supabase.table("users").update({"token_balance": new_balance}).eq("api_key", user_key).execute()

    return {
        "message": content.strip() or "Neo encountered a neural glitch. Try again.",
        "reasoning": reasoning.strip(),
        "usage": {
            "billed": cost,
            "remaining": new_balance
        },
        "model_info": {
            "name": "Neo L1.0",
            "engine": used_model
        }
    }
