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

# --- Init ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeoL1")

app = FastAPI(title="Signaturesi Neo L1.0")

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
    raise RuntimeError("Infrastructure Fail")

# --- Personality & Models ---
# Human-like personality focus
NEO_IDENTITY = (
    "You are Neo L1.0, an elite autonomous reasoning engine with real-time web access. "
    "Internal Monologue (CoT): Think like a high-IQ human expert—weigh facts, be curious, critique findings. "
    "Search Usage: If query needs fresh data, use browser_search immediately. "
    "Response Style: Natural, concise, and direct. No 'As an AI' filler. "
    "Limit: 2-3 sharp sentences or bullets. Knowledge cutoff: July 2025."
)

GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]

# --- Core Logic ---
def extract_final_answer(reasoning: str) -> str:
    """Extracts a human-readable answer from complex CoT blocks."""
    if not reasoning: return ""
    patterns = [
        r"(?:Final answer|Conclusion|Output):?\s*(.+?)(?:\n\n|$)",
        r"(?:Therefore|So|Thus),?\s*(?:we )?answer:?\s*(.+?)(?:\n\n|$)",
        r"(?:\n|^)[•\*\-]\s*(.+?)(?=\n[•\*\-]|$)"
    ]
    for pat in patterns:
        match = re.search(pat, reasoning, re.IGNORECASE | re.DOTALL)
        if match: return match.group(1).strip()
    
    # Fallback to last substantial line
    lines = [l.strip() for l in reasoning.split('\n') if len(l.strip()) > 20]
    return lines[-1] if lines else reasoning[:250].strip()

async def call_groq_with_tools(messages: List[Dict]):
    """Calls Groq with fallback logic and browser search capability."""
    for model in GROQ_MODELS:
        for attempt in range(2):
            try:
                response = groq_client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=0.8,
                    max_completion_tokens=1500,
                    reasoning_effort="medium" if "gpt-oss" in model else None,
                    tools=[{"type": "browser_search"}], # LIVE SEARCH ENABLED
                    stream=False
                )
                return response, model
            except Exception as e:
                logger.warning(f"{model} attempt {attempt+1} failed: {e}")
                await asyncio.sleep(1)
    raise HTTPException(503, "Neural nodes saturated. Try again.")

# --- Endpoints ---

@app.get("/")
def health():
    return {"status": "online", "engine": "Neo-L1.0", "search": "enabled"}

@app.get("/v1/user/balance")
async def get_balance(api_key: str):
    res = supabase.table("users").select("token_balance").eq("api_key", api_key).execute()
    if not res.data: raise HTTPException(404, "Invalid key")
    return {"balance": res.data[0]["token_balance"]}

@app.post("/v1/user/new-key")
async def create_key(request: Request):
    new_key = f"sig-live-{secrets.token_urlsafe(24)}"
    country = request.headers.get("x-vercel-ip-country", "Global")
    try:
        supabase.table("users").insert({"api_key": new_key, "token_balance": 1500, "country": country}).execute()
        return {"api_key": new_key, "balance": 1500}
    except:
        raise HTTPException(500, "Database busy")

@app.post("/v1/chat/completions")
async def chat_proxy(request: Request, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Key missing")
    
    user_key = authorization.replace("Bearer ", "").strip()
    body = await request.json()
    user_messages = body.get("messages", [])

    # 1. Auth & Balance
    user_res = supabase.table("users").select("token_balance").eq("api_key", user_key).execute()
    if not user_res.data: raise HTTPException(401, "Invalid Key")
    
    balance = user_res.data[0]["token_balance"]
    if balance <= 0: raise HTTPException(402, "Balance exhausted")

    # 2. AI Processing
    payload = [{"role": "system", "content": NEO_IDENTITY}] + user_messages
    ai_raw, used_model = await call_groq_with_tools(payload)

    # 3. Extraction
    msg_obj = ai_raw.choices[0].message
    content = msg_obj.content or ""
    reasoning = getattr(msg_obj, 'reasoning', "") or ""

    if not content.strip() and reasoning:
        content = extract_final_answer(reasoning)

    # 4. Billing
    # Deduction based on completion (output + thinking) to be fair
    tokens_used = ai_raw.usage.completion_tokens
    new_balance = max(0, balance - tokens_used)
    supabase.table("users").update({"token_balance": new_balance}).eq("api_key", user_key).execute()

    return {
        "message": content.strip() or "Neo couldn't formulate a response.",
        "reasoning": reasoning.strip(),
        "usage": {
            "billed": tokens_used,
            "remaining": new_balance,
            "total_call_tokens": ai_raw.usage.total_tokens
        },
        "model_info": {
            "engine": used_model,
            "version": "2025.4-L1"
        }
    }
