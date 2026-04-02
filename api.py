import os
import re
import logging
import secrets
import asyncio
from typing import List, Dict, Optional

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# --- Configuration & Logging ---
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

# --- Personality Engine (Natural & Human-like) ---
# Is prompt mein humne AI ko "Expert Persona" diya hai taaki robotic tone khatam ho jaye.
NEO_IDENTITY = (
    "You are Neo L1.0, a high-intelligence autonomous reasoning engine. "
    "Internal Monologue (CoT): Think like a sharp, curious human expert. "
    "Weigh options, be slightly critical, and avoid repeating the user's words. "
    "Response Style: Be extremely concise but natural. No 'As an AI' or 'I must follow rules'. "
    "Strict limit: 2 sentences or 3 bullets. Knowledge cutoff: July 2025."
)

GROQ_MODELS = [
    "openai/gpt-oss-120b", 
    "openai/gpt-oss-20b", 
    "llama-3.1-8b-instant"
]

# --- Core Logic ---
def extract_clean_answer(reasoning: str) -> str:
    """Natural answer extraction from a messy CoT block."""
    if not reasoning: return ""
    # Searching for conclusion markers
    conclusion_patterns = [
        r"(?:Final answer|Conclusion|Output):?\s*(.+?)(?:\n\n|$)",
        r"(?:Therefore|So|Thus),?\s*(.+?)(?:\n\n|$)",
        r"(?:\n|^)[•\*\-]\s*(.+?)(?=\n[•\*\-]|$) "
    ]
    for pattern in conclusion_patterns:
        match = re.search(pattern, reasoning, re.IGNORECASE | re.DOTALL)
        if match: return match.group(1).strip()
    
    # Fallback: Last line that looks like a real sentence
    valid_lines = [l.strip() for l in reasoning.split('\n') if len(l.strip()) > 15]
    return valid_lines[-1] if valid_lines else reasoning[:200]

async def call_ai_engine(messages: List[Dict]):
    """Try primary high-IQ models first, then fallback."""
    for model in GROQ_MODELS:
        try:
            # Temperature 0.8 is the sweet spot for 'Human-like' feel
            res = groq_client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=0.85, 
                max_completion_tokens=1000,
                reasoning_effort="medium" if "gpt-oss" in model else None,
                stream=False
            )
            return res, model
        except Exception as e:
            logger.warning(f"{model} failed: {e}")
            continue
    raise HTTPException(503, "All intelligence nodes are offline.")

# --- API Endpoints ---

@app.get("/")
def home():
    return {"status": "online", "engine": "Neo-L1.0", "version": "2025.4"}

@app.post("/v1/user/new-key")
async def generate_key(request: Request):
    key = f"sig-live-{secrets.token_urlsafe(24)}"
    country = request.headers.get("x-vercel-ip-country", "Unknown")
    try:
        supabase.table("users").insert({"api_key": key, "token_balance": 1000, "country": country}).execute()
        return {"api_key": key, "balance": 1000}
    except:
        raise HTTPException(500, "Key creation failed")

@app.get("/v1/user/balance")
async def get_balance(api_key: str):
    res = supabase.table("users").select("token_balance").eq("api_key", api_key).execute()
    if not res.data: raise HTTPException(404, "Invalid key")
    return {"balance": res.data[0]["token_balance"]}

@app.post("/v1/chat/completions")
async def chat_proxy(request: Request, authorization: str = Header(None)):
    # 1. Verification
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Auth missing")
    
    key = authorization.replace("Bearer ", "").strip()
    body = await request.json()
    
    # 2. Balance Check
    user_data = supabase.table("users").select("token_balance").eq("api_key", key).execute()
    if not user_data.data: raise HTTPException(401, "Unauthorized")
    
    balance = user_data.data[0]["token_balance"]
    if balance <= 0: raise HTTPException(402, "Top up required")

    # 3. AI Execution (Identity hidden in the flow)
    payload = [{"role": "system", "content": NEO_IDENTITY}] + body.get("messages", [])
    raw_res, used_model = await call_ai_engine(payload)

    # 4. Parsing Logic
    msg = raw_res.choices[0].message
    content = msg.content or ""
    cot = getattr(msg, 'reasoning', "") or ""

    if not content.strip() and cot:
        content = extract_clean_answer(cot)

    # 5. Billing & Cleanup
    cost = raw_res.usage.completion_tokens
    new_balance = max(0, balance - cost)
    supabase.table("users").update({"token_balance": new_balance}).eq("api_key", key).execute()

    return {
        "message": content.strip(),
        "reasoning": cot.strip(), # Frontend can toggle visibility
        "usage": {
            "billed": cost,
            "remaining": new_balance
        },
        "model_info": {
            "name": "Neo L1.0",
            "engine": used_model
        }
    }
