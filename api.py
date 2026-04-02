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

# 1. Environment Variables aur Logging setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Signaturesi Neo L1.0 API")

# 2. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Clients Initialization
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    groq_client = Groq()
    logger.info("Connected to Supabase and Groq successfully.")
except Exception as e:
    logger.error(f"Initialization Error: {e}")
    raise RuntimeError("Critical: Infrastructure connection failed.")

# 4. System Configuration
SYSTEM_PROMPT = (
    "Mode: Think. Triggers: [M]=MathHints, [C]=CodeSnippet, [H]=Health, [G]=General. "
    "Default:[G]. Format: ≤2 telegraphic sentences or 3 short bullets. No intro/outro/tags. "
    "Max 60 tokens. Your knowledge was last updated on July 27, 2025."
)

GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b",
    "llama-3.1-8b-instant",
]

# --- Helper Functions ---

def extract_answer_from_reasoning(reasoning: str) -> str:
    """Reasoning block se final answer nikalne ka logic."""
    if not reasoning:
        return ""
    
    patterns = [
        r"(?:So|Therefore|Thus),?\s*(?:we )?answer:?\s*(.+?)(?:\n\n|$)",
        r"Final (?:answer|output):?\s*(.+?)(?:\n\n|$)",
        r"Output:?\s*(.+?)(?:\n\n|$)",
    ]
    
    for pat in patterns:
        m = re.search(pat, reasoning, re.IGNORECASE | re.DOTALL)
        if m:
            ans = m.group(1).strip()
            if ans: return ans
            
    # Fallback: Last valid line
    lines = [l.strip() for l in reasoning.split('\n') if l.strip()]
    for line in reversed(lines):
        if len(line) > 10 and not any(x in line for x in ["Mode:", "Default:", "Triggers:"]):
            return line
            
    return reasoning[:200].strip()

async def call_groq_with_fallback(messages: List[Dict[str, str]]):
    """Multiple models fallback system."""
    for model in GROQ_MODELS:
        for attempt in range(2):
            try:
                completion = groq_client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=0.7,
                    max_completion_tokens=2048,
                    top_p=1,
                    reasoning_effort="medium", # Keeps CoT reasoning
                    stream=False,
                    # Note: Kuch models tools support nahi karte, fallback handle karega
                    tools=[{"type": "browser_search"}] 
                )
                return completion, model
            except Exception as e:
                logger.warning(f"Model {model} failed (Attempt {attempt+1}): {e}")
                await asyncio.sleep(1)
                
    raise HTTPException(status_code=500, detail="All AI engines are busy. Try again.")

# --- Endpoints ---

@app.get("/")
def home():
    return {"status": "Online", "brand": "Signaturesi", "engine": "Neo L1.0"}

@app.get("/v1/user/balance")
def get_balance(api_key: str):
    try:
        resp = supabase.table("users").select("token_balance").eq("api_key", api_key).execute()
        if not resp.data:
            raise HTTPException(404, "Invalid API Key")
        return {"api_key": api_key, "balance": resp.data[0]["token_balance"]}
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(500, "Database error")

@app.post("/v1/user/new-key")
async def generate_key(request: Request):
    new_key = f"sig-live-{secrets.token_urlsafe(16)}"
    country = request.headers.get("x-vercel-ip-country") or "Unknown"
    
    try:
        supabase.table("users").insert({
            "api_key": new_key, 
            "token_balance": 1000, 
            "country": country
        }).execute()
        return {"api_key": new_key, "balance": 1000}
    except Exception as e:
        raise HTTPException(500, "Key generation failed")

@app.post("/v1/chat/completions")
async def chat_proxy(request: Request, authorization: str = Header(None)):
    # 1. Validation
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Auth header missing")
    
    user_api_key = authorization.split(" ")[1]
    body = await request.json()
    
    if body.get("model") != "Neo-L1.0":
        raise HTTPException(400, "Use 'Neo-L1.0' as model name")
    
    user_messages = body.get("messages")
    if not user_messages:
        raise HTTPException(400, "No messages provided")

    # 2. Balance Check
    try:
        resp = supabase.table("users").select("token_balance").eq("api_key", user_api_key).execute()
        if not resp.data:
            raise HTTPException(401, "Invalid Key")
        
        balance = resp.data[0]["token_balance"]
        if balance <= 0:
            raise HTTPException(402, "Insufficient tokens")
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(500, "Auth server error")

    # 3. AI Processing
    messages_payload = [{"role": "system", "content": SYSTEM_PROMPT}] + user_messages
    ai_response, used_model = await call_groq_with_fallback(messages_payload)

    # 4. Content & Reasoning Extraction
    res_msg = ai_response.choices[0].message
    content = res_msg.content or ""
    # Reasoning field capture kar rahe hain
    raw_reasoning = getattr(res_msg, 'reasoning', "") 

    # Agar content blank hai (gpt-oss behavior), toh reasoning se extract karein
    if not content.strip() and raw_reasoning:
        content = extract_answer_from_reasoning(raw_reasoning)

    # 5. Token Accounting
    # 'completion_tokens' mein reasoning + answer dono count hote hain
    tokens_used = ai_response.usage.completion_tokens
    new_balance = max(0, balance - tokens_used)
    
    try:
        supabase.table("users").update({"token_balance": new_balance}).eq("api_key", user_api_key).execute()
    except Exception as e:
        logger.error(f"Failed to update balance: {e}")

    # 6. Final Response
    return {
        "message": content.strip() or "Sorry, I couldn't generate an answer.",
        "reasoning": raw_reasoning.strip(), # Frontend pe show karne ke liye
        "usage": {
            "completion_tokens": tokens_used,
            "total_tokens": ai_response.usage.total_tokens
        },
        "model_info": {
            "public_name": "Neo-L1.0",
            "internal_engine": used_model
        }
    }
