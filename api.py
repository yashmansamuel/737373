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

# Environment variables load karein
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Signaturesi Neo L1.0 API")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants aur Client Initialization
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    groq_client = Groq()
    logger.info("Connected to Supabase and Groq successfully.")
except Exception as e:
    logger.error(f"Initialization Error: {e}")
    raise RuntimeError("Critical: Cannot connect to infrastructure.")

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
    """Reasoning field se final answer nikalne ke liye optimized patterns."""
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
            
    # Fallback: Akhri mofeed line uthayen
    lines = [l.strip() for l in reasoning.split('\n') if l.strip()]
    for line in reversed(lines):
        if len(line) > 10 and not any(x in line for x in ["Mode:", "Default:", "Triggers:"]):
            return line
            
    return reasoning[:200].strip()

async def call_groq_with_fallback(messages: List[Dict[str, str]]):
    """Multiple models par fallback mechanism."""
    for model in GROQ_MODELS:
        for attempt in range(2):
            try:
                completion = groq_client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=0.7,
                    max_completion_tokens=2048,
                    top_p=1,
                    reasoning_effort="medium",
                    stream=False,
                    tools=[{"type": "browser_search"}]
                )
                logger.info(f"Success with model: {model}")
                return completion, model
            except Exception as e:
                logger.warning(f"Model {model} failed on attempt {attempt+1}: {e}")
                await asyncio.sleep(1)
                
    raise HTTPException(status_code=500, detail="All AI models are currently unavailable.")

# --- API Endpoints ---

@app.get("/")
def home():
    return {"status": "Online", "brand": "Signaturesi", "version": "Neo L1.0 (2025)"}

@app.get("/v1/user/balance")
def get_balance(api_key: str):
    try:
        resp = supabase.table("users").select("token_balance").eq("api_key", api_key).execute()
        if not resp.data:
            raise HTTPException(404, "Invalid API Key")
        return {"api_key": api_key, "balance": resp.data[0]["token_balance"]}
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        logger.error(f"Supabase Error: {e}")
        raise HTTPException(500, "Internal database error")

@app.post("/v1/user/new-key")
async def generate_key(request: Request):
    new_key = f"sig-live-{secrets.token_urlsafe(16)}"
    country = request.headers.get("x-vercel-ip-country") or request.headers.get("cf-ipcountry") or "Unknown"
    
    try:
        supabase.table("users").insert({
            "api_key": new_key, 
            "token_balance": 1000, 
            "country": country
        }).execute()
        return {"api_key": new_key, "balance": 1000, "country": country}
    except Exception as e:
        logger.error(f"Key Generation Error: {e}")
        raise HTTPException(500, "Could not generate new key")

@app.post("/v1/chat/completions")
async def chat_proxy(request: Request, authorization: str = Header(None)):
    # Auth Validation
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")
    
    user_api_key = authorization.split(" ")[1]
    body = await request.json()
    
    if body.get("model") != "Neo-L1.0":
        raise HTTPException(400, "Invalid model selection. Use 'Neo-L1.0'")
    
    user_messages = body.get("messages")
    if not user_messages:
        raise HTTPException(400, "No messages provided")

    # User Check & Balance Validation
    try:
        resp = supabase.table("users").select("token_balance").eq("api_key", user_api_key).execute()
        if not resp.data:
            raise HTTPException(401, "Unauthorized: Invalid API Key")
        
        current_balance = resp.data[0]["token_balance"]
        if current_balance <= 0:
            raise HTTPException(402, "Payment Required: Balance exhausted")
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        logger.error(f"Database/Auth Error: {e}")
        raise HTTPException(500, "Internal Server Error")

    # AI Call
    messages_payload = [{"role": "system", "content": SYSTEM_PROMPT}] + user_messages
    ai_response, used_model = await call_groq_with_fallback(messages_payload)

    # Content Processing
    res_msg = ai_response.choices[0].message
    content = res_msg.content or ""

    # Check for empty content in gpt-oss reasoning models
    if not content.strip() and hasattr(res_msg, 'reasoning') and res_msg.reasoning:
        content = extract_answer_from_reasoning(res_msg.reasoning)
        logger.info(f"Answer extracted from reasoning field for {used_model}")

    if not content.strip():
        content = "AI response generation failed. Please refine your prompt."

    # Token Accounting (Deduct only completion/output tokens)
    tokens_consumed = ai_response.usage.completion_tokens
    updated_balance = max(0, current_balance - tokens_consumed)
    
    try:
        supabase.table("users").update({"token_balance": updated_balance}).eq("api_key", user_api_key).execute()
    except Exception as e:
        logger.error(f"Balance Update Failed: {e}")

    return {
        "message": content.strip(),
        "usage": {
            "completion_tokens": tokens_consumed,
            "total_tokens": ai_response.usage.total_tokens
        },
        "model_info": {
            "public_name": "Neo-L1.0 (2025)",
            "engine": used_model
        }
    }
