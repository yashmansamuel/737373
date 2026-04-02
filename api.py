import os
import logging
import secrets
import re
import asyncio
from typing import List, Tuple
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# -----------------------------
# Configuration & Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SignaturesiNeo")

app = FastAPI(title="Signaturesi Neo L1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Service Clients
# -----------------------------
SUPABASE: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# Model Hierarchy (Optimized for Stability)
# -----------------------------
MODELS = [
    "openai/gpt-oss-20b",           
    "openai/gpt-oss-safeguard-20b", 
    "openai/gpt-oss-120b"           
]

# -----------------------------
# Refined Neutral System Prompt
# -----------------------------
# Added "Fluidity" instructions to ensure quality without increasing token length.
SYSTEM_PROMPT = (
    "Act: Neo-Concise Professional. Task: Provide a high-quality, direct answer using search data. "
    "Style: Neutral, objective, and precise. "
    "Constraints: No conversational fillers, no reasoning steps, no meta-talk. "
    "Format: Maximum 2 fluid sentences or 3 concise bullets."
)

# -----------------------------
# Data Models
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

class NewKeyResponse(BaseModel):
    api_key: str
    balance: int
    status: str

# -----------------------------
# Content Extraction Utility
# -----------------------------
def extract_best_content(message_obj) -> str:
    """Extracts content while maintaining structural integrity."""
    content = getattr(message_obj, 'content', "") or ""
    # Clean up any potential leftover tags or excessive whitespace
    cleaned = content.strip()
    return cleaned if len(cleaned) > 0 else "Response unavailable. Please refine your query."

# -----------------------------
# AI Engine: Performance Optimized
# -----------------------------
async def call_ai_with_fallback(messages: List[dict]) -> Tuple[object, str]:
    """Manages model transitions and minimizes input overhead."""
    # Context Optimization: System + Current Query Only
    minimal_context = [messages[0], messages[-1]] 

    for model in MODELS:
        try:
            params = {
                "model": model,
                "messages": minimal_context,
                "temperature": 0.2,           # Slightly increased for natural phrasing
                "max_completion_tokens": 180, 
                "stream": False,
            }
            
            # Neutral Reasoning Control: Effort set to None to prevent 'Hidden Billing'
            if "gpt-oss" in model:
                params["reasoning_effort"] = None 
                params["tools"] = [{"type": "browser_search"}]

            completion = GROQ.chat.completions.create(**params)
            logger.info(f"Active Engine: {model}")
            return completion, model

        except Exception as e:
            error_str = str(e).lower()
            logger.error(f"Fallback triggered for {model}: {error_str}")
            if "rate_limit" in error_str:
                await asyncio.sleep(0.5)
            continue

    raise HTTPException(503, "Neo-Engines are currently overloaded. Please try again.")

# -----------------------------
# Standard API Endpoints
# -----------------------------
@app.get("/")
def health():
    return {"status": "active", "engine": "Neo-L1.0", "provider": "Signaturesi"}

@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    resp = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    if not resp.data:
        raise HTTPException(404, "Invalid API Key")
    return {"api_key": api_key, "balance": resp.data["token_balance"]}

@app.post("/v1/user/new-key", response_model=NewKeyResponse)
async def generate_key(request: Request):
    new_api_key = f"sig-neo-{secrets.token_urlsafe(16)}"
    origin_country = request.headers.get("cf-ipcountry", "Global")
    SUPABASE.table("users").insert({
        "api_key": new_api_key,
        "token_balance": 2000,
        "country": origin_country
    }).execute()
    return {"api_key": new_api_key, "balance": 2000, "status": "Neo Active"}

@app.post("/v1/chat/completions")
async def neo_chat_proxy(payload: ChatRequest, authorization: str = Header(None)):
    # 1. Auth & Balance Check
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Authorization")
    
    api_key = authorization.replace("Bearer ", "")
    user_record = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    
    if not user_record.data:
        raise HTTPException(401, "Access Denied")
    
    current_bal = user_record.data["token_balance"]
    if current_bal <= 0:
        raise HTTPException(402, "Top-up Required")

    # 2. Execution
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + payload.messages
    ai_response, engine_used = await call_ai_with_fallback(full_messages)

    # 3. Billing Logic
    final_text = extract_best_content(ai_response.choices[0].message)
    usage = ai_response.usage
    total_tokens = usage.total_tokens
    
    # Secure balance update
    new_balance = max(0, current_bal - total_tokens)
    asyncio.create_task(
        asyncio.to_thread(
            SUPABASE.table("users").update({"token_balance": new_balance}).eq("api_key", api_key).execute
        )
    )

    # 4. Standard Neutral Response Format
    return {
        "id": f"neo_{secrets.token_hex(4)}",
        "object": "chat.completion",
        "model": "Neo-L1.0",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": final_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": total_tokens
        },
        "engine": engine_used
    }
