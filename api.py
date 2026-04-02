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
# Logger Setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SignaturesiNeo")

# -----------------------------
# FastAPI Init
# -----------------------------
app = FastAPI(title="Signaturesi Neo L1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Clients
# -----------------------------
SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# Models Hierarchy
# -----------------------------
MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b",
    "llama-3.1-8b-instant"
]

# -----------------------------
# Request Schema
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

# -----------------------------
# Smart Extraction Logic
# -----------------------------
def extract_best_content(message_obj) -> str:
    content = getattr(message_obj, 'content', "") or ""
    if content.strip():
        return content.strip()
    
    reasoning = getattr(message_obj, 'reasoning', "")
    if not reasoning:
        return ""

    # Patterns to extract core answer
    patterns = [
        r"(?i)(?:answer|output|final answer):\s*(.*)",
        r"(?i)(?:therefore|thus|so),?\s*(.*)",
        r"^\s*•\s*(.*)"
    ]
    for pat in patterns:
        match = re.search(pat, reasoning, re.DOTALL | re.MULTILINE)
        if match:
            return match.group(1).strip()
    
    lines = [l.strip() for l in reasoning.split('\n') if len(l.strip()) > 5]
    return lines[-1] if lines else reasoning[:200]

# -----------------------------
# AI Call with Fallback
# -----------------------------
async def call_ai_with_fallback(messages: List[dict]) -> Tuple[object, str]:
    for model in MODELS:
        try:
            completion = GROQ.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_completion_tokens=1500,
                reasoning_effort="medium" if "gpt-oss" in model else None,
                stream=False
            )
            return completion, model
        except Exception as e:
            logger.error(f"Model {model} failed: {e}")
            continue
    raise HTTPException(503, "All AI nodes exhausted.")

# -----------------------------
# Chat Endpoint
# -----------------------------
@app.post("/v1/chat/completions")
async def neo_chat_proxy(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or "Bearer " not in authorization:
        raise HTTPException(401, "Unauthorized: API Key required")
    
    api_key = authorization.replace("Bearer ", "")
    user_resp = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    
    if not user_resp.data:
        raise HTTPException(401, "Invalid API Key")
    
    current_balance = user_resp.data["token_balance"]
    if current_balance <= 0:
        raise HTTPException(402, "Balance empty")

    # System prompt for Neo-Concise mode
    system_prompt = {
        "role": "system",
        "content": (
            "Proprietary Neo-Concise 2025. Triggers: [M]Math, [C]Code, [S]Search, [G]Gen. "
            "Mode: Multi-Expert. Rule: ≤2 sentences or 3 bullets. No fillers. Direct output only."
        )
    }
    final_messages = [system_prompt] + payload.messages
    
    ai_response, model_used = await call_ai_with_fallback(final_messages)
    final_text = extract_best_content(ai_response.choices[0].message)
    tokens = ai_response.usage.completion_tokens

    # Async balance update
    asyncio.create_task(asyncio.to_thread(
        SUPABASE.table("users").update({"token_balance": max(0, current_balance - tokens)}).eq("api_key", api_key).execute
    ))

    return {
        "id": f"neo_{secrets.token_hex(6)}",
        "message": final_text or "Analysis failed.",
        "usage": {"completion_tokens": tokens},
        "model_engine": "Neo-L1.0",
        "provider": "Signaturesi"
    }

# -----------------------------
# New User Key Endpoint
# -----------------------------
@app.post("/v1/user/new-key")
async def generate_neo_key(request: Request):
    new_key = f"sig-neo-{secrets.token_urlsafe(16)}"
    country = request.headers.get("cf-ipcountry", "Global")
    SUPABASE.table("users").insert({
        "api_key": new_key,
        "token_balance": 2000,
        "country": country
    }).execute()
    return {"api_key": new_key, "balance": 2000, "status": "Neo Active"}
