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
# Logger
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SignaturesiNeo")

# -----------------------------
# FastAPI app
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
SUPABASE: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# Optimized Model Fallback (20B First for Stability)
# -----------------------------
MODELS = [
    "openai/gpt-oss-20b",           # Fast & High Rate Limit
    "openai/gpt-oss-safeguard-20b", # Security Layer
    "openai/gpt-oss-120b"           # Heavyweight Fallback
]

# -----------------------------
# Surgical System Prompt
# -----------------------------
SYSTEM_PROMPT = (
    "Act: Neo-Concise. Task: Direct factual output. "
    "Rules: No intros, no reasoning text, no corrections. "
    "Format: Max 2 sentences or 3 bullets. Be complete but lethal."
)

# -----------------------------
# Pydantic models
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
# Helper: Extract Content
# -----------------------------
def extract_best_content(message_obj) -> str:
    content = getattr(message_obj, 'content', "") or ""
    if content.strip():
        return content.strip()

    reasoning = getattr(message_obj, 'reasoning', "")
    if not reasoning:
        return ""

    patterns = [
        r"(?i)(?:answer|output|final answer):\s*(.*)",
        r"(?i)(?:therefore|thus|so),?\s*(.*)",
        r"^\s*[•\-\*]\s*(.*)",
    ]
    for pat in patterns:
        match = re.search(pat, reasoning, re.DOTALL | re.MULTILINE)
        if match:
            return match.group(1).strip()

    lines = [l.strip() for l in reasoning.split('\n') if len(l.strip()) > 5]
    return lines[-1] if lines else reasoning[:200]

# -----------------------------
# AI Call: Anti-Exhaustion Logic
# -----------------------------
async def call_ai_with_fallback(messages: List[dict]) -> Tuple[object, str]:
    # Context Trimming: Sirf System Prompt aur Last 2 messages bhejein (Saves Rate Limit)
    trimmed_messages = [messages[0]] + messages[-2:] if len(messages) > 2 else messages

    for model in MODELS:
        try:
            kwargs = {
                "model": model,
                "messages": trimmed_messages,
                "temperature": 0.2,           # Accuracy + Conciseness
                "max_completion_tokens": 180,  # Balanced limit for full answers
                "stream": False,
            }
            
            if "gpt-oss" in model:
                kwargs["reasoning_effort"] = "medium" # Logic perfection
                kwargs["tools"] = [{"type": "browser_search"}]

            completion = GROQ.chat.completions.create(**kwargs)
            logger.info(f"Success with {model}")
            return completion, model

        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"Model {model} failed: {error_msg}")
            
            # Agar rate limit hit ho, toh thoda ruk kar agle model par jayein
            if "rate_limit" in error_msg or "429" in error_msg:
                await asyncio.sleep(0.5)
            continue

    raise HTTPException(503, "All Neo-Engines exhausted. Try again in 5s.")

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def health():
    return {"status": "online", "model": "Neo L1.0", "engine": "Signaturesi-Hybrid"}

@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    resp = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    if not resp.data:
        raise HTTPException(404, "Invalid Key")
    return {"api_key": api_key, "balance": resp.data["token_balance"]}

@app.post("/v1/user/new-key", response_model=NewKeyResponse)
async def generate_key(request: Request):
    new_key = f"sig-neo-{secrets.token_urlsafe(16)}"
    country = request.headers.get("cf-ipcountry", "Global")
    SUPABASE.table("users").insert({
        "api_key": new_key,
        "token_balance": 2000,
        "country": country
    }).execute()
    return {"api_key": new_key, "balance": 2000, "status": "Neo Active"}

@app.post("/v1/chat/completions")
async def neo_chat_proxy(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Auth Failed")
    
    api_key = authorization.replace("Bearer ", "")
    user = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    
    if not user.data or user.data["token_balance"] <= 0:
        raise HTTPException(402, "Top-up required")

    if payload.model != "Neo-L1.0":
        raise HTTPException(400, "Use model: Neo-L1.0")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + payload.messages

    # Execute Hybrid Call
    ai_response, used_model = await call_ai_with_fallback(messages)

    # Process Output
    final_answer = extract_best_content(ai_response.choices[0].message)
    if not final_answer:
        final_answer = "Neo-Concise: Data processing failed. Try again."

    # Bill the user
    total_tokens = ai_response.usage.total_tokens
    new_balance = max(0, user.data["token_balance"] - total_tokens)

    asyncio.create_task(
        asyncio.to_thread(
            SUPABASE.table("users").update({"token_balance": new_balance}).eq("api_key", api_key).execute
        )
    )

    return {
        "id": f"neo_{secrets.token_hex(4)}",
        "message": final_answer,
        "usage": {"total_tokens": total_tokens},
        "engine": used_model,
        "status": "Success"
    }
