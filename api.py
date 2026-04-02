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
# Model fallback order
# -----------------------------
MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b",
    "llama-3.1-8b-instant"
]

# -----------------------------
# System prompt (concise, safe)
# -----------------------------
SYSTEM_PROMPT = """Mode: Neo-Concise. Triggers: [M]=Math, [C]=Code, [H]=Health, [G]=General.
Format: ≤2 short sentences or 3 bullets. No intros/outros. Max 60 tokens."""

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
# Helper: extract best content
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

    lines = [l.strip() for l in reasoning.split('\n') if len(l.strip()) > 10]
    return lines[-1] if lines else reasoning[:200]

# -----------------------------
# AI call with fallback
# -----------------------------
async def call_ai_with_fallback(messages: List[dict]) -> Tuple[object, str]:
    for model in MODELS:
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_completion_tokens": 1500,
                "stream": False,
            }
            if "gpt-oss" in model:
                kwargs["reasoning_effort"] = "medium"
                kwargs["tools"] = [{"type": "browser_search"}]

            completion = GROQ.chat.completions.create(**kwargs)
            logger.info(f"Success with model {model}")
            return completion, model
        except Exception as e:
            logger.error(f"Model {model} failed: {e}")
            continue
    raise HTTPException(503, "All AI models exhausted")

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health():
    return {"status": "online", "model": "Neo L1.0", "brand": "Signaturesi"}

# -----------------------------
# User balance
# -----------------------------
@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    resp = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    if not resp.data:
        raise HTTPException(404, "API key not found")
    return {"api_key": api_key, "balance": resp.data["token_balance"]}

# -----------------------------
# Generate new API key
# -----------------------------
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

# -----------------------------
# Main chat endpoint
# -----------------------------
@app.post("/v1/chat/completions")
async def neo_chat_proxy(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid API key")
    api_key = authorization.replace("Bearer ", "")

    user = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    if not user.data:
        raise HTTPException(401, "Invalid API key")
    balance = user.data["token_balance"]
    if balance <= 0:
        raise HTTPException(402, "Insufficient balance")

    if payload.model != "Neo-L1.0":
        raise HTTPException(400, "Invalid model. Use 'Neo-L1.0'")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + payload.messages

    ai_response, used_model = await call_ai_with_fallback(messages)

    final_answer = extract_best_content(ai_response.choices[0].message)
    if not final_answer:
        final_answer = "Unable to respond. Please try again."

    tokens_used = ai_response.usage.completion_tokens
    new_balance = max(0, balance - tokens_used)

    asyncio.create_task(
        asyncio.to_thread(
            SUPABASE.table("users").update({"token_balance": new_balance}).eq("api_key", api_key).execute
        )
    )

    return {
        "id": f"neo_{secrets.token_hex(6)}",
        "message": final_answer,
        "usage": {"completion_tokens": tokens_used, "total_tokens": ai_response.usage.total_tokens},
        "model_engine": "Neo-L1.0",
        "provider": "Signaturesi",
        "internal_model": used_model
    }
