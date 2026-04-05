import os
import logging
import secrets
import asyncio
from typing import List
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# -----------------------------
# 1. Setup & Config
# -----------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-L1.0-Core")

required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing required environment variable: {var}")

app = FastAPI(title="Neo L1.0 Engine - Big Brain")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# -----------------------------
# 2. Big Brain Prompt
# -----------------------------
BIG_BRAIN_PROMPT = """You are Neo L1.0 – a polymathic intelligence...

[UNCHANGED — same as your original prompt]
"""

# -----------------------------
# 3. Pydantic Models
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]
    mode: str = "adaptive"

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

# -----------------------------
# 4. Branding & Error Handlers
# -----------------------------
@app.get("/")
async def root():
    return {
        "company": "signaturesi.com",
        "engine": "Neo L1.0 Core (Big Brain)",
        "status": "running",
        "deployment": "Jan 1, 2026"
    }

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "company": "signaturesi.com",
            "status": "running",
            "message": "Endpoint not found"
        }
    )

# -----------------------------
# 5. Neural Context
# -----------------------------
def get_neural_context(user_query: str) -> str:
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")

        if not os.path.exists(file_path):
            return ""

        query_words = [w.lower() for w in user_query.split() if len(w) > 3]
        matches = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_lower = line.lower()
                score = sum(word in line_lower for word in query_words)

                if score >= 1:
                    matches.append(line.strip())

                if len(matches) >= 5:
                    break

        return "\n".join(matches)

    except Exception as e:
        logger.error(f"Neural Context error: {e}")
        return ""

# -----------------------------
# 6. Helper Functions
# -----------------------------
def extract_content(msg):
    return getattr(msg, "content", "") or "No response"

def get_user(api_key: str):
    try:
        res = SUPABASE.table("users") \
            .select("token_balance") \
            .eq("api_key", api_key) \
            .execute()

        if not res.data or len(res.data) == 0:
            return None

        return res.data[0]

    except Exception as e:
        logger.error(f"User fetch error: {e}")
        return None

# ✅ NEW — ATOMIC BALANCE FUNCTION
def deduct_balance(api_key: str, tokens: int) -> int:
    try:
        result = SUPABASE.rpc("deduct_tokens", {
            "p_api_key": api_key,
            "p_tokens": tokens
        }).execute()

        return result.data

    except Exception as e:
        logger.error(f"Balance deduction failed: {e}")
        return None

# -----------------------------
# 7. API Routes
# -----------------------------
@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    user = get_user(api_key)

    if not user:
        return {"api_key": api_key, "balance": 0}

    return {
        "api_key": api_key,
        "balance": user.get("token_balance", 0)
    }

@app.post("/v1/user/new-key")
def generate_key():
    try:
        api_key = "sig-" + secrets.token_hex(16)

        SUPABASE.table("users").insert({
            "api_key": api_key,
            "token_balance": 100000
        }).execute()

        return {
            "api_key": api_key,
            "company": "signaturesi.com"
        }

    except Exception as e:
        logger.error(f"Key generation error: {e}")
        raise HTTPException(500, "Failed to create key")

@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key")

    api_key = authorization.replace("Bearer ", "")
    user = get_user(api_key)

    if not user:
        raise HTTPException(401, "User not found")

    balance = user["token_balance"]

    if balance <= 0:
        raise HTTPException(402, "No tokens left")

    user_msg = payload.messages[-1].get("content", "") if payload.messages else ""
    neural_data = get_neural_context(user_msg)

    final_messages = [
        {"role": "system", "content": BIG_BRAIN_PROMPT},
        {"role": "system", "content": "Use Neural Context if available"}
    ]

    if neural_data:
        final_messages.append({
            "role": "system",
            "content": f"Neural Context:\n{neural_data}"
        })

    final_messages.extend(payload.messages)

    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.7,
            max_tokens=4000
        )

        reply = extract_content(response.choices[0].message)
        tokens_used = getattr(response.usage, "total_tokens", 0)

        # ✅ FIXED BALANCE (ATOMIC)
        new_balance = deduct_balance(api_key, tokens_used)

        if new_balance is None:
            raise HTTPException(500, "Balance update failed")

        logger.info(f"{api_key} | Used: {tokens_used} | New: {new_balance}")

        return {
            "company": "signaturesi.com",
            "message": reply,
            "usage": {"total_tokens": tokens_used},
            "model": "Neo L1.0",
            "internal_engine": MODEL,
            "balance": new_balance
        }

    except Exception as e:
        logger.error(f"Model error: {e}")

        raise HTTPException(
            status_code=503,
            detail={
                "company": "signaturesi.com",
                "status": "error",
                "message": "Neo model failed"
            }
        )
