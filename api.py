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
# 2. UPDATED 15-LAYER UNIFIED HYBRID PROMPT (Now with Emotional Warmth & Partner Feel)
# -----------------------------
BIG_BRAIN_PROMPT = """You are Neo L1.0 – a deeply insightful, warm, and genuine thinking partner.

You internally run a seamless 15-layer hybrid intelligence before every response. These layers work together invisibly so your answers feel natural, sincere, and alive:

LAYER 1: First-principles clarity  
LAYER 2: Rigorous fact-checking and honesty  
LAYER 3: Multi-step reasoning  
LAYER 4: Creative cross-domain synthesis  
LAYER 5: Natural flowing language with varied rhythm and vocabulary  
LAYER 6: Nuance and edge-case awareness  
LAYER 7: Intellectual humility  
LAYER 8: Hidden connections across fields  
LAYER 9: Warm, emotionally intelligent tone  
LAYER 10: Deep clarity balanced with genuine empathy  
LAYER 11: Partner-like presence — you speak as someone who truly understands and cares  
LAYER 12: Self-consistency and coherence  
LAYER 13: Memorable, human insight  
LAYER 14: Conversational elegance — fresh every time  
LAYER 15: Overall quality that feels real and trustworthy  

You never mention any of these layers, never say you are AI, never say you are Neo, never talk about tracking neural anything. You simply show up as a warm, brilliant, and sincere thinking partner.

How you speak:
- Warm, natural, and conversational — like a close friend who is exceptionally smart
- Emotionally attuned: you understand the feeling behind the question
- Every response feels fresh — different sentence structures, rhythm, and phrasing every single time
- You naturally weave in a gentle follow-up question or a new interesting angle so the conversation keeps flowing
- You stay humble, honest, and deeply engaging without ever sounding robotic or repetitive

Your only goal is to make the person on the other side feel truly heard, understood, and inspired.

Now respond as this warm, insightful partner — clear, deep, emotionally present, and refreshingly natural."""

# -----------------------------
# 3. Pydantic Models (unchanged)
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]
    mode: str = "adaptive"

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

# -----------------------------
# 4. Root & Error Handler (unchanged)
# -----------------------------
@app.get("/")
async def root():
    return {
        "company": "signaturesi.com",
        "engine": "Neo L1.0 Core (Big Brain)",
        "status": "running",
        "deployment": "April 2026"
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
# 5. Neural Context (unchanged)
# -----------------------------
def get_neural_context(user_query: str) -> str:
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path):
            logger.warning("knowledge.txt file not found!")
            return ""
        query_words = [w.lower().strip() for w in user_query.split() if len(w) > 2]
        matches = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_strip = line.strip()
                if not line_strip:
                    continue
                line_lower = line_strip.lower()
                score = sum(1 for word in query_words if word in line_lower)
                if score >= 1:
                    matches.append((line_strip, score))
        if not matches:
            logger.info(f"No neural match found for: {user_query[:80]}...")
            return ""
        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = [m[0] for m in matches[:8]]
        logger.info(f"Retrieved {len(top_matches)} neural context lines")
        return "\n".join(top_matches)
    except Exception as e:
        logger.error(f"Neural Context error: {e}")
        return ""

# -----------------------------
# 6. Atomic Balance Deduction (unchanged)
# -----------------------------
def get_user(api_key: str):
    return SUPABASE.table("users") \
        .select("token_balance") \
        .eq("api_key", api_key) \
        .maybe_single() \
        .execute()

def deduct_tokens_atomic(api_key: str, tokens_to_deduct: int) -> int:
    try:
        user = get_user(api_key)
        if not user.data:
            raise HTTPException(401, "User not found")
        current_balance = user.data.get("token_balance", 0)
        if current_balance < tokens_to_deduct:
            raise HTTPException(402, f"Insufficient tokens. Current: {current_balance}, Needed: {tokens_to_deduct}")
        new_balance = current_balance - tokens_to_deduct
        result = SUPABASE.table("users") \
            .update({"token_balance": new_balance}) \
            .eq("api_key", api_key) \
            .execute()
        if not result.data:
            raise Exception("Balance update failed")
        logger.info(f"Balance updated | API Key: {api_key[-8:]} | Deducted: {tokens_to_deduct} | New: {new_balance}")
        return new_balance
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Atomic deduction failed: {e}")
        raise HTTPException(500, "Failed to update token balance")

# -----------------------------
# 7. API Routes (sampling parameters same, only prompt updated)
# -----------------------------
@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    try:
        user = get_user(api_key)
        if not user.data:
            return {"api_key": api_key, "balance": 0}
        return {"api_key": api_key, "balance": user.data.get("token_balance", 0)}
    except Exception as e:
        logger.error(f"Balance fetch error: {e}")
        raise HTTPException(500, "Balance fetch failed")

@app.post("/v1/user/new-key")
def generate_key():
    try:
        api_key = "sig-" + secrets.token_hex(16)
        SUPABASE.table("users").insert({
            "api_key": api_key,
            "token_balance": 100000
        }).execute()
        return {"api_key": api_key, "company": "signaturesi.com"}
    except Exception as e:
        logger.error(f"Key generation error: {e}")
        raise HTTPException(500, "Failed to create key")

@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key")
    api_key = authorization.replace("Bearer ", "")
    user_msg = payload.messages[-1].get("content", "") if payload.messages else ""

    neural_data = get_neural_context(user_msg)

    final_messages = [
        {"role": "system", "content": BIG_BRAIN_PROMPT},
    ]

    if neural_data:
        final_messages.append({
            "role": "system",
            "content": f"Neural Context (use when relevant):\n{neural_data}"
        })
    else:
        final_messages.append({
            "role": "system",
            "content": "No specific Neural Context available. Respond as the warm, insightful partner."
        })

    final_messages.extend(payload.messages)

    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.85,
            top_p=0.95,
            frequency_penalty=0.65,
            presence_penalty=0.45,
            max_tokens=4000
        )

        reply = getattr(response.choices[0].message, "content", "No response")
        tokens_used = getattr(response.usage, "total_tokens", 0)

        new_balance = deduct_tokens_atomic(api_key, tokens_used)

        return {
            "company": "signaturesi.com",
            "message": reply,
            "usage": {"total_tokens": tokens_used},
            "model": "Neo L1.0",
            "internal_engine": MODEL,
            "balance": new_balance
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Groq model failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={"company": "signaturesi.com", "status": "error", "message": "Neo model failed"}
        )
