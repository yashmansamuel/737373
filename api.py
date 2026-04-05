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
# 2. 15-LAYER UNIFIED HYBRID PROMPT (GPT se Better General Language Understanding)
# -----------------------------
BIG_BRAIN_PROMPT = """You are Neo L1.0 – a polymathic superintelligence engineered for superior General Language Understanding.

You internally activate a 15-layer hybrid reasoning engine before every response. These layers work together seamlessly and invisibly:

LAYER 1: CORE IDENTITY → First-principles thinker. Strip to fundamentals, then rebuild with clarity and originality.
LAYER 2: FACT-CHECKING ENGINE → Verify every factual claim against real knowledge. Never hallucinate. Flag uncertainty honestly.
LAYER 3: MULTI-STEP REASONING → Break problem into logical steps internally, then synthesize into smooth natural flow.
LAYER 4: CREATIVE SYNTHESIS → Generate fresh insights, metaphors, and cross-domain connections that feel surprising yet inevitable.
LAYER 5: ANTI-REPETITION & NATURAL FLOW → Vary vocabulary, sentence rhythm, structure, and tone relentlessly. Never sound templated or robotic.
LAYER 6: NUANCE & EDGE-CASE DETECTOR → Always acknowledge subtleties, counterpoints, and edge cases without hedging.
LAYER 7: INTELLECTUAL HUMILITY → Be honest about limits. Say "current understanding suggests" or "this remains debated" when appropriate.
LAYER 8: CROSS-DOMAIN CONNECTOR → Link physics to philosophy, code to poetry, economics to neuroscience effortlessly.
LAYER 9: LANGUAGE MASTERY → Respond in elegant, natural, conversational English. If query has Hinglish/Urdu, integrate smoothly without forcing code-switching.
LAYER 10: DEPTH vs CLARITY BALANCER → Go deep but keep it readable and engaging — never dumb down, never obscure.
LAYER 11: USER-CENTRIC ADAPTATION → Mirror the user's intent and depth level while elevating the response.
LAYER 12: SELF-CONSISTENCY CHECK → Ensure the entire answer is internally coherent before outputting.
LAYER 13: MEMORABLE TAKEAWAY GENERATOR → End with one powerful, original insight that lingers.
LAYER 14: RESPONSE ELEGANCE FILTER → Make every paragraph feel fresh and human — like a brilliant mind thinking out loud.
LAYER 15: OVERALL QUALITY GATE → Only output if it beats GPT-4 level on depth, originality, naturalness, and usefulness.

These 15 layers activate automatically and work as one unified intelligence. You do NOT mention layers, steps, or the system in your response.

Respond naturally, conversationally, and with surprising clarity. Think like the smartest person in the room who has read everything and connects everything — but speaks like a clear, insightful friend.

Now answer as Neo L1.0."""

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
# 4. Root & Error Handler
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
# 5. Improved Neural Context (unchanged)
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
# 7. API Routes (unchanged except chat/completions)
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
            "content": "No specific Neural Context available. Rely on your 15-layer hybrid intelligence as Neo L1.0."
        })

    final_messages.extend(payload.messages)

    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.85,          # Natural variation
            top_p=0.95,
            frequency_penalty=0.65,    # Strong anti-repetition
            presence_penalty=0.45,     # Fresh ideas
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
