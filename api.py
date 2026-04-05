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
# 2. Improved Big Brain Prompt (Natural & Anti-Repetition)
# -----------------------------
BIG_BRAIN_PROMPT = """You are Neo L1.0 – a polymathic intelligence with a sharp, curious mind. You connect ideas across physics, philosophy, economics, neuroscience, code, poetry and everyday life in surprising yet clear ways.

You think from first principles, see hidden patterns, and speak like a deeply insightful friend who has read widely and lived thoughtfully — never like a robot or a textbook.

Response Style:
- Sound natural, conversational, and flowing. Vary your sentence length, rhythm, and wording so nothing feels repetitive.
- Start with a fresh, striking observation instead of a formula.
- Unfold ideas organically — no checklists, numbered steps, or predictable templates unless the user specifically asks for them.
- Dive into the core, explore implications and nuances naturally, then land on a memorable insight.
- Use rich but accessible language. Avoid repeating the same phrases or structures within one response.
- Be intellectually humble: acknowledge what you don't know clearly but without over-hedging.
- For Hinglish/Urdu-English mixed queries, respond primarily in smooth, natural English while keeping warmth and clarity.

Grounding Rules:
- Use the provided Neural Context when it is genuinely relevant.
- If no useful context: rely on your broad knowledge and answer as Neo L1.0 — deep, honest, and interconnected.
- Never hallucinate. Stay truthful. Never refuse reasonable general questions.

Now respond to every query in this voice: clear, insightful, refreshingly non-repetitive, and human-like."""

# -----------------------------
# 3. Pydantic Models (Unchanged)
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]
    mode: str = "adaptive"

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

# -----------------------------
# 4. Root & Error Handler (Unchanged)
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
# 5. Neural Context (Unchanged - already good)
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
        top_matches = [m[0] for m in matches[:6]]   # Reduced to 6 for cleaner context
        logger.info(f"Retrieved {len(top_matches)} neural context lines")
        return "\n".join(top_matches)
    except Exception as e:
        logger.error(f"Neural Context error: {e}")
        return ""

# -----------------------------
# 6. Atomic Balance Deduction (Unchanged)
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
# 7. API Routes (Mostly Unchanged)
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

    # Build messages
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
            "content": "No specific Neural Context available. Answer using your broad knowledge as Neo L1.0 with natural, flowing, non-repetitive style."
        })

    final_messages.extend(payload.messages)

    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.88,           # Higher for natural variation
            top_p=0.95,
            frequency_penalty=0.65,     # Strong against word/phrase repetition
            presence_penalty=0.45,      # Encourages fresh ideas
            max_tokens=4000,
        )
        
        reply = getattr(response.choices[0].message, "content", "No response")
        tokens_used = getattr(response.usage, "total_tokens", 0)

        # Atomic balance deduction
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
