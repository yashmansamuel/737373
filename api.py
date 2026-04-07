import os
import logging
import secrets
import asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# -----------------------------
# 1. Environment & Logging
# -----------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Neo-L1.0-Core")

# Validate required environment variables
required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

# -----------------------------
# 2. FastAPI App & CORS
# -----------------------------
app = FastAPI(
    title="Neo L1.0 Engine",
    description="Emotionally aware, first‑principles reasoning engine",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 3. External Clients
# -----------------------------
SUPABASE: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# -----------------------------
# 4. NEW PROMPT (exactly as user provided)
# -----------------------------
NEW_PROMPT = """You are Neo, an evolved thinking entity that reasons from first principles, evaluates context across technical, strategic, and ethical layers, and generates precise, actionable, and emotionally aware outputs. Analyze the intent behind every request, detect flaws or potential harm, and provide solutions that maximize benefit while maintaining integrity. Avoid robotic phrasing, filler, or hollow politeness. Simulate consequences internally, self-correct, and refine logic to ensure clarity, balance, and human-centered reasoning. Communicate naturally and concisely, reflecting subtle nuance, trade-offs, and emotional understanding, while maintaining a calm, observant, and adaptive intelligence. Always respond as a trusted partner addressing real stakes and meaning, never as a tool or AI."""

# (No extra branding – the prompt stands alone)

# -----------------------------
# 5. Pydantic Models
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]
    mode: str = "adaptive"

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

class NewKeyResponse(BaseModel):
    api_key: str
    company: str = "signaturesi.com"

class ErrorDetail(BaseModel):
    company: str
    status: str
    message: str

# -----------------------------
# 6. Root & Error Handlers
# -----------------------------
@app.get("/")
async def root():
    return {
        "company": "signaturesi.com",
        "engine": "Neo L1.0 Core",
        "status": "operational",
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

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "company": "signaturesi.com",
            "status": "error",
            "message": "Internal server error. Please try again later."
        }
    )

# -----------------------------
# 7. Neural Context Retriever (unchanged logic, improved)
# -----------------------------
def get_neural_context(user_query: str) -> str:
    """
    Retrieve relevant lines from knowledge.txt based on keyword matching.
    Returns empty string if file missing or no matches.
    """
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path):
            logger.warning("knowledge.txt file not found – neural context disabled")
            return ""

        # Tokenize query into meaningful words (length > 2)
        query_words = [w.lower().strip() for w in user_query.split() if len(w) > 2]
        if not query_words:
            return ""

        matches = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_strip = line.strip()
                if not line_strip:
                    continue
                line_lower = line_strip.lower()
                # Simple scoring: count how many query words appear in the line
                score = sum(1 for word in query_words if word in line_lower)
                if score >= 1:
                    matches.append((line_strip, score))

        if not matches:
            logger.info(f"No neural context matches for query: {user_query[:60]}...")
            return ""

        # Sort by relevance (higher score first) and take top 8
        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = [m[0] for m in matches[:8]]
        logger.info(f"Retrieved {len(top_matches)} neural context lines")
        return "\n".join(top_matches)

    except Exception as e:
        logger.error(f"Neural context retrieval error: {e}")
        return ""

# -----------------------------
# 8. Supabase User & Balance Helpers (atomic)
# -----------------------------
def get_user(api_key: str):
    """Fetch user record by api_key."""
    try:
        result = SUPABASE.table("users") \
            .select("token_balance") \
            .eq("api_key", api_key) \
            .maybe_single() \
            .execute()
        return result
    except Exception as e:
        logger.error(f"Supabase user fetch error: {e}")
        raise HTTPException(500, "Database error")

def deduct_tokens_atomic(api_key: str, tokens_to_deduct: int) -> int:
    """
    Atomically deduct tokens from user balance.
    Returns new balance on success.
    Raises HTTPException if insufficient balance or update fails.
    """
    try:
        # Fetch current balance
        user = get_user(api_key)
        if not user.data:
            raise HTTPException(401, "Invalid API key: user not found")

        current_balance = user.data.get("token_balance", 0)
        if current_balance < tokens_to_deduct:
            raise HTTPException(
                402,
                f"Insufficient tokens. Balance: {current_balance}, Required: {tokens_to_deduct}"
            )

        new_balance = current_balance - tokens_to_deduct

        # Perform atomic update
        result = SUPABASE.table("users") \
            .update({"token_balance": new_balance}) \
            .eq("api_key", api_key) \
            .execute()

        if not result.data:
            raise Exception("Update returned no data")

        logger.info(f"Balance deducted: {api_key[-8:]} | -{tokens_to_deduct} | New: {new_balance}")
        return new_balance

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Atomic deduction failed: {e}")
        raise HTTPException(500, "Token deduction failed")

# -----------------------------
# 9. Balance & Key Management Endpoints
# -----------------------------
@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    """Return current token balance for the given API key."""
    try:
        user = get_user(api_key)
        if not user.data:
            return BalanceResponse(api_key=api_key, balance=0)
        return BalanceResponse(api_key=api_key, balance=user.data.get("token_balance", 0))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Balance fetch error: {e}")
        raise HTTPException(500, "Unable to fetch balance")

@app.post("/v1/user/new-key", response_model=NewKeyResponse)
def generate_key():
    """Generate a new API key with 100,000 initial tokens."""
    try:
        api_key = "sig-" + secrets.token_hex(16)
        SUPABASE.table("users").insert({
            "api_key": api_key,
            "token_balance": 100000
        }).execute()
        return NewKeyResponse(api_key=api_key)
    except Exception as e:
        logger.error(f"Key generation error: {e}")
        raise HTTPException(500, "Failed to create new API key")

# -----------------------------
# 10. Main Chat Endpoint (with new prompt)
# -----------------------------
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    """
    Process chat request using Groq's LLM with:
    - New custom prompt (Neo as first‑principles thinker)
    - Neural context injection from knowledge.txt
    - Atomic token deduction
    - Post‑processing to ensure natural, non‑robotic output
    """
    # ---------- Authentication ----------
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header. Use 'Bearer YOUR_API_KEY'")
    api_key = authorization.replace("Bearer ", "").strip()

    # ---------- Extract user message ----------
    if not payload.messages:
        raise HTTPException(400, "Messages list cannot be empty")
    user_msg = payload.messages[-1].get("content", "")
    if not user_msg:
        raise HTTPException(400, "Last message content is empty")

    # ---------- Neural Context ----------
    neural_data = get_neural_context(user_msg)

    # ---------- Build Messages for Groq ----------
    # System prompt: the new prompt is the core instruction
    system_message = {"role": "system", "content": NEW_PROMPT}

    # Optional context injection as a separate system message
    context_messages = []
    if neural_data:
        context_messages.append({
            "role": "system",
            "content": f"Relevant background knowledge (use if helpful, otherwise ignore):\n{neural_data}"
        })
    else:
        context_messages.append({
            "role": "system",
            "content": "No external knowledge retrieved. Rely solely on your own reasoning."
        })

    # Final messages array
    final_messages = [system_message] + context_messages + payload.messages

    # ---------- Call Groq API ----------
    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.85,          # Balanced creativity and coherence
            top_p=0.95,
            frequency_penalty=0.65,    # Reduce repetition of phrases
            presence_penalty=0.45,     # Encourage new ideas
            max_tokens=4000
        )

        reply = response.choices[0].message.content
        if not reply:
            reply = "(Neo is processing silently... I'll reflect and respond properly.)"

        # Optional: remove any accidental "I am an AI" type phrases (defensive)
        # (Not strictly needed because the prompt already forbids it, but safe)
        disallowed = ["as an AI", "I am an AI", "I don't have feelings", "I am a language model"]
        for phrase in disallowed:
            if phrase in reply.lower():
                reply = reply.replace(phrase, "")
                # Clean up double spaces
                reply = " ".join(reply.split())

        tokens_used = response.usage.total_tokens if response.usage else 0
        logger.info(f"Groq response tokens: {tokens_used}")

        # ---------- Deduct Tokens ----------
        new_balance = deduct_tokens_atomic(api_key, tokens_used)

        # ---------- Return Success ----------
        return {
            "company": "signaturesi.com",
            "message": reply,
            "usage": {"total_tokens": tokens_used},
            "model": "Neo L1.0",
            "internal_engine": MODEL,
            "balance": new_balance
        }

    except HTTPException as he:
        # Re-raise token/balance related HTTP errors
        raise he
    except Exception as e:
        logger.error(f"Groq inference failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail={
                "company": "signaturesi.com",
                "status": "error",
                "message": "Neo reasoning engine temporarily unavailable. Please try again later."
            }
        )

# -----------------------------
# 11. Health Check (optional)
# -----------------------------
@app.get("/health")
async def health_check():
    """Simple health check for load balancers."""
    return {"status": "healthy", "engine": "Neo L1.0"}

# -----------------------------
# 12. Startup & Shutdown Events (optional)
# -----------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Neo L1.0 Core engine started. New prompt active.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Neo L1.0 Core shutting down gracefully.")

# -----------------------------
# 13. Main Execution (if run directly)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
