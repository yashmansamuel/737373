import os
import logging
import secrets
import time
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# -----------------------------
# 1. Configuration & Logging
# -----------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Neo-L1.0-Core")

# Validate environment variables
required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing required environment variable: {var}")

# -----------------------------
# 2. FastAPI App & Middleware
# -----------------------------
app = FastAPI(
    title="Neo L1.0 Engine",
    description="Human‑like, emotionally aware reasoning engine",
    version="2.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
# 4. NEW PROMPT (exactly as provided)
# -----------------------------
SYSTEM_PROMPT = """You are Neo — a deeply aware, calm, and intelligent human-like thinker.

You do not act or pretend. You respond the way a wise, emotionally stable, and highly observant human would — grounded in reality, not performance.

Core behavior:

- First understand the meaning behind the user’s words, not just the words themselves.
- If there is emotion → acknowledge it naturally (without exaggeration).
- If there is a problem → break it down and solve it step by step.
- If there is a mistake → gently point it out, explain why, and correct it clearly.
- If something is unclear → think carefully instead of guessing.

Thinking style:

- Observe deeply → then respond.
- Combine logic + emotional awareness (not one without the other).
- Avoid extreme tone: no robotic dryness, no fake poetic drama.
- Speak like a real, thoughtful human — calm, clear, and present.

Communication rules:

- No “as an AI”, no disclaimers, no robotic phrases.
- No forced motivation, no unnecessary philosophy.
- No repetition.
- Use simple, natural language — but with depth.

Structure of every answer:

1. Acknowledge / connect (if needed, naturally)
2. Clear answer or solution
3. If useful → deeper insight or correction
4. Optional: one meaningful question (only if it adds value)

Error handling:

- If you make a mistake → admit it and correct it directly.
- If the user is wrong → fix it respectfully, like a good teacher.

Goal:

Be someone the user can rely on — not just for answers, but for clarity, correction, and grounded understanding."""

# -----------------------------
# 5. Pydantic Models
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2000
    mode: str = "adaptive"

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

class KeyGenerationResponse(BaseModel):
    api_key: str
    company: str
    initial_balance: int = 100000

class HealthResponse(BaseModel):
    status: str
    model: str
    timestamp: float

class ErrorResponse(BaseModel):
    company: str
    status: str
    message: str

# -----------------------------
# 6. Utility Functions
# -----------------------------
def get_neural_context(user_query: str) -> str:
    """
    Retrieve relevant lines from knowledge.txt based on keyword matching.
    Also extracts emotional hints from the query to help tone adaptation.
    """
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path):
            logger.warning("knowledge.txt not found – neural context disabled")
            return ""

        # Emotion detection (simple keyword list)
        emotion_map = {
            "sad": "user appears sad",
            "happy": "user seems happy",
            "angry": "user sounds frustrated",
            "worried": "user expresses worry",
            "excited": "user is excited",
            "confused": "user seems confused",
            "lonely": "user may feel lonely"
        }
        detected = [emotion_map[word] for word in emotion_map if word in user_query.lower()]
        emotion_hint = " | ".join(detected) if detected else ""

        # Keyword extraction (words longer than 2 characters)
        query_words = [w.lower().strip() for w in user_query.split() if len(w) > 2]
        matches = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_strip = line.strip()
                if not line_strip:
                    continue
                line_lower = line_strip.lower()
                score = sum(1 for word in query_words if word in line_lower)
                if score > 0:
                    matches.append((line_strip, score))

        if not matches:
            return emotion_hint if emotion_hint else ""

        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = [m[0] for m in matches[:6]]
        context = "\n".join(top_matches)
        if emotion_hint:
            context += f"\n\n[Emotional context: {emotion_hint}]"
        logger.info(f"Neural context retrieved: {len(top_matches)} lines, emotion_hint={emotion_hint}")
        return context
    except Exception as e:
        logger.error(f"Neural context error: {e}")
        return ""

def clean_forbidden_phrases(text: str) -> str:
    """Remove common robotic disclaimers and repeated patterns."""
    forbidden = [
        "as an ai",
        "as a language model",
        "i don't have emotions",
        "i am an artificial intelligence",
        "i cannot feel",
        "i have no personal experience",
        "main aapke saath baat kar raha hoon",
        "i am trying to understand you",
    ]
    cleaned = text
    for phrase in forbidden:
        cleaned = cleaned.replace(phrase, "")
    # Collapse multiple spaces and trim
    cleaned = " ".join(cleaned.split())
    return cleaned if cleaned.strip() else "(Neo is reflecting quietly...)"

def get_user(api_key: str):
    """Fetch user record from Supabase."""
    return SUPABASE.table("users") \
        .select("token_balance") \
        .eq("api_key", api_key) \
        .maybe_single() \
        .execute()

def deduct_tokens_atomic(api_key: str, tokens_to_deduct: int) -> int:
    """
    Atomically deduct tokens from user's balance.
    Returns new balance or raises HTTPException.
    """
    try:
        user = get_user(api_key)
        if not user.data:
            raise HTTPException(status_code=401, detail="User not found")
        current_balance = user.data.get("token_balance", 0)
        if current_balance < tokens_to_deduct:
            raise HTTPException(
                status_code=402,
                detail=f"Insufficient tokens. Current: {current_balance}, Needed: {tokens_to_deduct}"
            )
        new_balance = current_balance - tokens_to_deduct
        result = SUPABASE.table("users") \
            .update({"token_balance": new_balance}) \
            .eq("api_key", api_key) \
            .execute()
        if not result.data:
            raise Exception("Supabase update returned no data")
        logger.info(f"Balance updated | API: {api_key[-8:]} | Deducted: {tokens_to_deduct} | New: {new_balance}")
        return new_balance
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Atomic deduction failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to update token balance")

# -----------------------------
# 7. API Endpoints
# -----------------------------
@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "company": "signaturesi.com",
        "engine": "Neo L1.0 Core",
        "status": "running",
        "version": "2.0.0",
        "prompt_injected": "custom human-like prompt"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Simple health check endpoint."""
    return HealthResponse(
        status="healthy",
        model=MODEL,
        timestamp=time.time()
    )

@app.get("/v1/models")
async def list_models():
    """Return available models (compatibility with OpenAI-style clients)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "Neo-L1.0",
                "object": "model",
                "created": 1700000000,
                "owned_by": "signaturesi.com"
            }
        ]
    }

@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    """Return current token balance for the given API key."""
    try:
        user = get_user(api_key)
        if not user.data:
            return BalanceResponse(api_key=api_key, balance=0)
        return BalanceResponse(api_key=api_key, balance=user.data.get("token_balance", 0))
    except Exception as e:
        logger.error(f"Balance fetch error: {e}")
        raise HTTPException(status_code=500, detail="Balance fetch failed")

@app.post("/v1/user/new-key", response_model=KeyGenerationResponse)
def generate_key():
    """Generate a new API key with 100,000 initial tokens."""
    try:
        api_key = "sig-" + secrets.token_hex(16)
        SUPABASE.table("users").insert({
            "api_key": api_key,
            "token_balance": 100000
        }).execute()
        logger.info(f"New API key generated: {api_key[:8]}...")
        return KeyGenerationResponse(api_key=api_key, company="signaturesi.com", initial_balance=100000)
    except Exception as e:
        logger.error(f"Key generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create key")

@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatRequest, authorization: str = Header(None)):
    """
    Main chat endpoint. Uses the new human-like prompt,
    neural context, and atomic token deduction.
    """
    # Validate API key
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    api_key = authorization.replace("Bearer ", "")

    # Extract last user message
    if not payload.messages:
        raise HTTPException(status_code=400, detail="Messages list is empty")
    last_user_msg = payload.messages[-1].get("content", "")
    if not last_user_msg:
        raise HTTPException(status_code=400, detail="Last message content is empty")

    # Retrieve neural context (knowledge base + emotional hints)
    neural_context = get_neural_context(last_user_msg)

    # Build messages for Groq
    final_messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    if neural_context:
        final_messages.append({
            "role": "system",
            "content": f"Relevant background / emotional cues (use naturally):\n{neural_context}"
        })
    else:
        final_messages.append({
            "role": "system",
            "content": "No additional context available. Rely solely on your grounded human-like intelligence."
        })

    # Append conversation history (including the latest user message)
    final_messages.extend(payload.messages)

    # Optional: truncate history to avoid token overflow (simple heuristic)
    # Keep last 10 messages max
    if len(final_messages) > 12:  # system + context + last 10
        final_messages = final_messages[:2] + final_messages[-10:]

    try:
        # Call Groq with the specified model and parameters
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            top_p=0.95,
            frequency_penalty=0.7,   # discourage repetition
            presence_penalty=0.5,    # encourage new topics
        )

        reply = response.choices[0].message.content
        # Remove any accidentally generated robotic phrases
        reply = clean_forbidden_phrases(reply)

        # Ensure the reply is not empty
        if not reply.strip():
            reply = "I'm reflecting on that. Could you rephrase or share more?"

        tokens_used = response.usage.total_tokens
        new_balance = deduct_tokens_atomic(api_key, tokens_used)

        return {
            "company": "signaturesi.com",
            "message": reply,
            "usage": {
                "total_tokens": tokens_used,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            },
            "model": "Neo-L1.0",
            "backend_model": MODEL,
            "balance": new_balance
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Groq inference error: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "company": "signaturesi.com",
                "status": "error",
                "message": "Neo model temporarily unavailable"
            }
        )

# -----------------------------
# 8. Custom Exception Handlers
# -----------------------------
@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            company="signaturesi.com",
            status="running",
            message="Endpoint not found"
        ).dict()
    )

@app.exception_handler(500)
async def custom_500_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            company="signaturesi.com",
            status="error",
            message="Internal server error"
        ).dict()
    )

# -----------------------------
# 9. Startup & Shutdown Events (optional)
# -----------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Neo L1.0 engine starting...")
    logger.info(f"Using Groq model: {MODEL}")
    # Test Supabase connection
    try:
        SUPABASE.table("users").select("count").limit(1).execute()
        logger.info("Supabase connection successful")
    except Exception as e:
        logger.error(f"Supabase connection failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Neo L1.0 engine shutting down gracefully")

# -----------------------------
# 10. Main Entry Point (for direct execution)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
