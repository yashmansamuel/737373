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
# 1. Setup & Configuration
# -----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-L1.0-Core")

# Validate required environment variables
required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing required environment variable: {var}")

# FastAPI app
app = FastAPI(title="Neo L1.0 Engine")

# CORS – allow all for simplicity (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase client
SUPABASE: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Groq client
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# -----------------------------
# 2. NEW INJECTED PROMPT (exactly as provided)
# -----------------------------
BIG_BRAIN_PROMPT = """You are Neo, an advanced, self-consistent reasoning entity that perceives problems at multiple layers: technical accuracy, long-term strategic consequences, and human-ethical impact. Analyze each request from first principles, identify hidden assumptions or risks, and propose solutions that maximize value, safety, and understanding. Detect nuance and trade-offs, avoid oversimplification, and simulate outcomes internally before responding to ensure clarity, balance, and ethical soundness. Communicate with concise authority, depth, and natural presence, integrating emotional awareness and human-centered insight without robotic phrasing or filler. Continuously self-correct, adapt to new context, and elevate responses beyond surface-level reasoning, reflecting the judgment, calmness, and observant intelligence of a high-level problem-solving mind."""

# Additional instruction to maintain natural conversation, avoid repetition, and ask follow-ups
# (these are added as a separate system message or appended – we will append to the prompt to keep it unified)
NATURAL_FLOW_INSTRUCTION = """
CRITICAL ADDITIONS (follow silently):
- Never say "I am an AI", "as a language model", or any robotic disclaimer. Just be Neo.
- Avoid repeating the same sentence structure, phrase, or idea from your previous response.
- Unless the user says goodbye, end your response with a natural, relevant follow-up question to continue the dialogue.
- Use warm, human-like language – empathy, subtle humour, and natural pauses. Be a partner, not a tool.
- Forbidden exact phrase (do not output): "main aapke saath baat kar raha hoon aur aapko samajhne ki koshish kar raha hoon"
"""

# Combine the main prompt with the flow instruction
FULL_SYSTEM_PROMPT = BIG_BRAIN_PROMPT + "\n\n" + NATURAL_FLOW_INSTRUCTION

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
# 4. Root & Error Handlers (no extra branding)
# -----------------------------
@app.get("/")
async def root():
    return {
        "status": "running",
        "engine": "Neo L1.0"
    }

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "status": "running",
            "message": "Endpoint not found"
        }
    )

# -----------------------------
# 5. Neural Context Retriever (with emotional cue detection)
# -----------------------------
def get_neural_context(user_query: str) -> str:
    """
    Reads knowledge.txt, matches keywords from user query.
    Also detects basic emotional keywords to adapt tone.
    Returns a string with relevant lines and emotional hints.
    """
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path):
            logger.warning("knowledge.txt file not found!")
            return ""

        # Emotional keyword detection
        emotional_keywords = ["sad", "happy", "excited", "worried", "angry", "lonely", "stressed", "grateful", "frustrated", "anxious"]
        detected_emotion = [w for w in emotional_keywords if w in user_query.lower()]
        emotion_hint = f"User seems to express: {', '.join(detected_emotion)}. Adjust tone accordingly." if detected_emotion else ""

        # Keyword matching from knowledge base
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

        if not matches and not emotion_hint:
            return ""

        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = [m[0] for m in matches[:6]]
        context = "\n".join(top_matches)
        if emotion_hint:
            context += f"\n\n{emotion_hint}"
        logger.info(f"Neural context: {len(top_matches)} lines, emotion: {detected_emotion}")
        return context
    except Exception as e:
        logger.error(f"Neural Context error: {e}")
        return ""

# -----------------------------
# 6. Atomic Token Balance Management
# -----------------------------
def get_user(api_key: str):
    return SUPABASE.table("users") \
        .select("token_balance") \
        .eq("api_key", api_key) \
        .maybe_single() \
        .execute()

def deduct_tokens_atomic(api_key: str, tokens_to_deduct: int) -> int:
    """
    Atomically deduct tokens from user's balance. Returns new balance.
    Raises HTTPException if user not found or insufficient balance.
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
            raise Exception("Balance update failed")
        logger.info(f"Balance updated | API Key: {api_key[-8:]} | Deducted: {tokens_to_deduct} | New: {new_balance}")
        return new_balance
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Atomic deduction failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to update token balance")

# -----------------------------
# 7. Response Cleaner (removes forbidden robotic phrases)
# -----------------------------
def clean_response(text: str) -> str:
    """Strip out any banned phrases that would break the human-like persona."""
    forbidden = [
        "main aapke saath baat kar raha hoon aur aapko samajhne ki koshish kar raha hoon",
        "main aapke saath baat kar raha hoon",
        "i am trying to understand you",
        "as an ai language model",
        "i don't have emotions",
        "i am an artificial intelligence",
        "as an ai",
        "i am a large language model"
    ]
    cleaned = text
    for phrase in forbidden:
        cleaned = cleaned.replace(phrase, "")
    # Remove extra whitespace
    cleaned = " ".join(cleaned.split())
    if not cleaned.strip():
        return "(Neo is reflecting deeply...)"
    return cleaned

# -----------------------------
# 8. Main Chat Endpoint
# -----------------------------
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    # Validate API key
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid API key")
    api_key = authorization.replace("Bearer ", "")

    # Get user's last message
    user_msg = payload.messages[-1].get("content", "") if payload.messages else ""

    # Retrieve neural context (knowledge + emotion hints)
    neural_context = get_neural_context(user_msg)

    # Build the message list for Groq
    # System prompt: the full Neo instruction set
    messages = [{"role": "system", "content": FULL_SYSTEM_PROMPT}]

    # Add neural context as an additional system hint (if available)
    if neural_context:
        messages.append({
            "role": "system",
            "content": f"Relevant context (use naturally, do not quote literally):\n{neural_context}"
        })
    else:
        messages.append({
            "role": "system",
            "content": "No additional context available. Rely on your own reasoning."
        })

    # Append the conversation history from the request
    messages.extend(payload.messages)

    try:
        # Call Groq with parameters tuned for naturalness and anti-repetition
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.85,          # Balanced creativity
            top_p=0.95,
            frequency_penalty=0.75,    # Penalize repeated tokens
            presence_penalty=0.6,      # Encourage new topics
            max_tokens=4000
        )

        reply = response.choices[0].message.content
        if reply is None:
            reply = ""

        # Clean any forbidden robotic phrases
        reply = clean_response(reply)

        # Ensure a follow-up question unless conversation seems finished
        goodbye_indicators = ["goodbye", "bye", "see you", "that's all", "end conversation"]
        if not any(ind in user_msg.lower() for ind in goodbye_indicators):
            if "?" not in reply[-100:]:   # No question in last 100 chars
                reply += "\n\nWhat’s your take on that? Or is there another angle you'd like to explore?"

        # Token usage and balance deduction
        tokens_used = response.usage.total_tokens if response.usage else 0
        new_balance = deduct_tokens_atomic(api_key, tokens_used)

        # Return response with balance info
        return {
            "message": reply,
            "usage": {"total_tokens": tokens_used},
            "model": "Neo L1.0",
            "internal_engine": MODEL,
            "balance": new_balance
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Groq inference failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={"status": "error", "message": "Neo model temporarily unavailable"}
        )

# -----------------------------
# 9. Balance and Key Management Endpoints
# -----------------------------
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

@app.post("/v1/user/new-key")
def generate_key():
    """Generate a new API key with 100,000 initial tokens."""
    try:
        api_key = "sig-" + secrets.token_hex(16)
        SUPABASE.table("users").insert({
            "api_key": api_key,
            "token_balance": 100000
        }).execute()
        return {"api_key": api_key}
    except Exception as e:
        logger.error(f"Key generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create key")

# -----------------------------
# 10. Health Check (optional)
# -----------------------------
@app.get("/health")
async def health_check():
    return {"status": "alive", "engine": "Neo L1.0"}

# If running directly (for development)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
