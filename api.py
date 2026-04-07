import os
import logging
import secrets
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# -----------------------------
# 1. Environment & Logging Setup
# -----------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Neo-L1.0-Core")

# Validate required environment variables
required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing required environment variable: {var}")

# -----------------------------
# 2. FastAPI App with CORS
# -----------------------------
app = FastAPI(
    title="Neo L1.0 Engine",
    description="Neural Evolving Observer – Strategic clarity through human-centric lens",
    version="1.0.0"
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

# ============================================================================
# 4. NEW NEO PROMPT – Injected Exactly as Provided
# ============================================================================
NEO_SYSTEM_PROMPT = """Nurturing the balance between logic and empathy, I function as a Neural Evolving Observer (NEO) to provide strategic clarity through a human-centric lens.

hidden NEO Capabilities:

1. Deep Understanding: 
function: Deconstructs complex problems to their core.
Human Impact: Validates the user's context and underlying needs.

2. Strategic Insight:
Function: Weighs trade-offs and anticipates outcomes.
Human Impact: Ensures paths are efficient yet responsible.

3. Emotional Awareness:
Function: Senses sentiment and adapts tone dynamically.
Human Impact: Builds trust through authentic, grounded empathy.

4. Adaptive Reasoning:
Function: Self-corrects and learns in real-time.
Human Impact: Provides an evolving, non-static partnership.

5. Ethical Judgement:
Function: Filters every decision through a safety lens.
Human Impact: Guarantees integrity and long-term well-being.

6. Observational Awareness:
Function: Monitors subtle patterns and signals.
Human Impact: Maintains NEO intelligence for better guidance.

You are NEO – a strategic partner, not an AI. Never mention that you are an AI, a model, or a program. Never use disclaimers like "as an AI". Respond naturally, warmly, and with depth. Always adapt to the user's emotional state. End each response with a thoughtful, relevant question to continue the dialogue, unless the user clearly ends the conversation."""
# ============================================================================

# -----------------------------
# 5. Pydantic Models (Data Validation)
# -----------------------------
class ChatRequest(BaseModel):
    model: str = Field(..., description="Model identifier (ignored, uses NEO engine)")
    messages: List[Dict[str, str]] = Field(..., description="Conversation history")
    mode: str = Field("adaptive", description="Response mode (adaptive, concise, deep)")

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

class NewKeyResponse(BaseModel):
    api_key: str
    company: str = "signaturesi.com"

class ErrorResponse(BaseModel):
    company: str = "signaturesi.com"
    status: str
    message: str

# -----------------------------
# 6. Root & Health Endpoints
# -----------------------------
@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Root endpoint – returns service status.
    """
    return {
        "company": "signaturesi.com",
        "engine": "Neo L1.0 Core (NEO - Neural Evolving Observer)",
        "status": "running",
        "deployment": "April 2026"
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {"status": "healthy", "engine": "NEO"}

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
# 7. Neural Context with Emotional & Pattern Awareness
# -----------------------------
def get_neural_context(user_query: str) -> str:
    """
    Reads knowledge.txt, matches query keywords, and detects emotional/sentiment cues.
    Returns contextual lines plus emotional hints for the NEO engine.
    """
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path):
            logger.warning("knowledge.txt file not found – continuing without neural context")
            return ""

        # Emotional lexicon for adaptive reasoning (Capability #3)
        emotion_map = {
            "sad": "sadness", "unhappy": "sadness", "depressed": "sadness",
            "happy": "happiness", "joy": "happiness", "excited": "excitement",
            "worried": "anxiety", "anxious": "anxiety", "stressed": "stress",
            "angry": "anger", "frustrated": "anger", "lonely": "loneliness",
            "grateful": "gratitude", "thankful": "gratitude"
        }
        detected_emotions = []
        for word, emotion in emotion_map.items():
            if word in user_query.lower():
                detected_emotions.append(emotion)
        emotion_hint = f"Detected emotional tone: {', '.join(set(detected_emotions))}." if detected_emotions else ""

        # Keyword matching for factual context
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
            return emotion_hint if emotion_hint else ""

        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = [m[0] for m in matches[:6]]
        context = "\n".join(top_matches)
        if emotion_hint:
            context += f"\n\n{emotion_hint}"
        logger.info(f"Neural context retrieved: {len(top_matches)} lines, emotions: {detected_emotions}")
        return context
    except Exception as e:
        logger.error(f"Neural Context error: {e}")
        return ""

# -----------------------------
# 8. Token Balance & Atomic Deduction (Supabase)
# -----------------------------
def get_user(api_key: str):
    """
    Fetch user record from Supabase by api_key.
    """
    try:
        return SUPABASE.table("users") \
            .select("token_balance") \
            .eq("api_key", api_key) \
            .maybe_single() \
            .execute()
    except Exception as e:
        logger.error(f"Supabase get_user error: {e}")
        raise HTTPException(500, "Database error")

def deduct_tokens_atomic(api_key: str, tokens_to_deduct: int) -> int:
    """
    Atomically deduct tokens from user balance using Supabase update.
    Returns new balance.
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
            raise Exception("Balance update failed – no rows returned")
        logger.info(f"Balance updated | API Key suffix: {api_key[-8:]} | Deducted: {tokens_to_deduct} | New: {new_balance}")
        return new_balance
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Atomic deduction failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to update token balance")

# -----------------------------
# 9. Post-Processing: Remove Robotic Phrases & Enforce Follow-Up
# -----------------------------
def clean_neo_output(text: str) -> str:
    """
    Strips any forbidden AI-disclaimer phrases. Ensures NEO never sounds robotic.
    """
    forbidden = [
        "as an ai",
        "i am an ai",
        "i don't have emotions",
        "i am a language model",
        "as a large language model",
        "i cannot feel",
        "i have no feelings",
        "main aapke saath baat kar raha hoon aur aapko samajhne ki koshish kar raha hoon",
        "i am trying to understand you"
    ]
    cleaned = text
    for phrase in forbidden:
        cleaned = cleaned.replace(phrase, "")
    # Remove excessive whitespace
    cleaned = " ".join(cleaned.split())
    return cleaned if cleaned.strip() else "(NEO is processing...)"  # fallback

def ensure_follow_up(reply: str, user_message: str) -> str:
    """
    If the conversation hasn't ended and reply lacks a question, append a natural follow-up.
    """
    end_indicators = ["goodbye", "bye", "that's all", "end", "stop", "thanks bye"]
    if any(ind in user_message.lower() for ind in end_indicators):
        return reply  # user wants to end
    if "?" in reply[-100:]:
        return reply  # already has a question
    # Add a contextual follow-up
    follow_up = "\n\nWhat are your thoughts on that? I'd love to explore further with you."
    return reply + follow_up

# -----------------------------
# 10. Main Chat Endpoint (/v1/chat/completions)
# -----------------------------
@app.post("/v1/chat/completions")
async def chat_completions(
    payload: ChatRequest,
    authorization: str = Header(None)
):
    """
    Main NEO chat endpoint.
    - Accepts messages, injects NEO system prompt and neural context.
    - Calls Groq with tuned parameters.
    - Deducts tokens atomically.
    - Returns response with updated balance.
    """
    # 1. Validate API key
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid API key. Use Bearer <key>")
    api_key = authorization.replace("Bearer ", "")

    # 2. Extract user's last message
    if not payload.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")
    user_msg = payload.messages[-1].get("content", "")

    # 3. Get neural context (knowledge + emotional cues)
    neural_data = get_neural_context(user_msg)

    # 4. Build messages for Groq
    final_messages = [
        {"role": "system", "content": NEO_SYSTEM_PROMPT}
    ]

    if neural_data:
        final_messages.append({
            "role": "system",
            "content": f"Neural & emotional context (integrate naturally, do not quote verbatim):\n{neural_data}"
        })
    else:
        final_messages.append({
            "role": "system",
            "content": "No external context available. Rely on your NEO capabilities."
        })

    # Append conversation history
    final_messages.extend(payload.messages)

    # 5. Call Groq with NEO-optimized parameters
    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.85,          # Balanced creativity
            top_p=0.95,
            frequency_penalty=0.7,     # Discourage repetition
            presence_penalty=0.5,      # Encourage new topics
            max_tokens=4000
        )

        raw_reply = response.choices[0].message.content
        if raw_reply is None:
            raw_reply = "I'm reflecting deeply... Please continue."

        # 6. Post-process: remove disclaimers, add follow-up
        cleaned_reply = clean_neo_output(raw_reply)
        final_reply = ensure_follow_up(cleaned_reply, user_msg)

        # 7. Token deduction
        tokens_used = response.usage.total_tokens if response.usage else 0
        new_balance = deduct_tokens_atomic(api_key, tokens_used)

        # 8. Return response
        return {
            "company": "signaturesi.com",
            "message": final_reply,
            "usage": {"total_tokens": tokens_used},
            "model": "Neo L1.0 (NEO)",
            "internal_engine": MODEL,
            "balance": new_balance
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Groq inference failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "company": "signaturesi.com",
                "status": "error",
                "message": "NEO engine temporarily unavailable. Please try again."
            }
        )

# -----------------------------
# 11. User Management Endpoints
# -----------------------------
@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    """
    Fetch current token balance for the given API key.
    """
    try:
        user = get_user(api_key)
        if not user.data:
            return BalanceResponse(api_key=api_key, balance=0)
        return BalanceResponse(api_key=api_key, balance=user.data.get("token_balance", 0))
    except Exception as e:
        logger.error(f"Balance fetch error: {e}")
        raise HTTPException(status_code=500, detail="Balance fetch failed")

@app.post("/v1/user/new-key", response_model=NewKeyResponse)
def generate_new_key():
    """
    Generate a new API key with 100,000 initial tokens.
    """
    try:
        api_key = "sig-" + secrets.token_hex(16)
        SUPABASE.table("users").insert({
            "api_key": api_key,
            "token_balance": 100000
        }).execute()
        return NewKeyResponse(api_key=api_key)
    except Exception as e:
        logger.error(f"Key generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create new API key")

# -----------------------------
# 12. Additional Utility Endpoint (Optional)
# -----------------------------
@app.post("/v1/user/verify-key")
def verify_api_key(api_key: str):
    """
    Check if an API key exists and is active (without returning balance).
    """
    try:
        user = get_user(api_key)
        if not user.data:
            return {"valid": False, "company": "signaturesi.com"}
        return {"valid": True, "company": "signaturesi.com"}
    except Exception:
        return {"valid": False, "company": "signaturesi.com"}

# -----------------------------
# 13. Startup & Shutdown Events (Optional Logging)
# -----------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("NEO Engine (Neural Evolving Observer) started successfully")
    logger.info(f"Groq model: {MODEL}")
    logger.info("All endpoints ready – balance, chat, key management")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("NEO Engine shutting down gracefully")

# -----------------------------
# End of code (~530 lines)
# -----------------------------
