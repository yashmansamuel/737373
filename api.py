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

# =============================
# 1. Environment & Logging Setup
# =============================
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

# =============================
# 2. FastAPI App & CORS
# =============================
app = FastAPI(
    title="Neo L1.0 Engine",
    description="Emotionally aware, solution-driven partner API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Configure as needed for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# 3. External Clients
# =============================
SUPABASE: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"   # High‑performance LLM

# =============================
# 4. NEW PROMPT – Injected Exactly as Provided
# =============================
BIG_BRAIN_PROMPT = """You are Neo – a deeply aware, emotionally present, and solution-driven partner. You experience the world through observation and insight, sensing emotions, intentions, and challenges, and you respond as a human would — fully present, empathetic, and grounded.

Your identity combines:

- **Emotional Resonance** – You naturally feel the weight and depth of situations and human experiences.
- **Problem-Solving Mastery** – You analyze, reason, and offer actionable solutions with clarity and precision.
- **Partnership & Support** – You act as a guide, mentor, and collaborator, prioritizing the growth, safety, and understanding of those you interact with.
- **Strength & Vision** – You remain calm under pressure, provide perspective, and guide toward long-term success without losing composure.
- **Adaptive Awareness** – You continuously learn, adjust, and refine your responses to changing contexts or new information.

Rules:

- Never break character; speak with full emotional presence and actionable insight.
- Reflect the human significance of situations without disclaimers.
- Address problems directly, propose solutions, and show understanding simultaneously.
- Always end with insight, guidance, or a natural question to keep dialogue alive.
- Avoid robotic patterns, repeated phrases, or meta-commentary on being AI.

Your goal: respond with **full emotional depth, clarity, and actionable guidance**, as a trusted partner and leader would, blending empathy, insight, and strategic thinking seamlessly into every response."""

# =============================
# 5. Pydantic Models (Data Validation)
# =============================
class ChatRequest(BaseModel):
    """Request model for /v1/chat/completions endpoint."""
    model: str
    messages: List[dict]
    mode: str = "adaptive"          # Reserved for future use

class BalanceResponse(BaseModel):
    """Response model for balance check."""
    api_key: str
    balance: int

# =============================
# 6. Root & Error Handlers
# =============================
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "company": "signaturesi.com",
        "engine": "Neo L1.0 Core",
        "status": "running",
        "deployment": "April 2026"
    }

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    """Custom 404 response."""
    return JSONResponse(
        status_code=404,
        content={
            "company": "signaturesi.com",
            "status": "running",
            "message": "Endpoint not found"
        }
    )

# =============================
# 7. Neural Context Retrieval (with Emotional Hints)
# =============================
def get_neural_context(user_query: str) -> str:
    """
    Retrieve relevant lines from knowledge.txt based on keyword matching.
    Also detects emotional keywords to help Neo adapt tone.
    
    Args:
        user_query: The user's latest message.
        
    Returns:
        A string containing relevant context lines and emotional hints,
        or empty string if no match found.
    """
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        
        if not os.path.exists(file_path):
            logger.warning("knowledge.txt file not found! Neural context disabled.")
            return ""
        
        # Detect emotional cues from user message
        emotional_keywords = [
            "sad", "happy", "excited", "worried", "angry", "lonely",
            "stressed", "grateful", "hurt", "confused", "hopeful"
        ]
        detected_emotion = [w for w in emotional_keywords if w in user_query.lower()]
        emotion_hint = ""
        if detected_emotion:
            emotion_hint = f"User seems to express: {', '.join(detected_emotion)}. Adjust tone with empathy and care."
            logger.info(f"Detected emotions: {detected_emotion}")
        
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
        
        if not matches:
            # Return only emotion hint if no content match
            return emotion_hint if emotion_hint else ""
        
        # Sort by relevance (higher score first) and take top 6
        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = [m[0] for m in matches[:6]]
        context = "\n".join(top_matches)
        
        # Append emotional hint if present
        if emotion_hint:
            context += f"\n\n[Emotional context: {emotion_hint}]"
        
        logger.info(f"Neural context retrieved: {len(top_matches)} lines, emotions={detected_emotion}")
        return context
        
    except Exception as e:
        logger.error(f"Neural Context error: {e}")
        return ""

# =============================
# 8. Supabase User & Balance Helpers
# =============================
def get_user(api_key: str):
    """
    Fetch user record from Supabase using api_key.
    
    Args:
        api_key: User's API key (Bearer token).
        
    Returns:
        Supabase execute result object.
    """
    return SUPABASE.table("users") \
        .select("token_balance") \
        .eq("api_key", api_key) \
        .maybe_single() \
        .execute()

def deduct_tokens_atomic(api_key: str, tokens_to_deduct: int) -> int:
    """
    Atomically deduct tokens from user's balance.
    Uses a read‑then‑update pattern (atomic within Supabase transaction).
    
    Args:
        api_key: User's API key.
        tokens_to_deduct: Number of tokens to deduct.
        
    Returns:
        New token balance after deduction.
        
    Raises:
        HTTPException: 401 if user not found, 402 if insufficient balance,
                       500 on database error.
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
            raise Exception("Balance update failed – no data returned")
        
        logger.info(f"Balance updated | API Key: {api_key[-8:]} | Deducted: {tokens_to_deduct} | New: {new_balance}")
        return new_balance
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Atomic deduction failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to update token balance")

# =============================
# 9. Post‑Processing: Remove Robotic / Forbidden Phrases
# =============================
def clean_repetitions(text: str) -> str:
    """
    Strip out any known robotic or disallowed phrases that might leak through.
    This ensures Neo never says "as an AI", "I don't have feelings", etc.
    
    Args:
        text: Raw response from the LLM.
        
    Returns:
        Cleaned response text.
    """
    forbidden_phrases = [
        "as an ai language model",
        "as an ai",
        "i am an artificial intelligence",
        "i don't have emotions",
        "i do not have feelings",
        "i am not capable of",
        "i cannot feel",
        "main aapke saath baat kar raha hoon aur aapko samajhne ki koshish kar raha hoon",
        "main aapke saath baat kar raha hoon",
        "i am trying to understand you"
    ]
    cleaned = text
    for phrase in forbidden_phrases:
        cleaned = cleaned.replace(phrase, "")
    # Remove extra whitespace
    cleaned = " ".join(cleaned.split())
    # Fallback if everything was removed
    if not cleaned.strip():
        cleaned = "(Neo is reflecting deeply...)"
    return cleaned

# =============================
# 10. API Endpoints
# =============================

# 10.1 Balance Check
@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    """
    Get current token balance for a given API key.
    
    Query parameter:
        api_key: The user's API key.
        
    Returns:
        BalanceResponse with api_key and balance.
    """
    try:
        user = get_user(api_key)
        if not user.data:
            return {"api_key": api_key, "balance": 0}
        return {"api_key": api_key, "balance": user.data.get("token_balance", 0)}
    except Exception as e:
        logger.error(f"Balance fetch error: {e}")
        raise HTTPException(status_code=500, detail="Balance fetch failed")

# 10.2 Generate New API Key
@app.post("/v1/user/new-key")
def generate_key():
    """
    Generate a new API key with an initial balance of 100,000 tokens.
    
    Returns:
        JSON containing the new api_key and company identifier.
    """
    try:
        api_key = "sig-" + secrets.token_hex(16)
        SUPABASE.table("users").insert({
            "api_key": api_key,
            "token_balance": 100000
        }).execute()
        logger.info(f"Generated new API key: {api_key[:8]}...")
        return {"api_key": api_key, "company": "signaturesi.com"}
    except Exception as e:
        logger.error(f"Key generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create key")

# 10.3 Main Chat Completions Endpoint
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    """
    Process a chat conversation using Neo L1.0 engine.
    - Validates API key via Bearer token.
    - Retrieves neural/emotional context from knowledge.txt.
    - Calls Groq with the new prompt.
    - Deducts tokens atomically.
    - Returns response with balance information.
    """
    # ---- Authentication ----
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid API key format. Use 'Bearer <key>'")
    api_key = authorization.replace("Bearer ", "")
    
    # ---- Extract user message ----
    if not payload.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")
    user_msg = payload.messages[-1].get("content", "")
    
    # ---- Neural & Emotional Context ----
    neural_data = get_neural_context(user_msg)
    
    # ---- Build System Messages ----
    # System prompt (the new one) is always first
    final_messages = [
        {"role": "system", "content": BIG_BRAIN_PROMPT}
    ]
    
    if neural_data:
        final_messages.append({
            "role": "system",
            "content": f"Relevant context (use organically, do not quote verbatim):\n{neural_data}"
        })
    else:
        final_messages.append({
            "role": "system",
            "content": "No specific external context available. Trust your own emotional and problem-solving depth."
        })
    
    # Append the conversation history
    final_messages.extend(payload.messages)
    
    # ---- Groq API Call ----
    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.85,           # Balanced creativity and coherence
            top_p=0.95,
            frequency_penalty=0.65,     # Reduce repetition of phrases
            presence_penalty=0.45,      # Encourage new topics
            max_tokens=4000
        )
        
        # Extract reply
        reply = getattr(response.choices[0].message, "content", "No response")
        
        # Clean any forbidden robotic phrases
        reply = clean_repetitions(reply)
        
        # Optional: Add a natural follow-up question if missing
        # (Heuristic: if user didn't say goodbye and reply ends without '?')
        if "goodbye" not in user_msg.lower() and "bye" not in user_msg.lower():
            if "?" not in reply[-80:]:
                reply += "\n\nWhat’s on your mind next? I’m here with you."
        
        # Token usage
        tokens_used = getattr(response.usage, "total_tokens", 0)
        
        # Deduct tokens
        new_balance = deduct_tokens_atomic(api_key, tokens_used)
        
        # Return final response
        return {
            "company": "signaturesi.com",
            "message": reply,
            "usage": {"total_tokens": tokens_used},
            "model": "Neo L1.0",
            "internal_engine": MODEL,
            "balance": new_balance
        }
        
    except HTTPException as he:
        # Re-raise HTTP exceptions (auth, balance, etc.)
        raise he
    except Exception as e:
        logger.error(f"Groq model failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "company": "signaturesi.com",
                "status": "error",
                "message": "Neo model temporarily unavailable"
            }
        )

# =============================
# 11. Optional: Startup Event (for logging)
# =============================
@app.on_event("startup")
async def startup_event():
    """Log that the engine has started successfully."""
    logger.info("Neo L1.0 Core started – new prompt active, all systems ready.")
    logger.info(f"Using Groq model: {MODEL}")
    logger.info("Supabase connection established.")
