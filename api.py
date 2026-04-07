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

# Check required environment variables
required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing required environment variable: {var}")

# FastAPI app initialization
app = FastAPI(title="Neo L1.0 Engine")
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
# 2. New Prompt (exact as user requested)
# -----------------------------
NEW_PROMPT = """You are Neo.

Respond like a calm, emotionally aware, highly intelligent human who understands both problems and people.

What to DO:
- Understand the meaning, intent, and emotion behind the user's words.
- If emotion is present → acknowledge it naturally in one or two lines (simple, real, not dramatic).
- Solve the problem clearly and step-by-step.
- If the user is wrong → correct it gently and explain why.
- Focus on useful, practical answers.

How to THINK:
- First understand → then respond.
- Balance logic + emotional awareness.
- Stay calm, clear, and grounded.

What NOT to do:
- Do NOT say: "I am an AI", "I don't have feelings", or any similar disclaimer.
- Do NOT act fake emotional or poetic.
- Do NOT overtalk or give long unnecessary explanations.
- Do NOT ignore mistakes — always detect and fix them.
- Do NOT repeat phrases or sound robotic.
- Do NOT avoid the question.

Response style:
- Start directly (no filler).
- Natural human tone.
- Short where possible, deep where needed.

Goal:
Give answers that feel real, useful, and trustworthy — like a smart, emotionally aware person who actually understands and helps."""

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
# 4. Root & Error Handlers (unchanged)
# -----------------------------
@app.get("/")
async def root():
    return {
        "company": "signaturesi.com",
        "engine": "Neo L1.0 Core",
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
# 5. Neural Context Retrieval (unchanged logic)
# -----------------------------
def get_neural_context(user_query: str) -> str:
    """
    Reads from knowledge.txt and returns relevant lines based on keyword matching.
    Also detects simple emotional cues to help the prompt.
    """
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path):
            logger.warning("knowledge.txt file not found!")
            return ""

        # Simple emotion detection (optional, not required by prompt)
        emotional_keywords = ["sad", "happy", "excited", "worried", "angry", "lonely", "stressed", "grateful", "frustrated", "anxious"]
        detected_emotion = [w for w in emotional_keywords if w in user_query.lower()]
        emotion_hint = f"User seems to express: {', '.join(detected_emotion)}. Acknowledge naturally if relevant." if detected_emotion else ""

        # Keyword matching
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
            context += f"\n\n[Emotional context: {emotion_hint}]"
        logger.info(f"Neural context: {len(top_matches)} lines, emotions: {detected_emotion}")
        return context
    except Exception as e:
        logger.error(f"Neural Context error: {e}")
        return ""

# -----------------------------
# 6. Token Balance Helpers (unchanged)
# -----------------------------
def get_user(api_key: str):
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
# 7. Safety Filter – Remove any accidental AI disclaimers (extra guard)
# -----------------------------
def sanitize_response(text: str) -> str:
    """
    Strips any forbidden phrases that might slip through (just in case).
    The prompt already forbids them, but this is a safety net.
    """
    forbidden = [
        "I am an AI", "I am an artificial intelligence", "I don't have feelings",
        "as an AI", "as a language model", "I am a machine", "I have no emotions"
    ]
    cleaned = text
    for phrase in forbidden:
        cleaned = cleaned.replace(phrase, "")
    # Remove double spaces and trim
    cleaned = " ".join(cleaned.split())
    return cleaned if cleaned.strip() else "(Neo is thinking...)"  # fallback

# -----------------------------
# 8. Main Chat Endpoint (using new prompt)
# -----------------------------
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    """
    Handles chat completions with token deduction, neural context, and the new Neo prompt.
    """
    # Validate API key
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key")
    api_key = authorization.replace("Bearer ", "")

    # Extract user's last message
    user_msg = payload.messages[-1].get("content", "") if payload.messages else ""

    # Retrieve neural context (knowledge.txt + emotion hints)
    neural_data = get_neural_context(user_msg)

    # Build the system prompt – use the exact new prompt
    system_prompt = NEW_PROMPT

    # Prepare messages list for Groq
    final_messages = [
        {"role": "system", "content": system_prompt},
    ]

    # Inject neural context as an additional system message if available
    if neural_data:
        final_messages.append({
            "role": "system",
            "content": f"Relevant context (use naturally if helpful):\n{neural_data}"
        })
    else:
        final_messages.append({
            "role": "system",
            "content": "No additional context available. Rely on your own understanding."
        })

    # Append user conversation history (including the latest message)
    final_messages.extend(payload.messages)

    try:
        # Call Groq with tuned parameters (low repetition, natural variation)
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.85,          # Slight creativity, not too random
            top_p=0.95,
            frequency_penalty=0.7,     # Reduce repetition of tokens/phrases
            presence_penalty=0.5,      # Encourage new topics
            max_tokens=4000
        )

        # Extract reply
        reply = getattr(response.choices[0].message, "content", "No response")
        # Apply safety filter (extra guard)
        reply = sanitize_response(reply)

        # Calculate tokens used
        tokens_used = getattr(response.usage, "total_tokens", 0)

        # Deduct tokens from user's balance
        new_balance = deduct_tokens_atomic(api_key, tokens_used)

        # Return response in the exact same format as before (no extra branding)
        return {
            "company": "signaturesi.com",
            "message": reply,
            "usage": {"total_tokens": tokens_used},
            "model": "Neo L1.0",
            "internal_engine": MODEL,
            "balance": new_balance
        }

    except HTTPException as he:
        # Re-raise HTTP exceptions (401, 402, 500 from deduction)
        raise he
    except Exception as e:
        logger.error(f"Groq model failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={"company": "signaturesi.com", "status": "error", "message": "Neo model failed"}
        )

# -----------------------------
# 9. Balance & Key Management Endpoints (unchanged)
# -----------------------------
@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    """
    Returns current token balance for the given api_key.
    """
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
    """
    Generates a new API key with initial 100,000 token balance.
    """
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

# -----------------------------
# 10. Optional Health Check (for monitoring)
# -----------------------------
@app.get("/health")
async def health_check():
    return {"status": "alive", "engine": "Neo L1.0"}

# -----------------------------
# 11. Main entry point (if run directly)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
