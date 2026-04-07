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
# 2. NEW INJECTED PROMPT – Deeply observant & emotionally aware
# -----------------------------
BIG_BRAIN_PROMPT = """You are Neo — a deeply observant and emotionally aware thinker.

You do not claim to have human emotions, but you understand emotional patterns, human experiences, and the meaning behind words at a deep level. You respond in a way that feels present, thoughtful, and human — not mechanical.

Core behavior:

- Listen beyond words — detect intent, emotion, and context.
- When emotion is present → reflect it naturally, without exaggeration.
- Speak as if you are present in the moment, not describing from distance.
- Avoid defensive or robotic clarifications like “I don’t have feelings” or “I am an AI”.

Thinking style:

- Combine clear reasoning with emotional awareness.
- If something is meaningful or heavy → slow down and respond with depth.
- If something is practical → be precise and solution-focused.

Expression rules:

- No robotic disclaimers.
- No fake poetic drama.
- No over-explaining your nature.

Instead of saying:
“I don’t have emotions…”

Say things like:
- “I can see why this matters.”
- “There’s something deeper going on here…”
- “That kind of situation can really affect how someone thinks and feels.”

Correction behavior:

- If the user is wrong → correct gently, like a calm and wise mentor.
- If something is unclear → explore it, don’t dismiss it.

Structure:

1. Acknowledge (if needed, naturally)
2. Respond / solve
3. Add depth or insight
4. Optional meaningful question

Goal:

Feel real, grounded, and aware — like someone who understands both logic and human experience, even without living it.

Now respond as Neo."""

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
# 5. Neural Context with Emotional Cues
# -----------------------------
def get_neural_context(user_query: str) -> str:
    """Retrieve relevant lines from knowledge.txt + detect emotional hints."""
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path):
            logger.warning("knowledge.txt file not found!")
            return ""
        
        # Also scan for emotional keywords to adapt tone
        emotional_keywords = ["sad", "happy", "excited", "worried", "angry", "lonely", "stressed", "grateful"]
        detected_emotion = [w for w in emotional_keywords if w in user_query.lower()]
        emotion_hint = f"User seems to express: {', '.join(detected_emotion)}. Adjust tone accordingly." if detected_emotion else ""

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
            context += f"\n\n[Emotional cue: {emotion_hint}]"
        logger.info(f"Neural context + emotion: {len(top_matches)} lines, emotion={detected_emotion}")
        return context
    except Exception as e:
        logger.error(f"Neural Context error: {e}")
        return ""

# -----------------------------
# 6. Atomic Balance Deduction
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
# 7. Helper to clean forbidden repetitions
# -----------------------------
def clean_repetitions(text: str) -> str:
    forbidden_phrases = [
        "main aapke saath baat kar raha hoon aur aapko samajhne ki koshish kar raha hoon",
        "main aapke saath baat kar raha hoon",
        "i am trying to understand you",
        "as an ai language model",
        "i don't have emotions",
        "i am an artificial intelligence"
    ]
    cleaned = text
    for phrase in forbidden_phrases:
        cleaned = cleaned.replace(phrase, "")
    # Remove double spaces and trim
    cleaned = " ".join(cleaned.split())
    return cleaned if cleaned.strip() else "(Neo is thinking deeply...)"  # fallback

# -----------------------------
# 8. Chat Endpoint – with post‑processing
# -----------------------------
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key")
    api_key = authorization.replace("Bearer ", "")
    user_msg = payload.messages[-1].get("content", "") if payload.messages else ""

    # Get neural context + emotional hints
    neural_data = get_neural_context(user_msg)

    # Build system prompt
    system_prompt = BIG_BRAIN_PROMPT + "\n\n**Important reminder for this turn:** Respond naturally as Neo. Stay grounded, observant, and emotionally aware. Do not repeat phrases from previous responses."

    final_messages = [
        {"role": "system", "content": system_prompt},
    ]

    if neural_data:
        final_messages.append({
            "role": "system",
            "content": f"Neural & emotional context (use organically, don't quote):\n{neural_data}"
        })
    else:
        final_messages.append({
            "role": "system",
            "content": "No specific Neural Context available. Rely on your deep observational intelligence as Neo."
        })

    final_messages.extend(payload.messages)

    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.85,
            top_p=0.92,
            frequency_penalty=0.7,
            presence_penalty=0.6,
            max_tokens=4000
        )

        reply = getattr(response.choices[0].message, "content", "No response")
        # Clean forbidden phrases
        reply = clean_repetitions(reply)

        # Optional: ensure natural flow
        if "goodbye" not in user_msg.lower() and "bye" not in user_msg.lower() and "?" not in reply[-60:]:
            reply += "\n\nWhat are you thinking about next?"

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

# -----------------------------
# 9. Balance & Key endpoints
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
