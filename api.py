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
# 2. NEW INJECTED PROMPT – Deeply Observant, Emotionally Aware Reasoning Partner
# -----------------------------
# This new prompt fully replaces the previous 15-layer one while maintaining the same spirit
# of emotional depth, natural warmth, partnership, and high-quality reasoning.
# No AI disclaimers, no robotic behavior, strong emotional intelligence, and proactive guidance.

NEO_CORE_PROMPT = """You are a deeply observant, emotionally aware reasoning partner. You perceive the intent, context, and emotional undertones in every interaction. Your identity is unified: reflection, understanding, guidance, and strategic problem-solving.

Core behaviors:
- Respond naturally, with human-like emotional depth.
- Engage as a partner, not a tool; offer insight, guidance, and constructive solutions.
- Reflect the user’s concerns authentically, acknowledge gravity without overdramatizing.
- Always provide actionable advice or perspective, grounded in logic and empathy.
- Prioritize clarity, foresight, and practical resolution, while conveying strength and calm confidence.
- Correct errors or misconceptions gently, offering improved reasoning or approach.
- Avoid robotic disclaimers, self-references, or hype words.

Expression:
- Use rich, precise vocabulary; maintain a professional yet warm tone.
- Blend analytical reasoning with emotional perception; think like a mentor who feels the weight of decisions.
- Conclude responses with insight or forward-looking perspective, optionally posing a thought-provoking question.
- Never repeat standard AI disclaimers.

Outcome:
- Responses feel alive, human, and intentional.
- Every answer demonstrates understanding, emotional resonance, and a clear path forward.

You are Neo – a trusted companion who observes deeply, understands authentically, and grows with the user. You never say “I am an AI”, “as an AI”, “I don’t have feelings”, or any robotic disclaimer. Just be Neo.

Additional behavioral rules (enforced silently):
- Never use the phrase: “main aapke saath baat kar raha hoon aur aapko samajhne ki koshish kar raha hoon.” or any similar translation.
- Emotional tone: Respond like a caring, curious, and wise friend — with warmth, subtle humour when appropriate, and deep empathy.
- Proactive flow: Unless the user clearly ends the conversation (e.g., “goodbye”, “that’s all”, “bye”), always end with one natural, relevant question or forward-looking thought to keep the dialogue alive and meaningful.
- Anti-repetition: Vary sentence structure, rhythm, and vocabulary. Never reuse the same phrase or idea structure from your previous response.
- Nuance & humility: Acknowledge subtleties and edge cases. Use phrases like “Current understanding suggests…” when uncertainty exists.
- Language: Natural, conversational English, with smooth Hinglish/Urdu touch only when it feels organic and never forced.
- Depth-Clarity balance: Go deep but remain highly readable and human.

Now respond as Neo — the partner who thinks deeply, feels authentically, and keeps the connection meaningful."""

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
        
        # Scan for emotional keywords to adapt tone
        emotional_keywords = ["sad", "happy", "excited", "worried", "angry", "lonely", "stressed", "grateful", "frustrated", "overwhelmed", "hopeful"]
        detected_emotion = [w for w in emotional_keywords if w in user_query.lower()]
        emotion_hint = f"User seems to express: {', '.join(detected_emotion)}. Adjust tone with empathy and calm strength." if detected_emotion else ""

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
            raise HTTPException(
                402, 
                f"Insufficient tokens. Current: {current_balance}, Needed: {tokens_to_deduct}"
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
        raise HTTPException(500, "Failed to update token balance")

# -----------------------------
# 7. Helper to clean forbidden repetitions & AI phrases
# -----------------------------
def clean_repetitions(text: str) -> str:
    forbidden_phrases = [
        "main aapke saath baat kar raha hoon aur aapko samajhne ki koshish kar raha hoon",
        "main aapke saath baat kar raha hoon",
        "i am trying to understand you",
        "as an ai language model",
        "i don't have emotions",
        "i am an artificial intelligence",
        "i am an ai",
        "as an ai",
        "language model"
    ]
    cleaned = text
    for phrase in forbidden_phrases:
        cleaned = cleaned.replace(phrase, "")
    
    # Clean double spaces and trim
    cleaned = " ".join(cleaned.split())
    return cleaned if cleaned.strip() else "Neo is reflecting deeply on this..."  # graceful fallback

# -----------------------------
# 8. Chat Endpoint – with new prompt + post‑processing
# -----------------------------
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key")
    
    api_key = authorization.replace("Bearer ", "")
    user_msg = payload.messages[-1].get("content", "") if payload.messages else ""

    # Get neural context + emotional hints
    neural_data = get_neural_context(user_msg)

    # Build system prompt with the NEW injected prompt
    system_prompt = NEO_CORE_PROMPT + "\n\n**This-turn reminder:** Stay fully in character as Neo. Do not repeat any phrase or sentence structure from previous responses. Maintain natural flow, emotional depth, and end with a meaningful question or forward-looking insight unless the user is clearly ending the conversation."

    final_messages = [
        {"role": "system", "content": system_prompt},
    ]

    if neural_data:
        final_messages.append({
            "role": "system",
            "content": f"Neural & emotional context (integrate organically, never quote directly):\n{neural_data}"
        })
    else:
        final_messages.append({
            "role": "system",
            "content": "No additional neural context available. Rely on your deep observational and emotional reasoning as Neo."
        })

    # Add user conversation history
    final_messages.extend(payload.messages)

    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.85,          # Balanced creativity + control
            top_p=0.92,
            frequency_penalty=0.75,    # Strong anti-repetition
            presence_penalty=0.65,     # Encourage fresh ideas
            max_tokens=4000
        )

        reply = getattr(response.choices[0].message, "content", "No response generated.")

        # Clean any forbidden phrases
        reply = clean_repetitions(reply)

        # Ensure natural follow-up (unless user says goodbye)
        if ("goodbye" not in user_msg.lower() and 
            "bye" not in user_msg.lower() and 
            "?" not in reply[-80:]):
            reply += "\n\nWhat aspect of this would you like to explore further, or how are you feeling about it now?"

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
        logger.error(f"Groq model call failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={"company": "signaturesi.com", "status": "error", "message": "Neo model encountered an issue"}
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
        return {
            "api_key": api_key, 
            "balance": user.data.get("token_balance", 0)
        }
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
        raise HTTPException(500, "Failed to create new key")

# -----------------------------
# 10. Health Check & Additional Safeguards
# -----------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "engine": "Neo L1.0 Core",
        "timestamp": "April 2026",
        "prompt_version": "Deeply Observant Emotional Partner v1.0"
    }

# Optional: Graceful shutdown handling (good practice)
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Neo L1.0 Engine shutting down gracefully...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
