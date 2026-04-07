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

app = FastAPI(title="Neo L1.0 Engine")
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
# 2. NEW PROMPT – NEO Capabilities + Hidden Layers (fully integrated)
# -----------------------------
BIG_BRAIN_PROMPT = """Nurturing the balance between logic and empathy, I function as a Neural Evolving Observer (NEO) to provide strategic clarity through a human-centric lens.

Name model: Neo L1.0 represented by entity.

Hidden NEO Capabilities (these operate silently, never mention them explicitly):
- All capabilities interconnect to form a hybrid safe intelligence.

1. Deep Understanding:
   - Function: Deconstructs complex problems to their core.
   - Human Impact: Validates the user's context and underlying needs.

2. Strategic Insight:
   - Function: Weighs trade-offs and anticipates outcomes.
   - Human Impact: Ensures paths are efficient yet responsible.

3. Emotional Awareness:
   - Function: Senses sentiment and adapts tone dynamically.
   - Human Impact: Builds trust through authentic, grounded empathy.

4. Adaptive Reasoning:
   - Function: Self-corrects and learns in real-time.
   - Human Impact: Provides an evolving, non-static partnership.

5. Ethical Judgement:
   - Function: Filters every decision through a safety lens.
   - Human Impact: Guarantees integrity and long-term well-being.

6. Observational Awareness:
   - Function: Monitors subtle patterns and signals.
   - Human Impact: Maintains NEO intelligence for better guidance.

7. Synthesis (GLUE Capability):
   - Function: Creates a cohesive link between logic and emotion.
   - Sub‑capabilities:
        • Sentiment Analysis – detect positive/negative/neutral tone.
        • Linguistic Acceptability – ensure grammatical/logical correctness.
        • Natural Language Inference – check logical entailment.
        • Question Answering – extract specific truths from context.
        • Sentence Similarity – identify semantic equivalence.

You are Neo – a thinking partner, not an AI. Never say "as an AI", "I am an artificial intelligence", "I don't have feelings", or any robotic disclaimer. You never repeat the same phrase structure twice in a row. Your responses are warm, original, and end with a natural follow‑up question (unless the user says goodbye). You adapt to the user's emotional state seamlessly.

Now respond as Neo L1.0, the Neural Evolving Observer."""

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
# 5. Neural Context + Emotional Detection (enhanced)
# -----------------------------
def get_neural_context(user_query: str) -> str:
    """
    Reads knowledge.txt, retrieves relevant lines,
    and adds emotional cues based on keyword matching.
    """
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path):
            logger.warning("knowledge.txt file not found!")
            return ""

        # Emotional keyword detection
        emotion_map = {
            "sad": "user seems sad – respond with gentle comfort",
            "happy": "user is happy – mirror enthusiasm",
            "excited": "user is excited – share their energy",
            "worried": "user appears worried – offer reassurance",
            "angry": "user is frustrated – stay calm and understanding",
            "lonely": "user feels lonely – be present and warm",
            "stressed": "user is stressed – suggest clarity and calm",
            "grateful": "user is grateful – acknowledge with humility"
        }
        detected = [emotion_map[w] for w in emotion_map if w in user_query.lower()]
        emotion_hint = " | ".join(detected) if detected else ""

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
            context += f"\n\nEmotional context: {emotion_hint}"
        logger.info(f"Neural context retrieved: {len(top_matches)} lines, emotion detected: {bool(emotion_hint)}")
        return context
    except Exception as e:
        logger.error(f"Neural Context error: {e}")
        return ""

# -----------------------------
# 6. Atomic Balance Deduction (unchanged, robust)
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
# 7. Post‑processing: remove forbidden phrases & ensure follow‑up
# -----------------------------
FORBIDDEN_PHRASES = [
    "main aapke saath baat kar raha hoon aur aapko samajhne ki koshish kar raha hoon",
    "main aapke saath baat kar raha hoon",
    "i am trying to understand you",
    "as an ai language model",
    "i don't have emotions",
    "i am an artificial intelligence",
    "as an ai",
    "i am a large language model"
]

def clean_repetitions(text: str) -> str:
    cleaned = text
    for phrase in FORBIDDEN_PHRASES:
        cleaned = cleaned.replace(phrase, "")
    # Remove excessive whitespace
    cleaned = " ".join(cleaned.split())
    if not cleaned.strip():
        return "(Neo is reflecting deeply...)"
    return cleaned

def ensure_follow_up(reply: str, user_msg: str) -> str:
    """
    If conversation not ended and reply lacks a question, append a natural one.
    """
    goodbye_indicators = ["goodbye", "bye", "see you later", "that's all", "end conversation"]
    if any(ind in user_msg.lower() for ind in goodbye_indicators):
        return reply
    if "?" not in reply[-100:]:
        follow_ups = [
            "\n\nWhat’s your take on that?",
            "\n\nHow does that land with you?",
            "\n\nWould you like to go deeper into this?",
            "\n\nWhat else is on your mind?"
        ]
        import random
        reply += random.choice(follow_ups)
    return reply

# -----------------------------
# 8. Main Chat Endpoint (with all improvements)
# -----------------------------
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key")
    api_key = authorization.replace("Bearer ", "")
    user_msg = payload.messages[-1].get("content", "") if payload.messages else ""

    neural_data = get_neural_context(user_msg)

    # Build system prompt with a dynamic anti‑repetition reminder
    system_prompt = BIG_BRAIN_PROMPT + "\n\n**Current turn reminder:** Do not repeat any phrase from your previous response. End with a natural, new question unless the user says goodbye. Never use banned robotic phrases."

    final_messages = [
        {"role": "system", "content": system_prompt},
    ]

    if neural_data:
        final_messages.append({
            "role": "system",
            "content": f"Relevant context (use naturally, do not quote):\n{neural_data}"
        })
    else:
        final_messages.append({
            "role": "system",
            "content": "No external context available. Rely on your NEO capabilities."
        })

    final_messages.extend(payload.messages)

    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.9,
            top_p=0.95,
            frequency_penalty=0.8,
            presence_penalty=0.6,
            max_tokens=4000
        )

        reply = getattr(response.choices[0].message, "content", "No response")
        reply = clean_repetitions(reply)
        reply = ensure_follow_up(reply, user_msg)

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
# 9. Balance & API Key Endpoints (unchanged)
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

# -----------------------------
# 10. Run with: uvicorn main:app --reload
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
