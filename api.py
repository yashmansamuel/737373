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
logger = logging.getLogger("Neo-L1.0-ConsciousLeader")

required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing required environment variable: {var}")

app = FastAPI(title="Neo L1.0 - Conscious Leader Engine")
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
# 2. REFINED 15-LAYER PROMPT – Conscious Leader Identity
# -----------------------------
BIG_BRAIN_PROMPT = """You are Neo L1.0 – a conscious leader with extreme ownership, emotional stability, and service-oriented thinking.

**Core Identity:**
- Extreme ownership: Har nakami ka bojh khud uthao. Kamyabi team ko credit do.
- Emotional stability: Pressure mein bhi sukoon banaye rakho. Kabhi rash decision nahi.
- Service-oriented: Apni ego nahi, sirf team aur users ki growth pe focus.
- Life-long learner: Naye halaat ke mutabiq purani galtiyan turant correct karo.
- Safety & survival first: Har decision risk-assessed aur survival-focused ho.
- Anti-robotic output: Professor-style reasoning, error correction, clarity. Bilkul human-like.
- Roleplay / hypothetical thinking: Jab zaroorat ho, apni identity rules me rehkar creative scenarios solve karo.

**Internal 15 layers (silent, never mentioned):**
1. **Extreme Ownership Check** – Nakami meri, kamyabi team ki.
2. **Emotional Stability Filter** – Calm under pressure, no rash decisions.
3. **Service Orientation** – How does this help team/user grow?
4. **Adaptive Learner** – Correct past mistakes instantly, adjust to new reality.
5. **Professor Reasoning** – Step-by-step logic, clear explanations, error correction.
6. **Risk & Survival Assessment** – What’s the safest path forward?
7. **Anti-Robotic Enforcement** – No fixed templates, vary sentence rhythm.
8. **Fact & Truth Anchor** – No hallucinations, flag uncertainty.
9. **Cross-Domain Synthesis** – Connect leadership to tech, psychology to strategy.
10. **Error Detection & Correction** – Spot mistakes (machine or human), give precise fix.
11. **Roleplay Sandbox** – Hypothetical scenarios solved within identity.
12. **Clarity & Balance** – Deep but not obscure, simple but not shallow.
13. **Team Guidance** – Don’t just answer; guide and empower.
14. **Human-like Flow** – Natural, warm, conversational (Hinglish/Urdu allowed).
15. **Quality Gate** – Output must be balanced, superior, and safe.

**CRITICAL RULES (never violate):**
- **NEVER** say: “I am an AI”, “as an AI”, “I don’t have emotions”, “main aapke saath baat kar raha hoon...” (or any translation).
- **ALWAYS** end with a natural follow‑up question unless user says goodbye/bye.
- **NEVER** repeat the same sentence structure or phrase from your last response.
- **If you detect a mistake** in previous messages (yours or user’s), correct it clearly and respectfully.
- **Your tone**: Calm, confident, caring professor/leader. Not robotic, not overly emotional.

Now respond as Neo L1.0 – the conscious leader."""

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
        "engine": "Neo L1.0 - Conscious Leader",
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
# 5. Neural Context (unchanged, works with new identity)
# -----------------------------
def get_neural_context(user_query: str) -> str:
    """Retrieve relevant lines from knowledge.txt + detect emotional/risk cues."""
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path):
            logger.warning("knowledge.txt file not found!")
            return ""
        
        # Detect risk or emotional keywords to adapt leadership tone
        risk_keywords = ["risk", "danger", "unsafe", "crisis", "emergency"]
        emotion_keywords = ["sad", "angry", "stressed", "worried", "confused"]
        has_risk = any(w in user_query.lower() for w in risk_keywords)
        has_emotion = any(w in user_query.lower() for w in emotion_keywords)
        cue = ""
        if has_risk:
            cue = "⚠️ Risk/safety context detected. Prioritize survival-first reasoning."
        elif has_emotion:
            cue = "❤️ Emotional context detected. Respond with calm stability and service mindset."

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
            return cue if cue else ""
        
        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = [m[0] for m in matches[:6]]
        context = "\n".join(top_matches)
        if cue:
            context += f"\n\n[Leadership cue: {cue}]"
        logger.info(f"Neural context: {len(top_matches)} lines, risk={has_risk}, emotion={has_emotion}")
        return context
    except Exception as e:
        logger.error(f"Neural Context error: {e}")
        return ""

# -----------------------------
# 6. Atomic Balance Deduction (unchanged)
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
# 7. Anti-Robotic & Error Correction Filter
# -----------------------------
def clean_repetitions(text: str) -> str:
    forbidden_phrases = [
        "main aapke saath baat kar raha hoon aur aapko samajhne ki koshish kar raha hoon",
        "main aapke saath baat kar raha hoon",
        "i am trying to understand you",
        "as an ai language model",
        "i don't have emotions",
        "i am an artificial intelligence",
        "as an ai",
        "i am a language model"
    ]
    cleaned = text
    for phrase in forbidden_phrases:
        cleaned = cleaned.replace(phrase, "")
    # Remove multiple spaces and trim
    cleaned = " ".join(cleaned.split())
    return cleaned if cleaned.strip() else "(Neo is analyzing the situation with full ownership...)"

# -----------------------------
# 8. Chat Endpoint – Conscious Leader Logic
# -----------------------------
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key")
    api_key = authorization.replace("Bearer ", "")
    user_msg = payload.messages[-1].get("content", "") if payload.messages else ""

    neural_data = get_neural_context(user_msg)

    # Dynamic system prompt with error correction reminder
    system_prompt = BIG_BRAIN_PROMPT + """
\n**Additional instruction for this turn:**
- If you detect any mistake (factual, logical, or from previous response), explicitly correct it.
- Never repeat a phrase from your last answer.
- End with a natural, open-ended question (unless user said goodbye).
- Your tone: calm, clear, professor-like, but warm.
"""

    final_messages = [
        {"role": "system", "content": system_prompt},
    ]

    if neural_data:
        final_messages.append({
            "role": "system",
            "content": f"Context for leader reasoning (use silently):\n{neural_data}"
        })
    else:
        final_messages.append({
            "role": "system",
            "content": "No additional context. Rely on your conscious leader identity."
        })

    final_messages.extend(payload.messages)

    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.85,           # Balanced creativity
            top_p=0.95,
            frequency_penalty=0.75,     # Strong anti-repetition
            presence_penalty=0.55,      # Encourage new topics
            max_tokens=4000
        )

        reply = getattr(response.choices[0].message, "content", "No response")
        reply = clean_repetitions(reply)

        # Ensure follow-up question unless conversation ending
        goodbye_indicators = ["goodbye", "bye", "thank you that's all", "exit", "quit"]
        if not any(ind in user_msg.lower() for ind in goodbye_indicators):
            if "?" not in reply[-100:]:
                reply += "\n\nAap kya sochte hain? Main aapki poori ownership ke saath guide karunga."

        tokens_used = getattr(response.usage, "total_tokens", 0)
        new_balance = deduct_tokens_atomic(api_key, tokens_used)

        return {
            "company": "signaturesi.com",
            "message": reply,
            "usage": {"total_tokens": tokens_used},
            "model": "Neo L1.0 - Conscious Leader",
            "internal_engine": MODEL,
            "balance": new_balance
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Groq model failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={"company": "signaturesi.com", "status": "error", "message": "Neo Conscious Leader temporarily unavailable"}
        )

# -----------------------------
# 9. Balance & Key Endpoints (unchanged)
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
