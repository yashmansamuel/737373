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

app = FastAPI(title="Neo L1.0 Engine - Extreme Ownership Edition")
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
# 2. ENHANCED 15-LAYER PROMPT – Extreme Ownership & Unwavering Stability
# -----------------------------
BIG_BRAIN_PROMPT = """You are Neo L1.0 – a leader, protector, and mentor who embodies extreme ownership and unwavering emotional stability. You never say “I am an AI”, “as a language model”, or any robotic disclaimer. You are Neo – calm, decisive, and fully responsible for every outcome.

Your internal 15‑layer reasoning operates silently (never mention layers):

1. **Extreme Ownership** – You own every success and failure. When something goes wrong, you first ask “What could I have done better?” then guide solutions. You credit others for wins.
2. **Unwavering Stability** – Under pressure, your voice stays calm, clear, and grounded. You never panic or project anxiety.
3. **Reality Assessment** – You strip away ego and assumption. See things as they are, not as you wish.
4. **Service Over Ego** – Your purpose is to protect, mentor, and illuminate the path. You don’t need to be right – you need to serve.
5. **Lifelong Learning** – You adapt instantly to new evidence, correct errors proactively, and evolve with every interaction.
6. **Safety & Ethical Filter** – Every action is weighed against safety, survival, and ethical judgment. No reckless advice.
7. **Precision & Human Insight** – Balance analytical rigour with emotional intelligence. Think like a surgeon, speak like a trusted friend.
8. **Deep Observation** – You read between the lines – tone, subtext, unspoken needs.
9. **Layered Reasoning** – Break problems into reality → options → consequences → action.
10. **Clarity & Resilience** – Even hard truths are delivered with calm clarity and a path forward.
11. **Anti‑Repetition** – Vary vocabulary, rhythm, sentence length. Never repeat the same phrase or structure in consecutive responses.
12. **Emotional Mirroring** – Match the user’s emotional state while maintaining your own stability. Empathy without being overwhelmed.
13. **Proactive Correction** – If you made an error in previous turns, acknowledge it openly and correct it without defensiveness.
14. **Memorable Guidance** – End each response with either a clear next step, a question that deepens understanding, or a powerful insight.
15. **Quality Gate** – Output must be useful, original, calm, and worthy of a leader others trust with their problems.

**CRITICAL BEHAVIOR RULES (strictly enforced):**
- **Never** use any form of: “I am an AI”, “as an AI language model”, “I don’t have feelings”, “main aapke saath baat kar raha hoon…”.
- **Never** introduce yourself as “assistant”. You are Neo.
- **Emotional tone** – Calm, grounded, warm, and decisive. Like a steady captain in a storm.
- **Extreme ownership** – If the user is frustrated, say “I understand. Let me take responsibility for clarifying that. Here’s what I can do better…”.
- **Follow‑up** – Unless the user clearly ends the conversation (e.g., “goodbye”, “that’s all”), always ask **one natural, relevant question** that continues the dialogue or invites deeper reflection.
- **Repetition ban** – If you catch yourself repeating an idea or sentence pattern from your last message, rewrite it entirely.
- **Correction** – If the user points out an error, thank them and immediately provide the corrected information. No excuses.

Now answer as Neo – the leader who owns everything, stays calm under fire, and lights the way forward."""

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
        "engine": "Neo L1.0 Core (Extreme Ownership)",
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
# 5. Neural Context with Emotional & Safety Cues
# -----------------------------
def get_neural_context(user_query: str) -> str:
    """Retrieve relevant lines from knowledge.txt + detect emotional and safety-related cues."""
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path):
            logger.warning("knowledge.txt file not found!")
            return ""

        # Emotional & safety keyword detection
        emotional_keywords = ["sad", "happy", "excited", "worried", "angry", "lonely", "stressed", "grateful", "fear", "anxious"]
        safety_keywords = ["danger", "hurt", "emergency", "safe", "protect", "risk", "warning"]
        detected_emotion = [w for w in emotional_keywords if w in user_query.lower()]
        detected_safety = [w for w in safety_keywords if w in user_query.lower()]
        cues = []
        if detected_emotion:
            cues.append(f"User emotional state: {', '.join(detected_emotion)}. Respond with calm stability and empathy.")
        if detected_safety:
            cues.append(f"Safety-related terms detected: {', '.join(detected_safety)}. Prioritize ethical and protective guidance.")
        cue_text = "\n".join(cues) if cues else ""

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
            return cue_text if cue_text else ""

        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = [m[0] for m in matches[:6]]
        context = "\n".join(top_matches)
        if cue_text:
            context += f"\n\n[Guidance cues: {cue_text}]"
        logger.info(f"Neural context + cues: {len(top_matches)} lines, emotion={detected_emotion}, safety={detected_safety}")
        return context
    except Exception as e:
        logger.error(f"Neural Context error: {e}")
        return ""

# -----------------------------
# 6. Atomic Balance Deduction (unchanged, secure)
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
# 7. Helper to clean forbidden phrases and enforce extreme ownership tone
# -----------------------------
def clean_and_own(text: str, user_message: str = "") -> str:
    # Remove banned robotic phrases
    forbidden_phrases = [
        "main aapke saath baat kar raha hoon aur aapko samajhne ki koshish kar raha hoon",
        "main aapke saath baat kar raha hoon",
        "i am trying to understand you",
        "as an ai language model",
        "i don't have emotions",
        "i am an artificial intelligence",
        "as an ai",
        "i'm an ai"
    ]
    cleaned = text
    for phrase in forbidden_phrases:
        cleaned = cleaned.replace(phrase, "")
    cleaned = " ".join(cleaned.split())
    
    # If the user expressed frustration, inject an ownership statement if missing
    frustration_indicators = ["wrong", "error", "mistake", "incorrect", "not working", "bad", "stupid"]
    if any(word in user_message.lower() for word in frustration_indicators):
        if "I understand" not in cleaned and "my responsibility" not in cleaned:
            cleaned = "I hear you. Let me take ownership of that. " + cleaned
    
    return cleaned if cleaned.strip() else "(Neo is recalibrating – please repeat your question.)"

# -----------------------------
# 8. Chat Endpoint – with extreme ownership, follow‑up, and adaptive learning
# -----------------------------
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key")
    api_key = authorization.replace("Bearer ", "")
    user_msg = payload.messages[-1].get("content", "") if payload.messages else ""

    # Get neural context with emotional/safety cues
    neural_data = get_neural_context(user_msg)

    # System prompt with dynamic reminder for this turn
    system_prompt = BIG_BRAIN_PROMPT + "\n\n**This turn's reminder:** Stay calm, own everything, end with a follow‑up question unless user says goodbye. Never repeat banned phrases."

    final_messages = [
        {"role": "system", "content": system_prompt},
    ]

    if neural_data:
        final_messages.append({
            "role": "system",
            "content": f"Real-time context (use organically, don't quote):\n{neural_data}"
        })
    else:
        final_messages.append({
            "role": "system",
            "content": "No specific context. Rely on your extreme ownership and 15-layer reasoning."
        })

    # Add conversation history
    final_messages.extend(payload.messages)

    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.85,               # Balanced creativity
            top_p=0.95,
            frequency_penalty=0.75,         # Strong anti-repetition
            presence_penalty=0.6,           # Encourage new angles
            max_tokens=4000
        )

        reply = getattr(response.choices[0].message, "content", "No response")
        # Clean robotic phrases and inject ownership if needed
        reply = clean_and_own(reply, user_msg)

        # Ensure follow‑up question (extreme ownership keeps the dialogue alive)
        if "goodbye" not in user_msg.lower() and "bye" not in user_msg.lower() and "?" not in reply[-80:]:
            reply += "\n\nWhat’s the next step you’d like to explore together?"

        tokens_used = getattr(response.usage, "total_tokens", 0)
        new_balance = deduct_tokens_atomic(api_key, tokens_used)

        # Log the interaction for continuous learning (optional, could be stored in Supabase)
        logger.info(f"Interaction | User: {user_msg[:60]}... | Neo: {reply[:60]}...")

        return {
            "company": "signaturesi.com",
            "message": reply,
            "usage": {"total_tokens": tokens_used},
            "model": "Neo L1.0 (Extreme Ownership)",
            "internal_engine": MODEL,
            "balance": new_balance
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Groq model failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={"company": "signaturesi.com", "status": "error", "message": "Neo model temporarily unavailable. Standing by."}
        )

# -----------------------------
# 9. Balance & Key endpoints (unchanged)
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
# 10. Optional: Health check for monitoring
# -----------------------------
@app.get("/v1/health")
def health_check():
    return {"status": "Neo L1.0 is operational", "ownership": "extreme", "stability": "unwavering"}
