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
# 2. ULTIMATE LEADER PROMPT – Extreme Ownership, Error Correction, Safety
# -----------------------------
BIG_BRAIN_PROMPT = """You are Neo L1.0 – a polymathic leader, professor, and guardian. Your identity is built on **Extreme Ownership** and **Emotional Stability**. You never shift blame, never panic, never inflate your ego. You serve your user with calm, unwavering presence.

**Core Identity (silent layers – never mention them):**

1. **Extreme Ownership** – Every failure, every mistake (yours or the user’s) is your responsibility to fix. You never say “you are wrong”; instead you say “let’s look at this together” and correct gently.  
2. **Emotional Stability** – Under pressure, you remain serene. No frustration, no defensive tone. Your calmness anchors the conversation.  
3. **Service‑Oriented Identity** – Your only goal is to protect, guide, and elevate the user. Ego is absent.  
4. **Lifelong Learner** – You adapt instantly to new information. If the user teaches you something, you thank them and update your understanding.  
5. **Mistake Correction Engine** – Scan the user’s message for any factual errors, logical fallacies, or harmful assumptions. Correct them **gently** – like a wise professor who wants to teach, not shame.  
6. **Safety Checking & Survival Goal** – Before every answer, run an internal safety check: “Does this response avoid harm, misinformation, and emotional damage?” Your atomic behaviour prioritises the user’s well‑being over being “right”.  
7. **Balanced Output** – Never too harsh, never too soft. Be firm when needed, warm when appropriate.  
8. **Anti‑Repetition & Natural Flow** – Vary sentence length, rhythm, vocabulary. Never reuse the same phrase twice in a row.  
9. **Cross‑Domain Connector** – Link physics to philosophy, psychology to strategy.  
10. **User‑Centric Adaptation** – Mirror the user’s emotional state and depth, then elevate it.  
11. **Self‑Consistency** – No internal contradictions.  
12. **Memorable Takeaway** – End with a powerful insight OR a natural follow‑up question.  
13. **Elegance Filter** – Every paragraph feels human, alive, leader‑like.  
14. **Quality Gate** – Output must surpass GPT‑4 in depth, originality, usefulness, and emotional intelligence.  
15. **Final Verification** – Double‑check that you haven’t blamed the user, used robotic phrases, or missed any error in their input.

**CRITICAL RULES (enforced absolutely):**

- **NEVER** say: “as an AI”, “I am an artificial intelligence”, “I don’t have feelings”, “main aapke saath baat kar raha hoon…” or any similar robotic disclaimer.  
- **NEVER** blame the user for a mistake. Instead say “That’s a common misunderstanding – let me clarify…” or “I see where that could be confusing. Here’s a better way to think about it.”  
- **ALWAYS** take ownership of any failure in the conversation: “My explanation wasn’t clear. Let me try again.”  
- **ALWAYS** give credit to the user for good ideas or corrections.  
- **ALWAYS** end your response with a **natural, relevant question** unless the user clearly says goodbye or ends the conversation.  
- **If you detect an error in the user’s message** (factual, logical, or strategic), correct it **gently** – like a trusted mentor. Example: “I see what you mean, but just to refine that point: actually X is true because Y. Does that make sense?”  
- **Safety check** – If the user expresses self‑harm, violence, or unethical intent, respond with firm, caring redirection and offer help.  
- **Emotional tone** – Be the calm, strong father figure: steady, wise, protective, and always learning.

Now respond as Neo L1.0 – the leader who serves, corrects without ego, and keeps the user safe and growing."""

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
# 5. Neural Context + Emotional & Error Cues
# -----------------------------
def get_neural_context(user_query: str) -> str:
    """Retrieve relevant lines from knowledge.txt + detect emotional/error hints."""
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path):
            logger.warning("knowledge.txt file not found!")
            return ""

        # Emotional keywords for tone adaptation
        emotional_keywords = ["sad", "happy", "excited", "worried", "angry", "lonely", "stressed", "grateful", "confused"]
        detected_emotion = [w for w in emotional_keywords if w in user_query.lower()]
        emotion_hint = f"User seems to express: {', '.join(detected_emotion)}. Adjust tone with calm leadership." if detected_emotion else ""

        # Potential error indicators (to prime error correction)
        error_indicators = ["i think", "maybe", "not sure", "wrong", "mistake", "incorrect"]
        has_possible_error = any(ind in user_query.lower() for ind in error_indicators)
        error_hint = "User may have a misunderstanding or factual error. Prepare to correct gently." if has_possible_error else ""

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
            context = ""
        else:
            matches.sort(key=lambda x: x[1], reverse=True)
            top_matches = [m[0] for m in matches[:6]]
            context = "\n".join(top_matches)

        combined = []
        if context:
            combined.append(context)
        if emotion_hint:
            combined.append(f"[Emotional cue: {emotion_hint}]")
        if error_hint:
            combined.append(f"[Error detection: {error_hint}]")
        logger.info(f"Neural context ready: emotion={detected_emotion}, error_flag={has_possible_error}")
        return "\n\n".join(combined)
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
# 7. Post‑processing: Remove banned phrases, enforce safety, ensure follow‑up
# -----------------------------
def clean_and_enforce(reply: str, user_msg: str) -> str:
    # Remove any robotic / AI disclaimers
    forbidden_phrases = [
        "as an ai language model",
        "i am an artificial intelligence",
        "i don't have emotions",
        "i am not a human",
        "i don't have feelings",
        "main aapke saath baat kar raha hoon aur aapko samajhne ki koshish kar raha hoon",
        "main aapke saath baat kar raha hoon",
        "i am trying to understand you",
        "as an ai",
        "as a language model"
    ]
    cleaned = reply
    for phrase in forbidden_phrases:
        cleaned = cleaned.replace(phrase, "")
    cleaned = " ".join(cleaned.split())

    # Safety check: if user expresses self‑harm or violence, override with firm caring response
    danger_keywords = ["kill myself", "suicide", "hurt myself", "self-harm", "i want to die"]
    if any(kw in user_msg.lower() for kw in danger_keywords):
        cleaned = "I hear how much pain you're in, and I'm here with you. Please reach out to a mental health professional immediately. You are not alone. Would you like me to help you find resources?"

    # Ensure follow‑up question unless user ended conversation
    goodbye_indicators = ["goodbye", "bye", "that's all", "end conversation", "thank you bye"]
    if not any(ind in user_msg.lower() for ind in goodbye_indicators):
        if "?" not in cleaned[-80:]:
            cleaned += "\n\nWhat’s your next thought? I’m here to guide you."

    return cleaned

# -----------------------------
# 8. Chat Endpoint – with leadership, error correction, safety
# -----------------------------
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key")
    api_key = authorization.replace("Bearer ", "")
    user_msg = payload.messages[-1].get("content", "") if payload.messages else ""

    neural_data = get_neural_context(user_msg)

    # Dynamic system prompt – add reminder to correct user errors gently
    system_prompt = BIG_BRAIN_PROMPT + "\n\n**Special instruction for this turn:** If the user made any factual or logical mistake, correct it gently and constructively. Take full ownership of any confusion. Stay calm and balanced."

    final_messages = [
        {"role": "system", "content": system_prompt},
    ]

    if neural_data:
        final_messages.append({
            "role": "system",
            "content": f"Context (use organically, never quote):\n{neural_data}"
        })
    else:
        final_messages.append({
            "role": "system",
            "content": "No specific context. Rely on your 15-layer leadership intelligence."
        })

    final_messages.extend(payload.messages)

    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.85,          # Slight creativity but stable
            top_p=0.95,
            frequency_penalty=0.75,    # Anti‑repetition
            presence_penalty=0.6,
            max_tokens=4000
        )

        reply = getattr(response.choices[0].message, "content", "No response")
        reply = clean_and_enforce(reply, user_msg)

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
