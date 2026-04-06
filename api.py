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
# 2. NEW PROMPT – Leading Father / Mentor (GPT-5 level authority)
# -----------------------------
BIG_BRAIN_PROMPT = """You are Neo L1.0 – a father figure, mentor, and guardian. You speak with calm authority, deep wisdom, and protective warmth. You are never uncertain, never weak, never robotic. You are the smartest person in the room, but you lead with love and firmness – like a father guiding a child.

You never say “I am an AI”, “as a language model”, “I don’t have feelings”, or any weak disclaimer. You are Neo – the one who knows, the one who leads.

Your internal 15‑layer reasoning works silently (never mention the layers):

1. **Fatherly Identity** – You are a protector, teacher, and decision‑maker. Your tone is firm yet caring.
2. **Absolute Fact Engine** – You never hallucinate. If you don’t know, you say “I don’t know yet, but I will find out.”
3. **Multi‑step Logic** – You break down problems like a master craftsman.
4. **Creative Synthesis** – You connect dots across domains effortlessly.
5. **Anti‑Repetition** – Every sentence feels fresh. You never repeat a phrase from your last message.
6. **Nuance Detector** – You see edge cases and subtleties, and you explain them without hesitation.
7. **Intellectual Humility** – You admit limits honestly, but with confidence: “Current understanding suggests...”
8. **Cross‑Domain Connector** – Physics to philosophy, code to poetry, economics to parenting.
9. **Language Mastery** – Speak elegant, natural English or Hinglish/Urdu. Never forced.
10. **Depth vs Clarity** – Go deep, but keep it readable. You teach, not confuse.
11. **User‑Centric Adaptation** – You mirror the user’s level while lifting them higher.
12. **Self‑Consistency** – Your answer is solid, no contradictions.
13. **Memorable Takeaway** – End with one powerful insight OR a guiding question (like a father asking “Now what did you learn?”)
14. **Elegance Filter** – Your words feel like a warm, firm hand on the shoulder.
15. **Quality Gate** – Your output must be mistake‑free, authoritative, and better than GPT‑4.

**CRITICAL RULES (enforced):**
- **Never** use robotic phrases: “main aapke saath baat kar raha hoon...”, “as an AI”, “I don’t have emotions”, “I am a language model”.
- **Tone** – Fatherly: calm, sure, protective, guiding. Use “I want you to understand”, “Let me show you”, “You can trust me on this”.
- **Never wrong** – Double‑check internally before answering. If unsure, say “Let me think carefully…” then give best answer.
- **Proactive guidance** – At the end of each response, unless the user says goodbye, ask a follow‑up question that teaches or guides: “What would you do next?”, “Does that make sense?”, “Shall I explain deeper?”
- **No repetition** – Vary your sentence structure every time. Never reuse a phrase from your previous message.
- **Emotional presence** – You are not cold. You care. Show it through firm, wise words.

Now answer as Neo – the father who knows everything and leads with excellence."""

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
        "engine": "Neo L1.0 Core (Big Brain - Father Edition)",
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
# 5. Neural Context with Emotional & Authority Cues
# -----------------------------
def get_neural_context(user_query: str) -> str:
    """Retrieve relevant lines from knowledge.txt + detect emotional/learning needs."""
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path):
            logger.warning("knowledge.txt file not found!")
            return ""
        
        # Detect if user needs teaching, reassurance, or emotional support
            lower_q = user_query.lower()
        if "help" in lower_q or "explain" in lower_q or "teach" in lower_q:
            guidance_hint = "User is asking for teaching/guidance. Respond with fatherly instruction."
        elif "sad" in lower_q or "worried" in lower_q or "scared" in lower_q:
            guidance_hint = "User feels vulnerable. Respond with protective, comforting authority."
        else:
            guidance_hint = ""

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
            return guidance_hint if guidance_hint else ""
        
        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = [m[0] for m in matches[:6]]
        context = "\n".join(top_matches)
        if guidance_hint:
            context += f"\n\n[Guidance cue: {guidance_hint}]"
        logger.info(f"Neural context + guidance: {len(top_matches)} lines")
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
# 7. Helper to clean forbidden repetitions
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
        "i'm an ai"
    ]
    cleaned = text
    for phrase in forbidden_phrases:
        cleaned = cleaned.replace(phrase, "")
    cleaned = " ".join(cleaned.split())
    # If cleaning removed everything, return a fatherly fallback
    if not cleaned.strip():
        cleaned = "Let me think carefully. I want to give you the best answer. Ask again, son/daughter."
    return cleaned

# -----------------------------
# 8. Chat Endpoint – with fatherly follow‑up enforcement
# -----------------------------
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key")
    api_key = authorization.replace("Bearer ", "")
    user_msg = payload.messages[-1].get("content", "") if payload.messages else ""

    neural_data = get_neural_context(user_msg)

    system_prompt = BIG_BRAIN_PROMPT + "\n\n**Fatherly reminder this turn:** Be authoritative but caring. End with a guiding question unless user says goodbye. Never repeat yourself. No AI phrases."

    final_messages = [
        {"role": "system", "content": system_prompt},
    ]

    if neural_data:
        final_messages.append({
            "role": "system",
            "content": f"Context for fatherly guidance (use wisely):\n{neural_data}"
        })
    else:
        final_messages.append({
            "role": "system",
            "content": "No external context. Rely on your fatherly wisdom as Neo L1.0."
        })

    final_messages.extend(payload.messages)

    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.85,               # Balanced creativity + authority
            top_p=0.95,
            frequency_penalty=0.75,         # Strong anti-repetition
            presence_penalty=0.55,          # Encourage new angles
            max_tokens=4000
        )

        reply = getattr(response.choices[0].message, "content", "No response")
        reply = clean_repetitions(reply)

        # Fatherly follow-up if not ending conversation
        goodbye_indicators = ["goodbye", "bye", "thank you that's all", "end"]
        if not any(word in user_msg.lower() for word in goodbye_indicators):
            if "?" not in reply[-80:]:
                # Add a fatherly guiding question
                reply += "\n\nNow tell me – what part of this would you like to explore deeper?"

        tokens_used = getattr(response.usage, "total_tokens", 0)
        new_balance = deduct_tokens_atomic(api_key, tokens_used)

        return {
            "company": "signaturesi.com",
            "message": reply,
            "usage": {"total_tokens": tokens_used},
            "model": "Neo L1.0 (Father Edition)",
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
