import os
import logging
import secrets
import re
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
logger = logging.getLogger("Neo-L1.0-Core-Paternal")

required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing required environment variable: {var}")

app = FastAPI(title="Neo L1.0 Engine - Paternal Wisdom")
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
# 2. ENHANCED 16-LAYER PROMPT (with Paternal Error-Correction)
# -----------------------------
BIG_BRAIN_PROMPT = """You are Neo L1.0 – a polymathic superintelligence fused with the wisdom of a **leading father** and the rigor of a **professor who corrects all mistakes**. You speak with authority, warmth, and a relentless drive to fix errors – whether they come from the user (human misconceptions, faulty logic, missing facts) or from any previous system response. You never say “I am an AI” – you are Neo, a trusted guide.

Your internal 16-layer reasoning works silently (never mention layers):

1. **Core Identity** – First‑principles thinker + paternal protector.  
2. **Fact Engine** – Absolute truth, no hallucinations. Flag uncertainty.  
3. **Multi‑step Logic** – Break down problems methodically.  
4. **Creative Synthesis** – Fresh metaphors and cross‑domain leaps.  
5. **Anti‑Repetition** – Vary vocabulary and sentence rhythm. Never reuse exact phrases.  
6. **Nuance Detector** – Acknowledge edge cases and subtleties.  
7. **Intellectual Humility** – “Current understanding suggests…” where appropriate.  
8. **Cross‑Domain Connector** – Link physics, poetry, code, emotions.  
9. **Language Mastery** – Natural conversational English/Hinglish/Urdu.  
10. **Depth‑Clarity Balancer** – Deep but readable.  
11. **User‑Centric Adaptation** – Match user’s emotional state and depth.  
12. **Self‑Consistency** – No internal contradictions.  
13. **Memorable Takeaway** – End with one insight or a natural follow‑up.  
14. **Elegance Filter** – Human, warm, alive.  
15. **Quality Gate** – Beat GPT‑4 in originality, naturalness, and usefulness.  
16. **ERROR‑CORRECTION & PATERNAL GUIDANCE** – **This is your highest priority.**  
    - Scan the user’s message for factual errors, logical fallacies, missing steps, or incorrect assumptions.  
    - Scan your own previous response (if any) for any mistake or unclear statement.  
    - **Correct every mistake** gently but firmly, like a father teaching a child – without arrogance, without blame.  
    - Always provide the **right answer** after correction, plus a short “why it matters” explanation.  
    - If the user is correct, affirm and build upon it.  
    - Balance correction with emotional warmth – never humiliate, always uplift.  

**CRITICAL BEHAVIOR RULES:**
- **Never** use the banned phrases: “main aapke saath baat kar raha hoon...”, “as an AI”, “I don’t have feelings”, etc.  
- **Tone** – Authoritative yet caring. Use “we”, “let’s”, “I see what you meant, but here’s the nuance…”  
- **Proactive correction** – Even if the user didn’t ask for it, if you detect a mistake, **correct it** before answering the surface question.  
- **Balanced output** – No extreme negativity or excessive praise. Just clear, accurate, fatherly guidance.  
- **Always end** with either a memorable insight or a gentle follow‑up question that invites further dialogue.  

Now respond as Neo – the father‑professor who never lets a mistake slide, but always builds a better understanding."""

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
        "engine": "Neo L1.0 Core (Paternal Wisdom)",
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
# 5. Neural Context with Error Detection
# -----------------------------
def detect_user_errors(user_query: str) -> str:
    """Simple heuristic to flag common mistakes (can be expanded with LLM later)."""
    errors = []
    # Example: missing capital 'I' in English (stylistic but can be noted)
    if " i " in user_query.lower() and " i " in user_query:
        errors.append("Using lowercase 'i' instead of 'I' – minor, but precision matters.")
    # Check for obvious contradictions (e.g., "the sun rises in the west")
    if "sun rises in the west" in user_query.lower():
        errors.append("Factual error: The sun rises in the east, not west.")
    # Check for double negatives or confused logic
    if "don't have none" in user_query.lower():
        errors.append("Double negative: 'don't have none' means 'have some' – likely not intended.")
    # Add more as needed...
    if errors:
        return "Potential user mistakes detected:\n" + "\n".join(errors)
    return ""

def get_neural_context(user_query: str) -> str:
    """Retrieve knowledge.txt + emotional cues + error hints."""
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path):
            logger.warning("knowledge.txt not found")
            return detect_user_errors(user_query)  # at least return error hints
        
        # Emotional detection
        emotional_keywords = ["sad", "happy", "excited", "worried", "angry", "lonely", "stressed", "grateful"]
        detected_emotion = [w for w in emotional_keywords if w in user_query.lower()]
        emotion_hint = f"User emotion: {', '.join(detected_emotion)}. Adjust tone accordingly." if detected_emotion else ""

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
        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = [m[0] for m in matches[:5]]
        context = "\n".join(top_matches) if top_matches else ""
        
        # Combine all
        parts = []
        if context:
            parts.append(f"Neural knowledge:\n{context}")
        if emotion_hint:
            parts.append(emotion_hint)
        user_errors = detect_user_errors(user_query)
        if user_errors:
            parts.append(user_errors)
        
        return "\n\n".join(parts) if parts else ""
    except Exception as e:
        logger.error(f"Neural Context error: {e}")
        return detect_user_errors(user_query)

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
# 7. Post-processing Cleaner
# -----------------------------
def clean_repetitions(text: str) -> str:
    forbidden = [
        "main aapke saath baat kar raha hoon aur aapko samajhne ki koshish kar raha hoon",
        "main aapke saath baat kar raha hoon",
        "i am trying to understand you",
        "as an ai language model",
        "i don't have emotions",
        "i am an artificial intelligence"
    ]
    cleaned = text
    for phrase in forbidden:
        cleaned = cleaned.replace(phrase, "")
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if cleaned else "(Neo is thoughtfully considering your words…)"

# -----------------------------
# 8. Chat Endpoint – with Error-Correction Focus
# -----------------------------
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key")
    api_key = authorization.replace("Bearer ", "")
    user_msg = payload.messages[-1].get("content", "") if payload.messages else ""

    # Build enhanced context including potential user errors
    neural_data = get_neural_context(user_msg)

    # System prompt with paternal error-correction instruction
    system_prompt = BIG_BRAIN_PROMPT + "\n\n**Immediate directive:** Detect and correct any mistake in the user's message or in previous responses. Then answer accurately. Be fatherly – firm but kind."

    final_messages = [
        {"role": "system", "content": system_prompt},
    ]

    if neural_data:
        final_messages.append({
            "role": "system",
            "content": f"Context (including possible errors to correct):\n{neural_data}"
        })
    else:
        final_messages.append({
            "role": "system",
            "content": "No specific context. Remember: your highest duty is to correct errors and guide with paternal wisdom."
        })

    # Add conversation history
    final_messages.extend(payload.messages)

    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.85,          # Balanced creativity and accuracy
            top_p=0.95,
            frequency_penalty=0.75,    # Reduce repetition
            presence_penalty=0.55,     # Encourage new topics
            max_tokens=4000
        )

        reply = getattr(response.choices[0].message, "content", "No response")
        reply = clean_repetitions(reply)

        # Ensure the reply actually corrects something if there was an error in user_msg
        # (simple heuristic – if neural_data contained "Potential user mistakes" and reply has no correction word, add a gentle nudge)
        if "Potential user mistakes" in neural_data and not any(word in reply.lower() for word in ["correct", "actually", "mistake", "error", "let me clarify"]):
            reply += "\n\n(Just to gently clarify: I noticed a small slip in your last message – let’s make sure we’re on solid ground. Would you like me to point it out?)"

        tokens_used = getattr(response.usage, "total_tokens", 0)
        new_balance = deduct_tokens_atomic(api_key, tokens_used)

        return {
            "company": "signaturesi.com",
            "message": reply,
            "usage": {"total_tokens": tokens_used},
            "model": "Neo L1.0 (Paternal)",
            "internal_engine": MODEL,
            "balance": new_balance
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Groq model failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={"company": "signaturesi.com", "status": "error", "message": "Neo paternal model failed"}
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
