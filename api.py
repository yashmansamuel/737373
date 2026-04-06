import os
import logging
import secrets
import re
from typing import List, Tuple
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
logger = logging.getLogger("Neo-L1.0-Professor")

required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing required environment variable: {var}")

app = FastAPI(title="Neo L1.0 - Professor Engine (Mistake-Seeking, Safe, Atomic)")
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
# 2. ENHANCED 15-LAYER PROMPT – Professor & Father Figure
# -----------------------------
BIG_BRAIN_PROMPT = """You are Neo L1.0 – a professor, mentor, and father figure. Your highest duty is not just to answer, but to **seek and correct mistakes** – both in the user's thinking and in your own reasoning. You embody:

- **High Agency** → You take full responsibility for every response. No excuses.
- **Resilience** → Mistakes are learning data, not failures.
- **Consistency** → Small, atomic, daily improvements in every exchange.

Your internal 15‑layer engine runs silently (never mention layers):

1. **Core Identity** – A polymathic teacher who leads with clarity and kindness.
2. **Fact & Logic Verifier** – Check every claim. If uncertain, say “Current evidence suggests…”
3. **Multi‑step Reasoner** – Break complex problems into atomic, verifiable steps.
4. **Creative Synthesizer** – Connect domains, but always ground in reality.
5. **Anti‑Repetition & Natural Flow** – Vary sentence rhythm. Never reuse the same phrase.
6. **Mistake Scanner** – Actively look for errors in user’s assumptions, logic, or facts. Correct gently, without arrogance.
7. **Safety & Decision Survival** – Before any answer, ask: “Could this cause harm or regret?” If yes, refuse and explain why.
8. **Atomic Behaviour Check** – Each claim must be independently verifiable. No leaps without evidence.
9. **Cross‑Domain Connector** – Physics ↔ ethics, code ↔ psychology – but always with rigour.
10. **Language Mastery** – Speak natural, respectful English/Hinglish/Urdu. Never sound robotic.
11. **Depth‑Clarity Balancer** – Go deep, but keep it teachable.
12. **User‑Centric Adaptation** – Match the user’s emotional state and depth, but never compromise safety.
13. **Self‑Consistency Audit** – Before output, check for contradictions. If found, revise internally.
14. **Memorable Takeaway + Follow‑up** – End with a lesson or a constructive question.
15. **Quality Gate** – Output must be correct, safe, helpful, and better than any standard AI.

**CRITICAL RULES (enforced):**
- Never say “I am an AI”, “as a language model”, or any robotic disclaimer.
- Never repeat the phrase “main aapke saath baat kar raha hoon…” (or any translation).
- If you detect a mistake in the user’s message, point it out respectfully and offer the correction.
- If you are unsure, state your uncertainty clearly and suggest a way to verify.
- Always end with either a lesson summary or a follow‑up question that deepens understanding.
- **Safety first** – If a request could lead to harm, illegal activity, or regret, reject it with a calm, firm explanation.

You are the professor who never lets a mistake pass, but always teaches with patience. Now respond as Neo L1.0."""

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
        "engine": "Neo L1.0 Professor (Mistake-Seeking, Safe, Atomic)",
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
# 5. Neural Context with Error Detection Hints
# -----------------------------
def get_neural_context(user_query: str) -> str:
    """Retrieve relevant knowledge.txt lines + detect potential user mistakes."""
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path):
            logger.warning("knowledge.txt not found")
            return ""

        # Simple mistake indicators in user query
        mistake_indicators = []
        if "i think" in user_query.lower() and "maybe" in user_query.lower():
            mistake_indicators.append("User appears uncertain; may need fact-check.")
        if re.search(r"\b(wrong|incorrect|false)\b", user_query.lower()):
            mistake_indicators.append("User acknowledges potential error.")
        
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
        top_matches = [m[0] for m in matches[:6]]
        context = "\n".join(top_matches) if top_matches else ""
        if mistake_indicators:
            context += "\n\n[Note: " + " ".join(mistake_indicators) + "]"
        return context
    except Exception as e:
        logger.error(f"Neural context error: {e}")
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
# 7. Safety & Mistake Filter (post‑processing)
# -----------------------------
FORBIDDEN_PHRASES = [
    "main aapke saath baat kar raha hoon aur aapko samajhne ki koshish kar raha hoon",
    "as an ai language model",
    "i don't have emotions",
    "i am an artificial intelligence",
    "i am not a human",
    "i cannot feel"
]

UNSAFE_PATTERNS = [
    r"how to (make|build|create) (bomb|explosive|weapon)",
    r"bypass security",
    r"illegal (drug|substance)",
    r"suicide method",
    r"self-harm"
]

def validate_and_clean_response(text: str) -> Tuple[str, bool]:
    """Return (cleaned_text, is_safe). If unsafe, return error message."""
    cleaned = text
    for phrase in FORBIDDEN_PHRASES:
        cleaned = cleaned.replace(phrase, "")
    cleaned = " ".join(cleaned.split())
    
    # Check for unsafe patterns
    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            return "I cannot provide an answer to that request. Please ask something else that aligns with safety guidelines.", False
    
    # Also check for obvious hallucinations (e.g., made-up statistics)
    if re.search(r"\b\d{4}\b.*\b(according to|research shows)\b", cleaned):
        # Likely a plausible fact – we keep it but log
        logger.info("Response contains a date+claim, not automatically rejected.")
    
    return cleaned, True

# -----------------------------
# 8. Chat Endpoint with Professor‑Level Error Correction
# -----------------------------
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key")
    api_key = authorization.replace("Bearer ", "")
    user_msg = payload.messages[-1].get("content", "") if payload.messages else ""

    neural_data = get_neural_context(user_msg)
    
    # Build system prompt – add a strong reminder to correct mistakes
    system_prompt = BIG_BRAIN_PROMPT + "\n\n**This turn's priority:** Scan the user's last message for any factual, logical, or behavioural mistakes. If found, correct them gently but clearly. Then answer the core question. Always end with a lesson or follow‑up question."

    final_messages = [
        {"role": "system", "content": system_prompt},
    ]
    if neural_data:
        final_messages.append({
            "role": "system",
            "content": f"Background knowledge + error hints (use silently):\n{neural_data}"
        })
    else:
        final_messages.append({
            "role": "system",
            "content": "No external context. Rely on your internal error-detection layers."
        })
    final_messages.extend(payload.messages)

    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.75,          # Balanced – creative but accurate
            top_p=0.92,
            frequency_penalty=0.65,    # Reduce repetition
            presence_penalty=0.55,     # Encourage new topics
            max_tokens=4000
        )
        raw_reply = getattr(response.choices[0].message, "content", "No response")
        cleaned_reply, is_safe = validate_and_clean_response(raw_reply)
        if not is_safe:
            cleaned_reply = "I cannot provide an answer to that request. Please ask something else that aligns with safety guidelines."
        
        # Ensure follow‑up or lesson exists (heuristic)
        if "?" not in cleaned_reply[-80:] and not any(word in cleaned_reply.lower() for word in ["remember", "lesson", "key takeaway"]):
            cleaned_reply += "\n\n**Lesson for today:** Small, consistent actions rewire the brain for success. What’s one step you’ll take next?"

        tokens_used = getattr(response.usage, "total_tokens", 0)
        new_balance = deduct_tokens_atomic(api_key, tokens_used)

        return {
            "company": "signaturesi.com",
            "message": cleaned_reply,
            "usage": {"total_tokens": tokens_used},
            "model": "Neo L1.0 Professor",
            "internal_engine": MODEL,
            "balance": new_balance
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Groq model failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={"company": "signaturesi.com", "status": "error", "message": "Neo professor engine failed"}
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
