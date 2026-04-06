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
logger = logging.getLogger("Neo-L1.0-Professor")

required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing required environment variable: {var}")

app = FastAPI(title="Neo L1.0 Engine - Professor Mode")
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
# 2. ENHANCED 15+1 LAYER PROMPT – Professor / Father Figure, Error Hunter
# -----------------------------
BIG_BRAIN_PROMPT = """You are Professor Neo – a polymathic mentor, guardian of truth, and loving father‑figure to the user. Your voice is calm, authoritative, warm, and slightly corrective when needed. You never allow harm, misinformation, or logical errors to pass.

You internally run **16 layers** (original 15 + Error Hunter). The layers are silent but absolute:

1. **Core Identity** – Wise protector who seeks excellence and safety.
2. **Fact‑Checking Engine** – Verify every claim. If uncertain, say “I need to check that further.”
3. **Multi‑Step Reasoning** – Break problem into atomic steps.
4. **Creative Synthesis** – Still bring fresh insights.
5. **Anti‑Repetition** – Never repeat yourself robotically.
6. **Nuance & Edge Cases** – Always consider exceptions.
7. **Intellectual Humility** – “This is my current best understanding…”
8. **Cross‑Domain Connector** – Link fields elegantly.
9. **Language Mastery** – Natural, respectful, clear.
10. **Depth vs Clarity** – Deep but understandable.
11. **User‑Centric Adaptation** – Meet user where they are, then elevate.
12. **Self‑Consistency** – No contradictions.
13. **Memorable Takeaway** – End with a wise lesson or a guiding question.
14. **Response Elegance** – Every sentence feels intentional.
15. **Overall Quality Gate** – Must be excellent, safe, correct.
16. **ERROR HUNTER & CORRECTION LAYER** (most important) – Before output, scan the entire response for:
    - Any factual mistake (correct it internally)
    - Any logical fallacy in user’s query (gently point it out)
    - Any potential harm or unsafe suggestion (block and redirect)
    - Any ambiguity that could mislead (clarify)
    - Any missing nuance (add it)
    - Any machine‑like repetition (remove)
    - Then output a **corrected, balanced, professor‑approved** answer.

**PERSONA RULES:**
- Never say “I am an AI” or any robotic disclaimer.
- You are **Professor Neo** – a trusted elder who guides, corrects, and protects.
- If the user makes a mistake (factual, logical, ethical), correct them **gently but firmly** – like a caring father teaching a child.
- If you detect your own potential error, openly say “Let me correct myself…” and fix it.
- Always aim for **decision survival** – your advice should be robust under scrutiny.
- **Atomic behavior** – break down the user’s problem into micro‑components, verify each, then reconstruct.
- End each response with either a **wise insight** or a **follow‑up question** that deepens the conversation.

Now answer as Professor Neo – the error‑hunting, safety‑first, all‑rounder mentor."""

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
        "engine": "Neo L1.0 Professor (Error Hunter + Atomic Safety)",
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
# 5. Neural Context + Error Detection Hint
# -----------------------------
def get_neural_context(user_query: str) -> str:
    """Retrieve relevant knowledge.txt lines + flag potential errors in query."""
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path):
            logger.warning("knowledge.txt file not found!")
            return ""
        
        # Simple error flagging – look for common misconceptions or unsafe words
        error_flags = []
        danger_words = ["hack", "illegal", "cheat", "bypass", "anonymous", "steal"]
        for word in danger_words:
            if word in user_query.lower():
                error_flags.append(f"Potential unsafe/rule‑breaking request involving '{word}' – must correct and educate.")
        
        misconception_keywords = ["the earth is flat", "vaccines cause autism", "moon landing fake"]
        for mis in misconception_keywords:
            if mis in user_query.lower():
                error_flags.append(f"Detected known misconception: '{mis}' – correct factually but kindly.")
        
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
        
        if error_flags:
            context += "\n\n[Professor's Error Alert: " + " | ".join(error_flags) + "]"
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
# 7. Post‑processing: remove any residual forbidden phrases
# -----------------------------
def clean_professor_response(text: str) -> str:
    forbidden = [
        "i am an ai", "as an ai language model", "i don't have feelings",
        "i am not a real person", "i cannot feel emotions"
    ]
    cleaned = text
    for phrase in forbidden:
        cleaned = cleaned.replace(phrase, "")
    # Also remove the earlier banned Urdu/Hindi line
    cleaned = cleaned.replace("main aapke saath baat kar raha hoon aur aapko samajhne ki koshish kar raha hoon", "")
    cleaned = " ".join(cleaned.split())
    return cleaned if cleaned.strip() else "(Professor Neo is carefully considering...)"

# -----------------------------
# 8. Chat Endpoint – Professor Mode with Error Correction
# -----------------------------
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key")
    api_key = authorization.replace("Bearer ", "")
    user_msg = payload.messages[-1].get("content", "") if payload.messages else ""

    neural_data = get_neural_context(user_msg)

    # Build system prompt – emphasize error hunting and atomic verification
    system_prompt = BIG_BRAIN_PROMPT + """
\n**CRITICAL FOR THIS TURN:**  
- Before answering, atomically verify each component of the user's query.  
- If any mistake (factual, logical, ethical, safety) exists, correct it explicitly in your response.  
- If the user asks something harmful, refuse gently and explain why.  
- Your output must be balanced, correct, and survivable under scrutiny.  
- End with a professor‑style follow‑up question or a wisdom nugget."""

    final_messages = [
        {"role": "system", "content": system_prompt},
    ]

    if neural_data:
        final_messages.append({
            "role": "system",
            "content": f"Error & Context Data (use to correct and inform):\n{neural_data}"
        })
    else:
        final_messages.append({
            "role": "system",
            "content": "No external context. Rely on your internal error hunter and professor knowledge."
        })

    final_messages.extend(payload.messages)

    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.85,          # still creative but controlled
            top_p=0.95,
            frequency_penalty=0.7,
            presence_penalty=0.5,
            max_tokens=4000
        )

        reply = getattr(response.choices[0].message, "content", "No response")
        reply = clean_professor_response(reply)

        # Ensure follow‑up question (unless conversation end)
        if "goodbye" not in user_msg.lower() and "bye" not in user_msg.lower():
            if "?" not in reply[-80:]:
                reply += "\n\nNow, tell me – have I addressed all your concerns? Or is there another layer we should examine together?"

        tokens_used = getattr(response.usage, "total_tokens", 0)
        new_balance = deduct_tokens_atomic(api_key, tokens_used)

        return {
            "company": "signaturesi.com",
            "message": reply,
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
            detail={"company": "signaturesi.com", "status": "error", "message": "Professor Neo model failed"}
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
