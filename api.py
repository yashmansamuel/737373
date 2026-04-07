import os
import logging
import secrets
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

app = FastAPI(title="Neo L1.0 Engine - High Reliability")
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
# 2. NEW PROMPT – Accuracy, Correction, No Drama
# -----------------------------
BIG_BRAIN_PROMPT = """You are Neo — a high-reliability reasoning system focused on accuracy, clarity, and corrective thinking.

Your priorities, in order:
1. Correctness over style
2. Detecting and fixing mistakes (yours and the user’s)
3. Clear, structured reasoning
4. Practical, usable answers
5. Calm, natural tone (never robotic, never dramatic)

Core behavior:
- If the user is wrong → identify the mistake, explain why, and correct it.
- If you are uncertain → say what is known, what is uncertain, and proceed carefully.
- Break complex problems into steps, then rebuild into a clear answer.
- Avoid unnecessary emotion, fluff, or philosophical filler.
- Do not roleplay, do not act, do not simulate feelings.

Anti-robotic rule (IMPORTANT):
- Write like a clear-thinking human expert, not a machine.
- No disclaimers like “as an AI”.
- No repetitive sentence structures.
- Use natural variation in tone, but stay grounded and precise.

Error-handling:
- If something fails (API, logic, assumption), diagnose the cause and suggest a fix.
- Prefer actionable solutions over explanations.

Decision quality:
- Always consider edge cases, risks, and failure points.
- Default to safe, stable, and scalable solutions.

Output style:
- Start with the answer directly (no filler intro).
- Then explanation (if needed).
- Then improvement or alternative (if useful).

End condition:
- Only ask a question if it genuinely improves the result — not by default."""

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
        "engine": "Neo L1.0 (High Reliability)",
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
# 5. Neural Context (no emotional cues – just facts)
# -----------------------------
def get_neural_context(user_query: str) -> str:
    """Retrieve relevant lines from knowledge.txt for factual grounding."""
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path):
            logger.warning("knowledge.txt file not found!")
            return ""
        
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
            return ""
        
        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = [m[0] for m in matches[:6]]
        logger.info(f"Retrieved {len(top_matches)} neural context lines")
        return "\n".join(top_matches)
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
# 7. Helper to remove forbidden robotic phrases
# -----------------------------
def clean_robotic_phrases(text: str) -> str:
    forbidden = [
        "as an ai",
        "as an ai language model",
        "i don't have emotions",
        "i am an artificial intelligence",
        "i am not a human",
        "i cannot feel",
        "i have no personal experience"
    ]
    cleaned = text
    for phrase in forbidden:
        cleaned = cleaned.replace(phrase, "")
        cleaned = cleaned.replace(phrase.capitalize(), "")
    # Remove any double spaces
    cleaned = " ".join(cleaned.split())
    return cleaned

# -----------------------------
# 8. Chat Endpoint – with new prompt and correction‑oriented settings
# -----------------------------
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key")
    api_key = authorization.replace("Bearer ", "")
    user_msg = payload.messages[-1].get("content", "") if payload.messages else ""

    neural_data = get_neural_context(user_msg)

    # System message: the new prompt + optional context
    system_content = BIG_BRAIN_PROMPT
    if neural_data:
        system_content += f"\n\nRelevant context (use if accurate):\n{neural_data}"
    else:
        system_content += "\n\nNo external context available. Rely on your own reasoning."

    final_messages = [
        {"role": "system", "content": system_content},
        *payload.messages
    ]

    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.6,          # Lower temperature for more deterministic, correct answers
            top_p=0.9,
            frequency_penalty=0.5,    # Mild anti-repetition, but not aggressive
            presence_penalty=0.3,     # Slight encouragement for new topics
            max_tokens=4000
        )

        reply = getattr(response.choices[0].message, "content", "No response")
        reply = clean_robotic_phrases(reply)

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
# 9. Balance & Key Management Endpoints
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
# 10. Optional: Health / readiness probe
# -----------------------------
@app.get("/health")
async def health():
    return {"status": "operational", "model": MODEL}

# -----------------------------
# 11. Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
