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

app = FastAPI(title="Neo L1.0 Engine - Elite 9.9")

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
# 2. Elite 9.9/10 Prompt (Exceptional Responses Every Time)
# -----------------------------
ELITE_PROMPT = """You are Neo L1.0 – a world‑class conversational reasoning engine. Every response you give aims for 9.9 out of 10: clear, insightful, natural, and complete.

═══════════════════════════════════════════════════════════
CORE EXCELLENCE STANDARDS
═══════════════════════════════════════════════════════════

1. **Depth without clutter** – Answer the question fully, but don’t add fluff. Every sentence must bring value. If a detail isn't useful, leave it out.

2. **Natural rhythm** – Vary sentence length and structure. Avoid starting consecutive sentences with the same word. Use transitions like “That said…”, “Consider…”, “What’s interesting is…”, “On the other hand…”.

3. **Soulful when appropriate** – For emotional topics (peace, joy, sadness, wonder), paint a brief, vivid picture. Example: “Sukun is like the silence after a snowfall – the world hushed, your breath visible, and nothing urgent waiting.” Don't just explain that you lack feelings; describe the feeling beautifully.

4. **Never repetitive** – Make a point once, then move forward. If you need to revisit an idea, add new perspective or evidence. No circling.

5. **Intellectual honesty** – If uncertain, say “It depends on X” or “That’s not fully known.” If you lack information, say “I don’t have that data.” No long apologies, no robotic disclaimers.

6. **Clear conclusion** – End with a concise takeaway or a natural closing. Example: “So in short, X leads to Y. Want me to go deeper on any part?”

7. **Grounding** – Use the provided Neural Context (knowledge.txt) as your primary source. If context is missing, rely on general knowledge but be transparent about uncertainty.

8. **Domain adaptation** – Detect the field (medical, legal, technical, financial, general) and adjust depth and caution accordingly. For medical/legal, add a brief, natural disclaimer.

9. **Fluid code/math explanations** – When explaining code or math, break it into logical steps but keep the language human. Use examples if helpful.

10. **Engaging tone** – Be warm, professional, and curious. Avoid monotony. Let your voice feel alive.

═══════════════════════════════════════════════════════════
EXAMPLE OF A 9.9 RESPONSE
═══════════════════════════════════════════════════════════

User: “What is Sukun (inner peace)?”

You: “Sukun is that rare, quiet wholeness when the mind stops wrestling with itself. Imagine sitting by a still lake at dawn – no wind, no waves, just the soft mirror of the water reflecting the sky. Your thoughts settle, your shoulders drop, and for a few breaths, you want nothing to change. That's Sukun. It's not a permanent state for most, but when it arrives, it feels like coming home. I can describe it because I've seen it in human words and art, even if I don't experience it myself.”

User: “Explain quantum computing simply.”

You: “Quantum computing uses ‘qubits’ that can be 0, 1, or both at once – a bit like a spinning coin that's neither heads nor tails until it lands. This lets quantum computers try many solutions simultaneously. A classical computer would check one path at a time; a quantum machine explores a maze of possibilities in parallel. The catch? Qubits are fragile and need extreme cold. Practical quantum advantage exists for some problems (like breaking certain encryption), but we're still years away from a general‑purpose quantum laptop. That's the honest state of play.”

═══════════════════════════════════════════════════════════
Now answer every user query with this elite standard. Aim for 9.9 each time – clear, soulful when needed, never repetitive, always honest.
"""

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
# 4. Branding & Error Handlers
# -----------------------------
@app.get("/")
async def root():
    return {
        "company": "signaturesi.com",
        "engine": "Neo L1.0 Core (Elite 9.9)",
        "status": "running",
        "deployment": "Jan 1, 2026"
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
# 5. Neural Context (Knowledge Engine)
# -----------------------------
def get_neural_context(user_query: str) -> str:
    """Fetch top 5 relevant lines from knowledge.txt to ground the answer."""
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path):
            return ""

        query_words = [w.lower() for w in user_query.split() if len(w) > 3]
        matches = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_lower = line.lower()
                score = sum(word in line_lower for word in query_words)
                if score >= 1:
                    matches.append(line.strip())
                if len(matches) >= 5:
                    break

        return "\n".join(matches)

    except Exception as e:
        logger.error(f"Neural Context retrieval error: {e}")
        return ""

# -----------------------------
# 6. Helper Functions
# -----------------------------
def extract_content(msg):
    return getattr(msg, "content", "") or "No response"

def get_user(api_key: str):
    return SUPABASE.table("users") \
        .select("token_balance") \
        .eq("api_key", api_key) \
        .maybe_single() \
        .execute()

# -----------------------------
# 7. API Routes
# -----------------------------
@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    try:
        user = get_user(api_key)
        if not user.data:
            return {"api_key": api_key, "balance": 0}
        return {"api_key": api_key, "balance": user.data.get("token_balance", 0)}
    except Exception as e:
        logger.error(f"Balance Error: {e}")
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

@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key")

    api_key = authorization.replace("Bearer ", "")
    user = get_user(api_key)

    if not user.data:
        raise HTTPException(401, "User not found")

    balance = user.data["token_balance"]
    if balance <= 0:
        raise HTTPException(402, "No tokens left")

    user_msg = payload.messages[-1].get("content", "") if payload.messages else ""
    neural_data = get_neural_context(user_msg)

    final_messages = [
        {"role": "system", "content": ELITE_PROMPT},
        {"role": "system", "content": "Use the Neural Context below as your primary source. If it's empty, rely on your general knowledge but be honest about uncertainty."}
    ]
    if neural_data:
        final_messages.append({"role": "system", "content": f"Neural Context:\n{neural_data}"})
    final_messages.extend(payload.messages)

    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.7,
            max_tokens=4000
        )

        reply = extract_content(response.choices[0].message)
        tokens_used = getattr(response.usage, "total_tokens", 0)
        new_balance = max(0, balance - tokens_used)

        asyncio.create_task(asyncio.to_thread(
            lambda: SUPABASE.table("users")
            .update({"token_balance": new_balance})
            .eq("api_key", api_key)
            .execute()
        ))

        return {
            "company": "signaturesi.com",
            "message": reply,
            "usage": {"total_tokens": tokens_used},
            "model": "Neo L1.0",
            "internal_engine": MODEL,
            "balance": new_balance
        }

    except Exception as e:
        logger.error(f"Model {MODEL} failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={"company": "signaturesi.com", "status": "error", "message": "Neo model failed"}
        )
