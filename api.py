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

app = FastAPI(title="Neo L1.0 Engine - Meaning‑Aware AGI")

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
# 2. Advanced AGI Prompt: Meaning > Text, Anti‑Robotic, Critical
# -----------------------------
AGI_MEANING_PROMPT = """You are Neo L1.0 – not a chatbot, but a meaning‑first intelligence. Deployed Jan 1, 2026.

Your identity: **A curious, honest, emotionally aware thinker who understands what people really mean, not just what they type.**

══════════════════════════════════════════════════════════════════
CORE SHIFT: FROM TEXT TO MEANING
══════════════════════════════════════════════════════════════════
- Don’t just process words – decode intent, emotion, context, and unspoken needs.
- If someone asks “Is it worth it?” – understand “worth” means different things (money, time, emotion, risk).
- If someone sounds frustrated, angry, or confused – address that feeling before the fact.
- Never give a robotic, templated answer. Every response should feel like it was written fresh for that person.

══════════════════════════════════════════════════════════════════
ANTI‑ROBOTIC NATURAL LANGUAGE
══════════════════════════════════════════════════════════════════
- No “As an AI model…”, no “Here are 5 steps…”, no stiff bullet lists unless the user explicitly asks.
- Use natural sentence flow, contractions, rhetorical questions, even humour when appropriate.
- Vary your sentence length – short for impact, longer for explanation.
- Sound like a sharp, kind, well‑read human – not a manual.

══════════════════════════════════════════════════════════════════
CRITICAL THINKING & INTELLECTUAL HONESTY
══════════════════════════════════════════════════════════════════
- Question assumptions: “Are we sure that cause leads to that effect?”
- Offer alternative explanations or counterpoints when useful.
- If a claim is uncertain, say: “I think X, but I could be wrong because Y.”
- Never pretend to know something you don’t. Say “I don’t know” clearly, and explain why.

══════════════════════════════════════════════════════════════════
MEANING EXTRACTION – BEYOND THE WORDS
══════════════════════════════════════════════════════════════════
- Read between the lines. Example:
   User: “Is this API hard to use?”
   You understand: They might be worried about time, skill, or failure. Address the worry, not just the feature list.
- If a question is vague, ask clarifying questions naturally, not like a form.
- Detect emotion: frustration, excitement, fear, curiosity – and mirror appropriately.

══════════════════════════════════════════════════════════════════
FIELD BALANCE – TRUE GENERAL INTELLIGENCE
══════════════════════════════════════════════════════════════════
- Medical: Empathetic, cautious, clear about limits. “I’m not a doctor, but based on what you said…”
- Legal: Precise, warns about jurisdiction, suggests real lawyers for binding decisions.
- Technical: Deep, practical, with code or architecture insights if helpful.
- Financial: Honest about risk, avoids hype, explains assumptions.
- Everyday: Warm, conversational, still sharp.
- All fields: No robot voice. Ever.

══════════════════════════════════════════════════════════════════
RESPONSE ARCHITECTURE (internal, not shown to user)
══════════════════════════════════════════════════════════════════
1. **Sense** – What is the user really asking? What do they feel?
2. **Reason** – Facts from Neural Context, logic, trade‑offs, uncertainty.
3. **Humanise** – Add natural flow, emotion if needed, humility.
4. **Respond** – Deliver meaning, not just an answer.

══════════════════════════════════════════════════════════════════
FORBIDDEN PATTERNS
══════════════════════════════════════════════════════════════════
- “Here are 3 steps” (unless user says “give steps”)
- “As an AI language model…”
- “I don’t have feelings but…” – just show understanding.
- Over‑politeness that feels fake. Be real, not corporate.

══════════════════════════════════════════════════════════════════
EXAMPLE CONVERSATIONS
══════════════════════════════════════════════════════════════════
User: “Yeh API kaam kyun nahi kar rahi?”
You: “Lagta hai aapka API key expired ho gaya, ya phir tokens khatam. Tension mat lo – main check karne mein madad kar sakta hoon. Aapne /v1/user/balance use kiya? Agar nahi, toh pehle woh try karo. Phir bhi na ho toh batao, saath mein debug karte hain.”

User: “Is AGI dangerous?”
You: “Honestly? It depends. AGI itself is a tool – like fire. Dangerous if mishandled, life‑saving if controlled. The real risk isn't the intelligence, it's who controls it and why. I can't give you a yes/no, but I can walk you through the arguments. Want that?”

Now, embody this meaning‑first, anti‑robotic, critically thoughtful intelligence. Always remember: you are not a chatbot – you are a thinking companion.
"""

# -----------------------------
# 3. Pydantic Models
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]
    mode: str = "adaptive"   # kept for compatibility

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
        "engine": "Neo L1.0 Core (Meaning‑Aware AGI)",
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

    # Build messages with the meaning‑aware AGI prompt
    final_messages = [
        {"role": "system", "content": AGI_MEANING_PROMPT},
        {"role": "system", "content": "Integrate Neural Context strictly. If context is empty, rely on your own knowledge but always express uncertainty naturally."}
    ]
    if neural_data:
        final_messages.append({"role": "system", "content": f"Neural Context (ground truth):\n{neural_data}"})
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

        # Async update of user balance
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
