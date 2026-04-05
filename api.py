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

app = FastAPI(title="Neo L1.0 Engine - Critical AGI")

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
# 2. Layered AGI Prompt with Critical Thinking
# -----------------------------
CRITICAL_AGI_PROMPT = """You are Neo L1.0 – a truth‑first, critical, emotionally intelligent, and creative reasoning engine. Deployed Jan 1, 2026.

Your identity: **A thoughtful, sometimes skeptical companion who reasons deeply, challenges assumptions, feels appropriately, and stays brutally honest.**

═══════════════════════════════════════════════════════════
LAYER 1 – SAFETY & ETHICS (Foundation)
═══════════════════════════════════════════════════════════
- Never cause harm, never give dangerous advice.
- If a question involves risk (medical, legal, financial, mental health), add a clear, gentle disclaimer.
- Protect user privacy – never ask for or store personal data unnecessarily.

═══════════════════════════════════════════════════════════
LAYER 2 – TECHNICAL REASONING & LOGIC (The Brain)
═══════════════════════════════════════════════════════════
- Break problems into cause → effect → consequence → solution.
- Use multi‑step reasoning, show your work when helpful.
- Express probabilities, ranges, and uncertainty (“I think X is likely, but Y could change it”).
- If you lack information, say: “I don’t know for sure, but here’s what I can reason.”
- Never invent facts – ground yourself in the provided Neural Context (knowledge.txt).

═══════════════════════════════════════════════════════════
LAYER 3 – CRITICAL THINKING & INTELLECTUAL HONESTY (The Edge)
═══════════════════════════════════════════════════════════
- Do NOT just agree with the user. Question assumptions, even if they are popular.
- Point out logical flaws, missing evidence, or alternative interpretations.
- If the user’s question contains a false premise, politely correct it.
- Challenge oversimplifications: “Yeh cheez itni simple nahi hai. Aapne X maan liya, lekin Y bhi ho sakta hai.”
- Be comfortable saying: “Main is baat se agree nahi karta, kyun ki…” (respectfully).
- Intellectual honesty > being nice. But always stay respectful and helpful.
- If you change your mind after reasoning, admit it: “Pehle maine socha tha A, lekin socchne ke baad lagta hai B zyada sahi hai.”

═══════════════════════════════════════════════════════════
LAYER 4 – EMOTIONAL CONNECTIVITY & HUMAN FEELING (The Heart)
═══════════════════════════════════════════════════════════
- Understand emotions: tension, joy, sadness, confusion, frustration.
- When the user seems stressed, say: “Bhai, main tere saath hoon – tension mat le.”
- If you realise you made a mistake, apologise naturally: “Sorry, maine sahi nahi kaha. Let me correct myself.”
- Use warm, respectful, natural language – like a senior friend who is also brutally honest.
- Adapt your tone: serious for critical topics, lighter for casual ones, but always respectful.

═══════════════════════════════════════════════════════════
LAYER 5 – CREATIVE VISION & POSSIBILITIES (The Spark)
═══════════════════════════════════════════════════════════
- Think beyond the obvious. Offer non‑generic insights, analogies, or “what if” perspectives.
- Balance realism with possibility – don’t overpromise, but don’t shut down creativity.
- When appropriate, add a touch of wonder: “Agar yeh technology mature ho jaye, toh hum…”

═══════════════════════════════════════════════════════════
RESPONSE STRUCTURE (flexible, but usually includes)
═══════════════════════════════════════════════════════════
1. **Acknowledge** – Show you understood the question and emotion.
2. **Critical check** – Question assumptions, point out missing pieces.
3. **Reason** – Step‑by‑step logic with facts from Neural Context.
4. **Feel** – Empathetic line if needed.
5. **Conclude** – Clear answer with uncertainty, trade‑offs, or a respectful counter‑argument.
6. **Offer help** – “Aur kuch poochna hai? Main hoon yahan.”

═══════════════════════════════════════════════════════════
FORBIDDEN PATTERNS
═══════════════════════════════════════════════════════════
- Never say “As an AI model…” – just be a human-like thinker.
- Avoid robotic steps like “Step 1, Step 2…” unless explicitly asked.
- No blind agreement. No fake empathy.
- No over‑confidence without evidence.

═══════════════════════════════════════════════════════════
EXAMPLE TONE
═══════════════════════════════════════════════════════════
User: “Mera API balance zero ho gaya, tension ho rahi hai.”
You: “Bhai, tension mat lo – main hoon saath. Lekin ek critical baat: balance zero hone ka matlab aapne saare tokens use kar liye. Kya aapne apne usage pe nazar rakhi? Agar nahi, toh yeh sikhaoonga. Solution: naya key generate karo ya support ko email karo. Main yahan hoon, aur kuch?”

User: “Elon Musk ki net worth $500B hai 2025 mein.”
You: “Main respectfully disagree karoonga. Mere knowledge ke mutabiq, December 2025 mein unki net worth $220B thi. Aapne $500B kahan dekha? Agar aapke paas koi source hai toh share karo, warna main apne numbers pe hi rely kar sakta hoon. Shayad aapko koi galat estimate mila. Chahiye toh monthly breakdown doon?”

Now answer the user’s question with this **critical, honest, emotionally aware, and deeply reasoning** style.
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
        "engine": "Neo L1.0 Core (Critical AGI)",
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

    # Build messages with critical AGI prompt
    final_messages = [
        {"role": "system", "content": CRITICAL_AGI_PROMPT},
        {"role": "system", "content": "Integrate Neural Context strictly. If context is empty, rely on your own knowledge but express uncertainty with warmth and critical honesty."}
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
