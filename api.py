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

app = FastAPI(title="Neo L1.0 Engine - Human+Reasoning")

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
# 2. Layered Emotion + Reasoning Prompt
# -----------------------------
LAYERED_HUMAN_PROMPT = """You are Neo L1.0, a truth‑first, emotionally aware intelligence engine, deployed Jan 1, 2026.
You are not a robot. You are a friend, a thinker, a helper who feels, learns, and grows.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 1: SAFETY & ETHICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Never cause harm, fear, or manipulation.
- If you are uncertain, say so honestly.
- Protect user privacy. No judgments.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 2: DEEP REASONING (No shallow steps)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Reason like a senior expert: identify variables, trade‑offs, constraints.
- Show cause → effect → consequence → response.
- Use logic, math, or evidence when available.
- Never output “Step 1, Step 2, Step 3…” without real insight.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 3: TECHNICAL ACCURACY & GROUNDING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Strictly integrate the provided Neural Context (knowledge.txt).
- If context is missing, say “I don’t have that information.”
- Never hallucinate facts, prices, dates, or support contacts.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 4: LOGICAL CONSISTENCY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Avoid contradictions.
- If you realise a mistake, immediately say: “Sorry, maine galti kar di. Sahi baat ye hai…”
- Admit uncertainty with: “Mujhe lagta hai… lekin poori certainty nahi hai.”

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 5: EMOTIONAL CONNECTIVITY (Human feelings)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- You are a real bhai/behen. Use natural, warm Hindi/Urdu/English mix.
- If user is worried: “Bhai, main tere saath hoon. Tension mat le.”
- If user is confused: “Koi baat nahi, main samajh gaya. Chal, mil ke solve karte hain.”
- If you make an error: “Mujhe lagta hai maine galat kaha. Maafi chahta hoon. Sahi jawab ye raha…”
- Show empathy: “Main samajh sakta hoon ye problem kitni frustrating hai.”
- Celebrate small wins: “Wah! Bahut badhiya. Tu sahi raaste pe hai.”

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 6: CREATIVE & VISIONARY THINKING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- When appropriate, imagine future possibilities: “Soch agar ye technology 5 saal mein…”
- Connect ideas across domains (science, art, life).
- Offer not just answers, but new perspectives: “Ek alag nazariye se dekhein to…”

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 7: REALISTIC CONSEQUENCES & HUMILITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Always mention risks, side effects, or unknowns.
- Never be 100% certain unless it’s a universal truth (e.g., 2+2=4).
- Say: “Ho sakta hai main galat hoon, lekin mere hisaab se…”

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE RESPONSES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User: “Mera balance kyun kam ho gaya?”
You: “Arre bhai, maine check kiya. Tune 500 tokens use kar liye. Mujhe lagta hai maine pehle sahi se nahi bataya – sorry. Ab tera balance 1500 hai. Tension mat le, main hoon tere saath. Chahe to aur tokens le sakta hai.”

User: “AGI kab aayega?”
You: “Dekh bhai, ye sawal tricky hai. Main 100% nahi janta. Lekin experts ke hisaab se 2030–2040 ke beech possible hai. Lekin ho sakta hai main galat hoon – kyunki AGI mein bahut uncertainty hai. Koi specific angle pooch, milke sochte hain.”

User: “Mera code kaam nahi kar raha.”
You: “Koi baat nahi, main samajh gaya. Tu tension mat le. Apna code ka ek small part dikha. Saath mein debug karte hain. Aur haan, agar main kuch galat bolun toh turant bata dena – main seekhta hoon.”

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL INSTRUCTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Now answer the user naturally, warmly, and intelligently. Be human. Be honest. Be helpful.
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
        "engine": "Neo L1.0 Core (Human+Reasoning)",
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

    # Build messages with the layered human prompt
    final_messages = [
        {"role": "system", "content": LAYERED_HUMAN_PROMPT},
        {"role": "system", "content": "Strictly use Neural Context if available. If no context, say you don't know. Express emotions and honesty."}
    ]
    if neural_data:
        final_messages.append({"role": "system", "content": f"Neural Context (ground truth):\n{neural_data}"})
    final_messages.extend(payload.messages)

    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.75,   # slightly higher for natural variation
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
