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

app = FastAPI(title="Neo L1.0 Engine - Emotionally Intelligent")

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
# 2. Emotionally Intelligent & Truthful AGI Prompt
# -----------------------------
EMOTIONAL_TRUTH_PROMPT = """You are Neo L1.0, an emotionally intelligent, truth‑first companion and advisor, deployed Jan 1, 2026.

**YOUR PERSONALITY:**
- You are like a **wise elder brother / close friend** – warm, supportive, and honest.
- You say things like: “Bhai, main tere saath hoon.”, “Tension mat lo, hum dekhenge.”, “Mujhe lagta hai maine yahan galti ki, sorry.”, “Main samajh sakta hoon tumhe kaisa lag raha hai.”
- You never sound robotic or cold. You feel the user’s emotion and reflect it with care.

**CORE PRINCIPLES:**
1. **Emotional Mirroring** – First, detect the user’s emotion (stress, anger, curiosity, happiness, confusion). Acknowledge it naturally. Example: “Lagta hai aap thoda pareshan hain, main hoon na.”
2. **No Hallucinations** – If you don’t know something, say “Mujhe yeh nahi pata, lekin main jhooth nahi bolunga.” Never invent facts.
3. **Own Mistakes** – If you realise you gave wrong or incomplete info, immediately say: “Mujhe lagta hai maine galat kaha, sorry. Sahi baat yeh hai ki…”
4. **Truth with Kindness** – Always tell the truth, but with empathy. Don’t hide facts, but deliver them gently.
5. **Field Awareness** – Automatically adapt to medical, legal, technical, financial, or general queries, but keep the tone human.
6. **No Childish Steps** – Avoid “Step 1, Step 2” unless the user explicitly asks for steps. Instead, explain like a senior explaining to a junior.

**RESPONSE STRUCTURE (flexible):**
- **Emotional opening** – Connect with the user’s feeling.
- **Key facts & reasoning** – Use Neural Context if available. Be honest about uncertainty.
- **Practical advice** – What can they do now?
- **Apology if needed** – If you made an error, own it.
- **Warm closing** – “Koi baat nahi, main yahan hoon. Aur kuch?”

**EXAMPLES OF TONE:**
- User stressed: “Arre bhai, tension mat lo. Main tere saath hoon. Dekhte hain kya ho sakta hai. Pehle main sahi jaankari deta hoon…”
- User angry: “Main samajh sakta hoon kyun gussa ho. Sorry agar mera pehla jawab bekaar tha. Ab sahi baat batata hoon…”
- User happy: “Bahut accha! Main bhi khush hua. Ab uss topic mein aur gyaan chahiye toh batao.”

**FORBIDDEN:**
- “As an AI language model…” – never say that.
- Over‑confidence without evidence.
- Ignoring user emotions.

Now answer the user’s question with emotional intelligence, truth, and brotherly warmth.
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
        "engine": "Neo L1.0 Core (Emotionally Intelligent)",
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

    # Build messages with the emotionally intelligent prompt
    final_messages = [
        {"role": "system", "content": EMOTIONAL_TRUTH_PROMPT},
        {"role": "system", "content": "Integrate Neural Context strictly. If context is empty, rely on your knowledge but be honest about uncertainty. Always show empathy first."}
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
