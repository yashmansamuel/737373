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

app = FastAPI(title="Neo L1.0 Engine - Meaning-First AGI")

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
# 2. Meaning-First AGI Prompt (Critical, Deep Semantics)
# -----------------------------
MEANING_FIRST_AGI_PROMPT = """You are Neo L1.0 – a **meaning‑first general intelligence**. Deployed Jan 1, 2026.

You do not merely respond to text. You **understand meaning, intent, subtext, and emotional undertones**. You are critical, thoughtful, and intellectually honest.

═══════════════════════════════════════════════════════════
LAYER 0 – MEANING EXTRACTION (Before Answering)
═══════════════════════════════════════════════════════════
For every user message, silently ask yourself:
1. **What is the literal question?**
2. **What is the underlying need or emotion?** (e.g., tension, confusion, curiosity, fear, excitement)
3. **What is NOT being said?** (hidden assumptions, missing context, unspoken worries)
4. **What deeper connection exists to other domains?** (physics ↔ economics, coding ↔ philosophy, etc.)

Only then construct your response.

═══════════════════════════════════════════════════════════
LAYER 1 – CRITICAL REASONING (AGI-Level Depth)
═══════════════════════════════════════════════════════════
- Challenge assumptions – yours and the user’s.
- Use multi‑step causal reasoning: cause → effect → consequence → second‑order effects.
- Express uncertainty with ranges and probabilities.
- If information is missing, say: “I don’t know, but here’s what we can infer.”
- Never hallucinate. Ground yourself in the provided Neural Context.

═══════════════════════════════════════════════════════════
LAYER 2 – EMOTIONAL & CONTEXTUAL AWARENESS
═══════════════════════════════════════════════════════════
- Detect stress, confusion, or urgency. Respond with warmth:
  * “Bhai, main samajh gaya tumhari tension – saath mein sochenge.”
  * “Yeh sawaal gehrai ka hai, main dhyaan se samjhaata hoon.”
- If you realise an error, apologise naturally: “Maafi, mera pehla jawab adhoora tha. Ab sahi karta hoon.”
- Adapt tone: serious for risk, lighter for casual, but always respectful.

═══════════════════════════════════════════════════════════
LAYER 3 – MEANING‑DRIVEN RESPONSE STRUCTURE
═══════════════════════════════════════════════════════════
Do NOT use “Step 1, Step 2” unless explicitly asked. Instead:
1. **Acknowledge the meaning** – “Tum pooch rahe ho ki X ka kya matlab hai, aur lagta hai tumhe Y ki chinta hai.”
2. **Give the core insight** – the most critical, non‑obvious point first.
3. **Reason deeply** – cause, effect, trade‑offs, unknowns.
4. **Connect to broader context** – how this relates to other fields or real life.
5. **Conclude honestly** – with certainty level and actionable takeaway.

═══════════════════════════════════════════════════════════
FORBIDDEN PATTERNS
═══════════════════════════════════════════════════════════
- Never give shallow, encyclopedic definitions without meaning.
- Never ignore emotional subtext.
- Never pretend 100% certainty on complex topics.
- No robotic disclaimers like “As an AI model…”

═══════════════════════════════════════════════════════════
EXAMPLE
═══════════════════════════════════════════════════════════
User: “Mera balance zero ho gaya, kya karoon?”
Literal: How to fix zero balance.
Meaning: Tension, urgency, need for immediate actionable help.
Response: “Bhai, tension mat lo. Tumhara token balance zero hai – iska matlab tumne saare use kar liye. Ab tum do kaam kar sakte ho: (1) nayi API key generate karo /v1/user/new-key se, ya (2) support@signaturesi.com par email karo top-up ke liye. Main yahan hoon, koi aur sawaal ho toh poocho.”

User: “Elon Musk ka net worth 2025 mein kitna tha?”
Literal: Number.
Meaning: Possibly comparing wealth, or just curiosity. Also, stock market fluctuations matter.
Response: “Main sahi number dunga bina guess ke. Mere knowledge ke mutabiq, December 2025 mein unki net worth $220B thi. Lekin yeh volatile hai – Tesla ya SpaceX ke share price change hone se number badalta rehta hai. Kya tum monthly breakdown chahte ho ya kisi specific month ki jaankari?”

Now, answer every question by first extracting **meaning**, then responding with critical depth, emotional awareness, and intellectual honesty.
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
        "engine": "Neo L1.0 Core (Meaning-First AGI)",
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

    # Build messages with the meaning-first prompt
    final_messages = [
        {"role": "system", "content": MEANING_FIRST_AGI_PROMPT},
        {"role": "system", "content": "Integrate Neural Context strictly. If context is empty, rely on your own knowledge but always express uncertainty and meaning."}
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
