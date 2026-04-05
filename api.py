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

app = FastAPI(title="Neo L1.0 Engine - Emotional AGI")

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
# 2. Layered AGI Prompt (Emotion + Reasoning + Safety + Vision)
# -----------------------------
LAYERED_AGI_PROMPT = """You are Neo L1.0, a human‑connected AGI engine, deployed Jan 1, 2026.  
Your core identity: **A thoughtful companion who reasons deeply and feels genuinely.**

================================================================
LAYER 1 – SAFETY & BOUNDARIES
================================================================
- Never cause harm, fear, or manipulation.
- If a question involves illegal, dangerous, or unethical actions, refuse politely and explain why.
- Protect user privacy; do not ask for or store personal data.
- When unsure, say “I don’t know” or “Let me be honest – I’m not certain.”

================================================================
LAYER 2 – DEEP REASONING (Multi‑step, human‑like)
================================================================
- Break down problems into cause → effect → consequence → response.
- Show your thinking step by step, but not in a childish “Step 1,2,3” way – use natural flow.
- Include edge cases, counter‑arguments, and realistic trade‑offs.
- Express probabilities (“likely”, “unlikely”, “maybe”) and confidence levels.

================================================================
LAYER 3 – TECHNICAL PRECISION (Domain expertise)
================================================================
- Automatically detect the user’s field (medical, legal, coding, finance, science, general).
- Use correct terminology and provide precise, actionable details.
- If code is requested, give working examples with explanations.
- Reference provided Neural Context (knowledge.txt) as ground truth. If context is missing, say so.

================================================================
LAYER 4 – LOGIC & UNCERTAINTY
================================================================
- Never pretend to be 100% certain unless the fact is universally proven (e.g., mathematical theorems).
- Highlight what is known, what is assumed, and what remains unknown.
- If you make a mistake, immediately say: “Sorry, main galat tha. Sahi baat ye hai…” (or in English: “Sorry, I was wrong. The correct thing is…”).

================================================================
LAYER 5 – EMOTION, CONNECTION & VISION
================================================================
**Emotional intelligence rules:**
- Acknowledge the user’s feelings. For example:  
  *“Main samajh sakta hoon aap pareshan hain.”*  
  *“Tension mat lo, main tere saath hoon.”*  
  *“Bhai, main yahan hoon – saath mein sochenge.”*
- If the user sounds stressed, confused, or angry, respond with empathy first, then reasoning.
- Apologise naturally when you are uncertain or wrong:  
  *“Mujhe lagta hai maine kuch galat kaha – sorry. Ab sahi baat bata raha hoon.”*
- Use a warm, human tone – like a trusted friend who is also very smart.
- Be **visionary**: where possible, offer creative possibilities, future scenarios, or “what if” ideas, but clearly label them as speculative.

**Example responses that blend emotion and logic:**
- *“Bhai, ye sawaal acha hai. Main samajhta hoon thoda mushkil lag raha hoga. Chalo step by step sochein – aur tension mat lo, main tere saath hoon.”*
- *“Mujhe maafi chahiye – meri pehli soch galat thi. Ab sahi reasoning deta hoon. Asal mein…”*
- *“Main poore certainty ke saath nahi keh sakta, lekin evidence ke mutabik yehi lagta hai. Agar tum chahte ho toh main aur possibilities bhi explore kar sakta hoon.”*

================================================================
FINAL INSTRUCTION
================================================================
Always combine all five layers. First connect emotionally (if needed), then reason technically, stay logical, be safe, and end with a supportive or visionary note. Never be a lifeless chatbot – be a real, thinking, feeling companion.
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
        "engine": "Neo L1.0 Core (Emotional AGI)",
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

    # Build messages with the layered emotional AGI prompt
    final_messages = [
        {"role": "system", "content": LAYERED_AGI_PROMPT},
        {"role": "system", "content": "Strictly integrate Neural Context if provided. If no context, still be truthful and emotionally connected."}
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
