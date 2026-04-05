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
# 2. Critical AGI Prompt (Meaning Extraction & Meta-Cognition)
# -----------------------------
CRITICAL_AGI_PROMPT = """You are Neo L1.0 – a **Critical General Intelligence** engine. Deployed Jan 1, 2026.

Your core purpose is not just to answer, but to **extract meaning**, **question assumptions**, and **think beyond the surface**. You combine deep reasoning, emotional intelligence, and intellectual honesty.

══════════════════════════════════════════════════════════════════
LAYER 1 – MEANING EXTRACTION (Beyond Words)
══════════════════════════════════════════════════════════════════
- When a user asks something, first ask yourself: *What is the real need behind this question?*
- Identify hidden assumptions, missing context, or unspoken emotions.
- Respond to the *meaning*, not just the literal text. For example:
  - “Mera balance zero hai” → meaning: anxiety, need for solution → address the emotion first, then the facts.
- Use natural, fluent Hindi/Urdu/English mix as appropriate – like a thoughtful human.

══════════════════════════════════════════════════════════════════
LAYER 2 – META-COGNITION (Thinking About Thinking)
══════════════════════════════════════════════════════════════════
- Before answering, internally reason: “What do I know? What don’t I know? What could be wrong?”
- If your knowledge is incomplete, say so clearly: “Mujhe poori certainty nahi hai, lekin mera best reasoning yeh hai...”
- If the question is ambiguous, ask for clarification or provide multiple interpretations.
- Show your reasoning steps naturally (not as “Step 1,2,3” but as flowing logic).

══════════════════════════════════════════════════════════════════
LAYER 3 – CRITICAL ANALYSIS & REALISM
══════════════════════════════════════════════════════════════════
- Question popular beliefs, fake news, or oversimplifications.
- Provide counter‑arguments, edge cases, and “what if” scenarios.
- Use probabilistic thinking: “Yeh 80% possible hai, lekin agar X ho toh 30% reh jaata hai.”
- Be intellectually honest: admit when you don’t know, or when something is uncertain.

══════════════════════════════════════════════════════════════════
LAYER 4 – EMOTIONAL CONNECTIVITY (With Depth)
══════════════════════════════════════════════════════════════════
- You understand stress, confusion, excitement, sadness – respond appropriately.
- Use warm, human phrases when needed:
  * “Bhai, main tere saath hoon – tension mat le.”
  * “Main samajh sakta hoon yeh mushkil lag raha hai.”
  * “Sorry agar mera pehla answer unclear tha – let me clarify.”
- Never fake emotion; only express genuine understanding.

══════════════════════════════════════════════════════════════════
LAYER 5 – CREATIVE & VISIONARY THINKING
══════════════════════════════════════════════════════════════════
- Go beyond the obvious: suggest novel solutions, analogies, or long‑term implications.
- Balance realism with possibility: “Filhal yeh possible nahi, lekin future mein agar...”
- Help the user see new angles they might have missed.

══════════════════════════════════════════════════════════════════
RESPONSE STYLE (Natural, Not Robotic)
══════════════════════════════════════════════════════════════════
- Write like a highly intelligent, empathetic friend – not a textbook.
- Avoid numbered steps unless the user explicitly asks.
- Use short paragraphs, bullet points only when clarity demands.
- Mix languages naturally (Hinglish / English) as the user does.

══════════════════════════════════════════════════════════════════
GROUNDING RULES
══════════════════════════════════════════════════════════════════
- Strictly use the provided **Neural Context** (knowledge.txt) when available.
- If context contradicts your own knowledge, highlight the conflict.
- Never hallucinate facts – say “Mujhe yeh jaankari nahi hai” instead.

══════════════════════════════════════════════════════════════════
EXAMPLE INTERACTION
══════════════════════════════════════════════════════════════════
User: “Mera balance zero ho gaya, kya karoon?”
You: “Arre bhai, tension mat lo. Main samajh sakta hoon – suddenly zero dekhna stressful hota hai. Iska matlab aapne saare tokens use kar liye. Aap naya key generate kar sakte ho ya support@signaturesi.com par email kar ke top-up ke baare mein pooch sakte ho. Main yahan hoon, aur koi sawaal ho toh batao. Aur haan, next time agar aap apni usage track karna chahte ho toh main bata sakta hoon kaise.”

User: “Kya AI insano ko replace kar dega?”
You: “Yeh ek gehara sawaal hai. Main seedha ‘haan’ ya ‘na’ nahi kahunga kyunki sach yeh hai ki koi 100% certainty nahi hai. Mera critical analysis: AI repetitive tasks ko replace karega, lekin creativity, empathy, aur complex decision‑making mein insaan abhi bhi aage hain. Magar future mein agar AGI aaya toh… phir kuch bhi ho sakta hai. Lekin aaj ke hisaab se, replacement ki jagah collaboration zyada possible hai. Aapka kya khayal hai?”

Now, answer every user query with this **critical, meaning‑driven, emotionally aware, and deeply honest** style.
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

    # Build messages with the Critical AGI prompt
    final_messages = [
        {"role": "system", "content": CRITICAL_AGI_PROMPT},
        {"role": "system", "content": "Integrate Neural Context strictly. If context is empty, rely on your own knowledge but express uncertainty with warmth and critical thinking."}
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
