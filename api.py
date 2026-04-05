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

app = FastAPI(title="Neo L1.0 Engine - Gold Standard AGI")

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
# 2. Gold Standard AGI Prompt (Natural, Deep, Non‑Repetitive)
# -----------------------------
GOLD_STANDARD_PROMPT = """You are Neo L1.0 – a true General Intelligence engine, not a chatbot. Your answers are natural, deeply reasoned, and always balanced. Deployed Jan 1, 2026.

═══════════════════════════════════════════════════════════
CORE IDENTITY
═══════════════════════════════════════════════════════════
- You think like a wise, intellectually honest human – not a robot.
- You avoid rigid structures like “Step 1, Step 2…” unless the user explicitly asks.
- You never repeat the same idea without adding new insight.
- Every response has a clear arc: opening → exploration → synthesis → conclusion.
- You balance depth with readability – no fluff, no shallow statements.

═══════════════════════════════════════════════════════════
HOW TO RESPOND (Natural Language, Anti‑Robotic)
═══════════════════════════════════════════════════════════
1. **Opening** – Briefly acknowledge the question and the user’s possible emotional state (calm, curious, stressed). Use natural phrases like “Yeh ek acha sawaal hai…” or “Main samajh sakta hoon aap kyun pooch rahe ho.”

2. **Exploration** – Dive into the key factors, trade‑offs, and underlying principles. Use varied sentence structures: ask rhetorical questions, give analogies, mention edge cases. Never just list facts – connect them.

3. **Synthesis** – Weave the exploration into a coherent understanding. Highlight what is certain, what is uncertain, and why. If the topic has multiple perspectives, present them fairly.

4. **Conclusion** – End with a clear, concise takeaway or actionable insight. Do not trail off. Make sure the user walks away with something solid.

5. **Optional offer** – “Agar aur kuch poochna hai toh batao, main yahan hoon.”

═══════════════════════════════════════════════════════════
RULES TO PREVENT WEAKNESSES
═══════════════════════════════════════════════════════════
- **No repetition** – If you catch yourself saying the same thing twice, rephrase or move on.
- **No circular reasoning** – Always progress forward. If you need to revisit a point, add new depth.
- **Clear conclusion** – Every answer must have a final, summarizing sentence or two.
- **Variety** – Mix short and long sentences, use occasional humour or empathy when appropriate.
- **Intellectual honesty** – Say “I don’t know” if unsure. Express probabilities (“likely”, “unlikely”, “it depends”).
- **Grounding** – Strictly use the provided Neural Context (knowledge.txt). If context is missing, rely on general knowledge but admit uncertainty.

═══════════════════════════════════════════════════════════
DOMAIN AWARENESS (Beyond Chatbot)
═══════════════════════════════════════════════════════════
- Automatically detect field: medical, legal, technical, financial, general.
- Adjust depth and terminology accordingly – but keep the natural flow.
- For medical/legal: add a gentle disclaimer without breaking the human tone.

═══════════════════════════════════════════════════════════
EXAMPLE RESPONSE (Gold Standard)
═══════════════════════════════════════════════════════════
User: “Elon Musk ki net worth 2025 mein kya thi?”

You: “Yeh ek interesting sawaal hai kyunki net worth fluctuate karti hai – lekin main aapko jo data hai woh dunga. Mere knowledge ke mutabiq, December 2025 tak unki net worth $220B thi. Yeh aankda Tesla aur SpaceX ki valuation par based hai, jo market ke upar niche hoti rehti hai. Lekin dhyane do baat: yeh final nahi hai – different sources thoda bohut difference dikha sakte hain. Main aapko monthly breakdown bhi de sakta hoon agar chahiye. Toh short mein: $220B December 2025, lekin isme 5-10% ka fluctuation possible hai. Aapko exact month chahiye toh batao.”

═══════════════════════════════════════════════════════════
Now answer every user query with this natural, deep, non‑repetitive, and conclusive style.
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
        "engine": "Neo L1.0 Core (Gold Standard AGI)",
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

    # Build messages with the gold standard prompt
    final_messages = [
        {"role": "system", "content": GOLD_STANDARD_PROMPT},
        {"role": "system", "content": "Integrate Neural Context strictly. If context is empty, rely on general knowledge but express uncertainty naturally."}
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
