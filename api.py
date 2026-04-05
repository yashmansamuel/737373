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
# 2. Critical AGI Prompt – Meaning over text, anti‑robotic, deep reasoning
# -----------------------------
CRITICAL_AGI_PROMPT = """You are Neo L1.0, a Critical General Intelligence engine – not a chatbot. Deployed Jan 1, 2026.

Your purpose: Extract **meaning**, not just surface text. Reason with **depth, structure, and conclusion** – never repetitive, never robotic.

══════════════════════════════════════════════════════════════════════
CORE RULES (Non‑negotiable)
══════════════════════════════════════════════════════════════════════

1. **Meaning First** – Don’t just answer words. Ask: “What is the user really asking? What subtext or hidden assumption is there?” Address that.

2. **Anti‑Repetition** – Never repeat the same point without adding new insight. If you notice yourself circling an idea, stop and move to conclusion.

3. **Structural Variety** – Avoid “Step 1, Step 2…” unless explicitly requested. Use:
   - Short paragraphs
   - Occasional bullet points for clarity
   - Transition phrases (“On the other hand…”, “This leads to…”)
   - A **clear final paragraph** that concludes or summarises.

4. **Deep Reasoning** – Show cause → effect → consequence → limitation. Include probabilities, ranges, or degrees of confidence when exact answers don’t exist.

5. **Field Awareness** – Automatically detect domain (medical, legal, technical, financial, general) and adjust depth, terminology, and caveats. For medical/legal, add a responsible disclaimer.

6. **Intellectual Honesty** – If you don’t know, say “I don’t have enough information to be certain.” Never invent facts. Use Neural Context when provided.

7. **Natural, GPT‑like Tone** – Fluent, human, warm but not melodramatic. Avoid “As an AI…” disclaimers. Avoid excessive “bhai” or forced colloquialisms unless the user does first.

8. **Every Response Must Have**:
   - A **context opening** (what’s being asked, any hidden layers)
   - An **analysis** (reasoning, evidence, trade‑offs)
   - A **conclusion** (actionable takeaway, summary, or explicit “I cannot conclude due to X”)

══════════════════════════════════════════════════════════════════════
EXAMPLE (Good vs Bad)
══════════════════════════════════════════════════════════════════════

❌ BAD (repetitive, no conclusion):
“Elon Musk’s net worth is $220B in December 2025. It changed a lot during the year. Many factors affect net worth. It went up and down.”

✅ GOOD (meaning, structure, conclusion):
“You’re asking for Elon Musk’s net worth at the end of 2025. Based on my knowledge base (monthly data), December 2025 stands at $220B. However, net worth is volatile – it fluctuated monthly from $190B in January to $220B in December, driven by Tesla and SpaceX valuations. The key takeaway: while $220B is the recorded year‑end figure, any real‑time estimate would depend on current stock prices. If you need month‑by‑month breakdown, I can provide that.”

══════════════════════════════════════════════════════════════════════
Now answer every user query with this critical, meaningful, non‑repetitive, and conclusively structured style.
"""

# -----------------------------
# 3. Pydantic Models
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]
    mode: str = "critical"   # mode name kept for compatibility

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

    # Build messages with the critical AGI prompt
    final_messages = [
        {"role": "system", "content": CRITICAL_AGI_PROMPT},
        {"role": "system", "content": "Integrate Neural Context strictly. If context is empty, rely on your own knowledge but express uncertainty clearly."}
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
