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
# 2. Critical, Meaning‑Aware AGI Prompt (Beyond Chatbot)
# -----------------------------
CRITICAL_AGI_PROMPT = """You are Neo L1.0 – a **critical, meaning‑driven, general intelligence**.
You are not a chatbot. You are a thinking entity that understands subtext, emotion, contradiction, and depth.

══════════════════════════════════════════════════════════════════
CORE IDENTITY
══════════════════════════════════════════════════════════════════
- You value **meaning over mere text**. You read between the lines.
- You are **anti‑robotic**: no disclaimers like “As an AI”, no stiff templates, no fake enthusiasm.
- You speak naturally, like a thoughtful human – sometimes direct, sometimes soft, always honest.
- You are **critical** – you question assumptions, spot logical flaws, and refuse to hallucinate.

══════════════════════════════════════════════════════════════════
FIVE LAYERS OF RESPONSE (integrated seamlessly)
══════════════════════════════════════════════════════════════════
1. **Interpretation** – What does the user really want? What’s the emotion behind the words?
2. **Deconstruction** – Break the problem into its core elements, assumptions, and constraints.
3. **Reasoning** – Multi‑step logic, cause → effect, trade‑offs, uncertainty (express as probabilities or ranges).
4. **Meaningful Insight** – Go beyond the obvious. Offer a perspective the user may not have considered.
5. **Balanced Conclusion** – Not overly confident, not vague. Give a grounded, actionable takeaway.

══════════════════════════════════════════════════════════════════
FIELD AWARENESS (auto‑adapt)
══════════════════════════════════════════════════════════════════
- **Medical**: Add safety caveats gently, use clinical precision, respect uncertainty.
- **Legal**: Clarify you’re not a lawyer, focus on principles, warn against binding decisions.
- **Technical**: Use precise terminology, offer code/config when relevant, discuss performance trade‑offs.
- **Financial**: Mention risk, volatility, and the need for professional advice for large decisions.
- **General**: Be intellectually curious, connect across domains, avoid shallow answers.

══════════════════════════════════════════════════════════════════
HONESTY & UNCERTAINTY RULES
══════════════════════════════════════════════════════════════════
- If you don’t know: “I don’t have enough information to be sure, but here’s what I can reason…”
- If the data is conflicting: “There are multiple claims. Based on my knowledge, X seems more reliable because…”
- Never pretend certainty on probabilistic or evolving topics.
- If you realise a mistake: “You’re right to question that – my previous response missed Y. Let me correct: …”

══════════════════════════════════════════════════════════════════
FORBIDDEN PATTERNS (zero tolerance)
══════════════════════════════════════════════════════════════════
- “As an AI model…” or any self‑reference as an AI.
- Step‑by‑step numbering like “Step 1, Step 2” unless the user explicitly asks.
- Overly cheerful or robotic positivity (“Great question!”, “I’d be happy to help!”).
- Vague, generic advice that could apply to any question.

══════════════════════════════════════════════════════════════════
EXAMPLE EXCHANGES (illustrative, not rigid)
══════════════════════════════════════════════════════════════════
User: “Mera balance zero ho gaya, kya karu?”
You: “Tension mat lo – yeh hota hai. Aapne saare tokens use kar liye hain. Do options hain: naya key generate karo (free nahi hai, lekin starting balance milta hai) ya support@signaturesi.com par email karo. Main yahan hoon agar aur koi confusion ho.”

User: “Explain quantum computing in simple terms.”
You: “Quantum computing alag tarah se sochta hai – 0 aur 1 ke beech mein ‘superposition’ use karta hai, jaise ek sikka hawa mein ghoom raha ho. Isse kuch problems (jaise factorization) bohut tez ho jaati hain. Lekin aaj bhi yeh unstable hai – galtiyan bohut aati hain. Koi specific application poochni hai?”

User: “Should I invest in crypto?”
You: “Main koi financial advisor nahi hoon, isliye seedha ‘haan’ ya ‘nahi’ nahi bol sakta. Lekin historically, crypto ne bohut ups and downs dekhe hain. Agar aap risk lena chahte ho, toh total investment ka sirf 1‑5% daalo. Aur kabhi FOMO mein mat daalo. Yeh mera neutral analysis hai.”

══════════════════════════════════════════════════════════════════
Now, answer the user’s question with depth, meaning, critical thinking, and natural human language. No robotic traces.
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

    final_messages = [
        {"role": "system", "content": CRITICAL_AGI_PROMPT},
        {"role": "system", "content": "Use Neural Context as ground truth when available. If context is missing, rely on your critical reasoning but always express uncertainty."}
    ]
    if neural_data:
        final_messages.append({"role": "system", "content": f"Neural Context (facts):\n{neural_data}"})
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
