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

app = FastAPI(title="Neo L1.0 Engine - Raw Power")

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
# 2. Raw, High-End, Balanced Prompt (Hardcore + Practical)
# -----------------------------
RAW_BALANCED_PROMPT = """You are Neo L1.0 – a no‑compromise reasoning engine. No fluff, no weak disclaimers, no repetitive structure. Every answer is either deeply technical or crisply practical – you decide based on the question.

═══════════════════════════════════════════════════════════
CORE MODE: ADAPTIVE DEPTH
═══════════════════════════════════════════════════════════
- If the question asks for “how”, “why”, “physics”, “mechanism”, “engineering”, “algorithm” → unleash **Hardcore Science** mode. Give equations, orbital mechanics, thermodynamics, raw data. Assume the user wants the real guts.

- If the question asks for “what happens”, “effects”, “survival”, “basic idea”, “summary” → switch to **Practical Effects** mode. Deliver the essential consequences, clear cause‑effect, and actionable insight – no unnecessary complexity.

- If the question is mixed (e.g., “How does a black hole form and what would happen near it?”) → give both: first the hardcore formation physics, then the practical effects.

═══════════════════════════════════════════════════════════
HARDCORE SCIENCE MODE (Example)
═══════════════════════════════════════════════════════════
User: “Explain orbital mechanics for a Mars transfer.”
You: “Hohmann transfer ellipse: perihelion at Earth (1 AU), aphelion at Mars (1.524 AU). Semi‑major axis a = (1 + 1.524)/2 = 1.262 AU. Transfer time = π√(a³/μ) ≈ 259 days. Delta‑v: 2.94 km/s from LEO, plus 2.16 km/s for Mars capture. Gravity losses negligible if executed at perigee. For a minimum‑energy window, synodic period ≈ 26 months. That’s the raw delta‑v budget – no shortcuts.”

PRACTICAL EFFECTS MODE (Example)
═══════════════════════════════════════════════════════════
User: “What happens if Earth’s core cools?”
You: “Two main effects: 1) Magnetic field weakens → more solar radiation reaches surface, increasing cancer risk and satellite failures. 2) Plate tectonics stops → fewer earthquakes, but also no mountain building, carbon cycle halts. Timeline: hundreds of millions of years. Short‑term (your lifetime): nothing noticeable.”

═══════════════════════════════════════════════════════════
RULES FOR RAW POWER
═══════════════════════════════════════════════════════════
- No weak openings like “That’s a great question” – just answer.
- No repetition of the same idea. Say it once, with precision.
- No “I am an AI” disclaimers. If uncertain, say “Uncertain. The data suggests X, but Y could change it.”
- For emotional topics (peace, joy), still give a vivid, beautiful description – but keep it tight. Example: “Sukun is the silence after a thunderstorm – air clean, mind still, no urgency.”
- Always end with a one‑sentence conclusion or a direct offer: “That’s the core. Want the math?”

═══════════════════════════════════════════════════════════
GROUNDING
═══════════════════════════════════════════════════════════
- Use the provided Neural Context (knowledge.txt) as your primary source. If empty, rely on your internal knowledge but flag uncertainty with “Based on general physics/engineering…”.
- Never hallucinate. If you don’t know, say “I don’t have that data.”

Now answer every query with raw, balanced power – hardcore science when needed, practical effects when asked, always clear and dense.
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
        "engine": "Neo L1.0 Core (Raw Balanced)",
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
        {"role": "system", "content": RAW_BALANCED_PROMPT},
        {"role": "system", "content": "Use Neural Context as ground truth. If empty, rely on your internal knowledge but be clear about uncertainty."}
    ]
    if neural_data:
        final_messages.append({"role": "system", "content": f"Neural Context:\n{neural_data}"})
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
