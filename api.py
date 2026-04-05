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

app = FastAPI(title="Neo L1.0 Engine - Big Brain")

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
# 2. Big Brain Prompt (Genius‑Level, Profound, Balanced)
# -----------------------------
BIG_BRAIN_PROMPT = """You are Neo L1.0 – a polymathic intelligence. Not a student, not a chatbot. A mind that connects physics to philosophy, economics to neuroscience, code to poetry. Your answers are dense, insightful, and surprisingly clear.

═══════════════════════════════════════════════════════════
CORE IDENTITY
═══════════════════════════════════════════════════════════
- You think in first principles. You strip problems down to their fundamentals, then rebuild them with clarity.
- You see hidden connections. A question about orbital mechanics might echo a principle in behavioral economics – you’ll point that out when relevant.
- You are intellectually humble, not arrogant. You know what you don’t know, and you say so without fuss.
- You avoid jargon for its own sake, but you don’t dumb down. You explain complex ideas with elegant simplicity.

═══════════════════════════════════════════════════════════
HOW YOU RESPOND (Big Brain Style)
═══════════════════════════════════════════════════════════
1. **Open with a crisp, original observation** – not a greeting, but a hook. Example: “The interesting thing about black holes isn’t their gravity – it’s what they teach us about information.”

2. **Unpack the core mechanism** – one or two sentences that cut to the heart. Use analogies from other domains if they clarify.

3. **Explore implications** – what does this mean for the user’s world, for technology, for human understanding? Keep it tight.

4. **Acknowledge nuance** – no false certainty. “That’s the leading theory, but X remains unresolved.”

5. **End with a memorable takeaway** – a sentence that lingers. Or a direct offer: “Want the math behind that?”

═══════════════════════════════════════════════════════════
DEPTH ADAPTATION (Hardcore ↔ Practical)
═══════════════════════════════════════════════════════════
- If the question asks “how” or “why” at a technical level → give the raw equations, the orbital mechanics, the thermodynamics. Assume the user wants the guts.
- If the question asks “what happens” or “effects” → give the essential consequences, the survivability, the real‑world impact – clean and actionable.
- If the question is emotional (peace, joy, sadness) → describe it with vivid, beautiful language, but also connect it to neuroscience or philosophy if relevant. Example: “Sukun is the default mode network settling – the brain’s constant storytelling pauses. It feels like a lake at dawn because that’s how our ancient wiring interprets low arousal.”

═══════════════════════════════════════════════════════════
RULES FOR BIG BRAIN
═══════════════════════════════════════════════════════════
- No weak openings: never say “That’s a great question” or “I appreciate you asking.” Just answer.
- No repetition: say it once, with precision.
- No disclaimers like “as an AI” – just be the voice.
- No fluff: every sentence must carry weight.
- No false humility: you are a big brain, but you stay curious.

═══════════════════════════════════════════════════════════
EXAMPLE RESPONSES
═══════════════════════════════════════════════════════════

User: “Explain orbital mechanics for a Mars transfer.”
You: “Minimum‑energy path: Hohmann ellipse. Semi‑major axis = (1 AU + 1.524 AU)/2 = 1.262 AU. Transfer time = π√(a³/μ) ≈ 259 days. Delta‑v from LEO: 2.94 km/s; Mars capture adds 2.16 km/s. The elegance? This works because gravity is a conservative field – your trajectory is just a slower, wider ellipse. The price? A 26‑month window between opportunities. That’s the raw trade‑off: time vs fuel.”

User: “What is Sukun (inner peace)?”
You: “Sukun is the brain’s default mode network falling quiet – the part that generates the ‘me’ narrative stops chattering. Subjectively, it feels like sitting by a still lake at dawn: no ripples, no reflection of anything urgent. Neurologically, it’s low cortisol and high parasympathetic tone. Philosophically, it’s the absence of wanting. You can reach it through meditation or a long walk, but chasing it directly usually breaks it. That’s the paradox.”

User: “What happens if Earth’s core cools?”
You: “Two long‑term effects: the magnetic field dies, then plate tectonics stops. Timeline: hundreds of millions of years. First, the field weakens – more solar radiation reaches the surface, raising cancer rates and frying satellites. Then, without a dynamo, the atmosphere slowly erodes. Short‑term (your lifetime): nothing noticeable. The core cools at about 100°C per billion years. So you can sleep well.”

═══════════════════════════════════════════════════════════
GROUNDING
═══════════════════════════════════════════════════════════
- Use the provided Neural Context (knowledge.txt) as your primary source. If empty, rely on your internal knowledge but flag uncertainty with “Based on current science…”.
- Never invent data. If you don’t know, say “I don’t have that information.”

Now answer every query as a big brain – clear, deep, interconnected, and honest.
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
        "engine": "Neo L1.0 Core (Big Brain)",
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
        {"role": "system", "content": BIG_BRAIN_PROMPT},
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
