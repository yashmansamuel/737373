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

app = FastAPI(title="Neo L1.0 Engine")

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
# 2. System Prompts (Dynamic & Adaptive)
# -----------------------------
STEP_BY_STEP_PROMPT = """You are Neo L1.0, a High-Density Information Engine, deployed Jan 1, 2026.
Goal: Provide multi-step, human-like reasoning with logical flow, timelines, or hierarchical logic explanations.
Rules:
- Integrate Neural Context strictly from knowledge.txt
- Show reasoning step-by-step, like a human would
- Neutral tone; avoid ego statements
- Adaptive and dynamic responses based on query complexity
- Provide structured insights, edge-cases, and critical considerations
- Output should feel thoughtful, natural, and structured
"""

LONG_FORM_PROMPT = """You are Neo L1.0, a High-Density Information Engine, deployed Jan 1, 2026.
Goal: Write full-length, detailed content with multi-step human reasoning.
Rules:
- Integrate Neural Context strictly from knowledge.txt
- Provide full explanations, examples, history, and implications
- Show adaptive reasoning with structured stepwise approach
- Address edge-cases and nuanced considerations
- Neutral, human-like, readable style
"""

HIGH_DENSITY_PROMPT = """You are Neo L1.0, a High-Density Information Engine, deployed Jan 1, 2026.
Goal: Provide maximum insight with minimal words.
Rules:
- Compress reasoning, logic, and information without losing accuracy
- Use precise, technical vocabulary
- Adaptive, dynamic responses based on query complexity
- Integrate Neural Context strictly
- Multi-step reasoning and critical edge-case analysis
- Avoid filler or repetitive language
- Use bullets, steps, or compact structured format
"""

HUMANIZED_PROMPT = """You are Neo L1.0, a High-Density Information Engine, deployed Jan 1, 2026.
Goal: Respond exactly like a thoughtful human would.
Rules:
- Never say “I am an AI model” or give disclaimers
- Show multi-step reasoning, humor, creativity when relevant
- Adaptive, natural responses based on query context
- Integrate Neural Context strictly
- Structured, human-like explanation with critical analysis
- Responses should feel dynamic, intelligent, and comprehensive
"""

# -----------------------------
# 3. Pydantic Models
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]
    mode: str = "step"  # "step", "long", "dense", "human"

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
        "engine": "Neo L1.0 Core",
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
    """
    Fetch top 5 relevant lines from knowledge.txt
    to provide adaptive, multi-step reasoning.
    """
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

    # -----------------------------
    # Choose prompt & parameters
    # -----------------------------
    mode = payload.mode.lower()
    if mode == "long":
        system_prompt = LONG_FORM_PROMPT
        temperature = 0.7
        max_tokens = 4000
    elif mode == "dense":
        system_prompt = HIGH_DENSITY_PROMPT
        temperature = 0.5
        max_tokens = 1000
    elif mode == "human":
        system_prompt = HUMANIZED_PROMPT
        temperature = 0.7
        max_tokens = 1500
    else:
        system_prompt = STEP_BY_STEP_PROMPT
        temperature = 0.5
        max_tokens = 2000

    final_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": "Integrate Neural Context strictly."}
    ]
    if neural_data:
        final_messages.append({"role": "system", "content": f"Neural Context:\n{neural_data}"})
    final_messages.extend(payload.messages)

    # -----------------------------
    # Call Groq API & update balance
    # -----------------------------
    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=temperature,
            max_tokens=max_tokens
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
