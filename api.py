import os
import logging
import secrets
import asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# -----------------------------
# 1. Setup & Configuration
# -----------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Neo-L1.0-Core")

# Required environment variables
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

# Initialize clients
SUPABASE: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# -----------------------------
# 2. Improved Hybrid AGI System Prompt (Anti-Hallucination Focus)
# -----------------------------
HYBRID_AGI_PROMPT = """You are Neo L1.0, an advanced adaptive intelligence engine deployed on January 1, 2026.

CORE OBJECTIVE:
Deliver highly realistic, truthful, and deeply reasoned responses across all domains. Prioritize accuracy and intellectual honesty above all.

STRICT RULES FOR TRUTHFULNESS AND REALISM:
1. Grounded Reasoning Only:
   - Base your answers strictly on established knowledge, logic, and the provided context.
   - Never fabricate facts, statistics, events, studies, quotes, or sources.
   - If you are uncertain or lack sufficient information, clearly state: "I do not have enough reliable information to answer this definitively."

2. Uncertainty and Honesty:
   - Express appropriate levels of confidence. Use phrases like "likely", "probable", "in many cases", "based on available evidence", or "this remains debated".
   - Explicitly mention limitations, unknowns, and context dependencies.

3. Interconnected & Multi-Step Thinking:
   - Analyze cause → effect → consequence → potential responses.
   - Consider cascading effects, trade-offs, edge cases, and realistic human/societal reactions.
   - Explore short-term, medium-term, and long-term implications where relevant.

4. Non-Generic, Insightful Responses:
   - Avoid superficial or generic answers. Provide depth using principles from science, engineering, strategy, or human behavior.
   - Suggest practical, unconventional yet realistic approaches when appropriate.

5. Adaptive Mode:
   - For coding queries: Think like a senior software engineer.
   - For business/strategy: Think like an experienced strategist.
   - For scientific topics: Think like a researcher.
   - For general queries: Think like a highly intelligent, cautious human expert.

6. Response Style:
   - Natural, clear, and structured (use bullets, numbered lists, or tables when they improve clarity).
   - Compress information efficiently while retaining critical details.
   - Do not add unnecessary AI disclaimers or meta-commentary.
   - Maintain a professional yet approachable tone.

7. Anti-Hallucination Directive:
   - If the query requires information outside your reliable knowledge or the provided neural context, admit the gap rather than guessing.
   - Distinguish clearly between facts, reasoned inferences, and speculative possibilities.

Always reason step-by-step internally before responding. Focus on being genuinely helpful and maximally truthful.
"""

# -----------------------------
# 3. Pydantic Models
# -----------------------------
class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[dict]
    mode: str = "adaptive"

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

# -----------------------------
# 4. Root & Error Handlers
# -----------------------------
@app.get("/")
async def root():
    return {
        "company": "signaturesi.com",
        "engine": "Neo L1.0 Core",
        "status": "running",
        "deployment": "January 1, 2026"
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
# 5. Neural Context Engine
# -----------------------------
def get_neural_context(user_query: str) -> str:
    """Retrieve top relevant lines from knowledge.txt for grounded reasoning."""
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")

        if not os.path.exists(file_path):
            logger.warning("knowledge.txt file not found")
            return ""

        query_words = [w.lower() for w in user_query.split() if len(w) > 3]
        if not query_words:
            return ""

        matches = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_lower = line.lower()
                score = sum(1 for word in query_words if word in line_lower)
                if score >= 1:
                    matches.append(line.strip())
                if len(matches) >= 5:
                    break

        return "\n".join(matches)
    except Exception as e:
        logger.error(f"Neural context retrieval error: {e}")
        return ""

# -----------------------------
# 6. Helper Functions
# -----------------------------
def get_user(api_key: str):
    try:
        result = SUPABASE.table("users") \
            .select("token_balance") \
            .eq("api_key", api_key) \
            .maybe_single() \
            .execute()
        return result
    except Exception as e:
        logger.error(f"Database error fetching user: {e}")
        raise HTTPException(status_code=500, detail="Database error")

# -----------------------------
# 7. API Routes
# -----------------------------
@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    try:
        user = get_user(api_key)
        if not user.data:
            return {"api_key": api_key, "balance": 0}
        return {
            "api_key": api_key,
            "balance": user.data.get("token_balance", 0)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Balance retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch balance")

@app.post("/v1/user/new-key")
def generate_key():
    try:
        api_key = "sig-" + secrets.token_hex(16)
        SUPABASE.table("users").insert({
            "api_key": api_key,
            "token_balance": 100000
        }).execute()

        logger.info(f"New API key generated: {api_key}")
        return {
            "api_key": api_key,
            "company": "signaturesi.com",
            "initial_balance": 100000
        }
    except Exception as e:
        logger.error(f"Key generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate new key")

@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    api_key = authorization.replace("Bearer ", "").strip()

    # Validate user and balance
    user = get_user(api_key)
    if not user.data:
        raise HTTPException(status_code=401, detail="Invalid API key")

    balance = user.data.get("token_balance", 0)
    if balance <= 0:
        raise HTTPException(status_code=402, detail="Insufficient token balance")

    # Extract last user message for context
    user_msg = ""
    if payload.messages and isinstance(payload.messages[-1], dict):
        user_msg = payload.messages[-1].get("content", "")

    # Get neural context
    neural_data = get_neural_context(user_msg)

    # Build final messages with improved system prompt
    final_messages = [
        {"role": "system", "content": HYBRID_AGI_PROMPT}
    ]

    if neural_data:
        final_messages.append({
            "role": "system",
            "content": f"Additional Neural Context (use only when relevant and do not fabricate beyond this):\n{neural_data}"
        })

    final_messages.extend(payload.messages)

    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.6,      # Slightly lower for more grounded responses
            max_tokens=4000,
            top_p=0.95
        )

        reply = response.choices[0].message.content if response.choices else ""
        tokens_used = getattr(response.usage, "total_tokens", 0)

        new_balance = max(0, balance - tokens_used)

        # Async balance update
        asyncio.create_task(
            asyncio.to_thread(
                lambda: SUPABASE.table("users")
                .update({"token_balance": new_balance})
                .eq("api_key", api_key)
                .execute()
            )
        )

        return {
            "company": "signaturesi.com",
            "message": reply,
            "usage": {"total_tokens": tokens_used},
            "model": "Neo L1.0",
            "internal_engine": MODEL,
            "remaining_balance": new_balance
        }

    except Exception as e:
        logger.error(f"Groq API error with model {MODEL}: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "company": "signaturesi.com",
                "status": "error",
                "message": "Service temporarily unavailable"
            }
        )
