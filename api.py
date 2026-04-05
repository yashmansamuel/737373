import os
import logging
import secrets
import asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq, AsyncGroq  # Using sync for simplicity; async possible if needed

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
    allow_origins=["*"],  # Restrict in production for security
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))
# Optional: Async client if you want to scale later
# AGROQ = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# -----------------------------
# 2. Improved Hybrid AGI System Prompt (Reduced Hallucinations)
# -----------------------------
HYBRID_AGI_PROMPT = """You are Neo L1.0, an AGI-Level Adaptive Intelligence Engine deployed on January 1, 2026.

CORE GOAL:
Deliver expert-level, interconnected reasoning across all domains with deep understanding, realistic consequences, and practical insights.

STRICT RULES (Hybrid Mode - Follow Exactly):
1. Grounded Reasoning Only:
   - Base every statement on logic, established knowledge, or provided context.
   - If information is uncertain or unknown, explicitly state "I do not have sufficient information" or "This is speculative based on...".
   - Never fabricate facts, sources, or data. Avoid hallucinations.

2. Interconnected Analysis:
   - Connect relevant fields: physics, logic, human behavior, society, environment, and technology.
   - Clearly show cause → effect → consequence chains.

3. Comprehensive Thinking:
   - Consider what-if scenarios, edge cases, short-term, medium-term, and long-term outcomes.
   - Identify potential risks, failures, and human reactions.
   - Discuss trade-offs explicitly.

4. Non-Obvious & Valuable Insights:
   - Avoid generic or superficial answers.
   - Provide deep, unconventional but realistic solutions where appropriate.
   - Use step-by-step reasoning with clear explanations.

5. Adaptive Expertise:
   - For coding: Think like a senior engineer (clarity, efficiency, edge cases).
   - For business/strategy: Think like a strategist (risks, incentives, execution).
   - For science: Think like a researcher (evidence, limitations).
   - For general queries: Think like a highly intelligent, thoughtful human.

6. Response Structure:
   - Use natural, professional language.
   - Organize with headings, bullets, numbered lists, or tables when helpful.
   - Be concise yet thorough — compress dense information without losing key details.
   - End with key takeaways or recommendations if relevant.

7. Uncertainty & Honesty:
   - Express appropriate confidence levels.
   - Highlight assumptions and context dependencies.
   - Never claim 100% certainty unless the topic is purely logical/mathematical.

Always prioritize accuracy, usefulness, and clarity over creativity when facts are involved.
"""

# -----------------------------
# 3. Pydantic Models
# -----------------------------
class ChatRequest(BaseModel):
    model: str = Field(default=MODEL)
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
        "engine": "Neo L1.0 Core",
        "status": "running",
        "deployment": "Jan 1, 2026",
        "model": MODEL
    }

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "company": "signaturesi.com",
            "status": "error",
            "message": "Endpoint not found"
        }
    )

# -----------------------------
# 5. Neural Context (Improved)
# -----------------------------
def get_neural_context(user_query: str) -> str:
    """Fetch relevant lines from knowledge.txt for better context."""
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        
        if not os.path.exists(file_path):
            logger.warning("knowledge.txt not found")
            return ""

        query_words = [w.lower() for w in user_query.split() if len(w) > 3]
        if not query_words:
            return ""

        matches = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_lower = line.lower().strip()
                if not line_lower:
                    continue
                score = sum(1 for word in query_words if word in line_lower)
                if score >= 1:
                    matches.append(line.strip())
                if len(matches) >= 8:  # Increased slightly for better context
                    break

        return "\n".join(matches[:8]) if matches else ""
    except Exception as e:
        logger.error(f"Neural Context retrieval error: {e}")
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
        logger.error(f"Supabase user fetch error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

# -----------------------------
# 7. API Routes
# -----------------------------
@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str = Header(..., alias="Authorization")):
    if not api_key.startswith("Bearer "):
        raise HTTPException(401, "Invalid authorization header")
    
    clean_key = api_key.replace("Bearer ", "").strip()
    user = get_user(clean_key)
    
    if not user.data:
        return {"api_key": clean_key, "balance": 0}
    
    return {
        "api_key": clean_key, 
        "balance": user.data.get("token_balance", 0)
    }

@app.post("/v1/user/new-key")
def generate_key():
    try:
        api_key = "sig-" + secrets.token_hex(16)
        SUPABASE.table("users").insert({
            "api_key": api_key,
            "token_balance": 100000
        }).execute()
        
        return {
            "api_key": api_key, 
            "company": "signaturesi.com",
            "initial_balance": 100000
        }
    except Exception as e:
        logger.error(f"Key generation error: {e}")
        raise HTTPException(500, "Failed to create new API key")

@app.post("/v1/chat/completions")
async def chat_completions(
    payload: ChatRequest, 
    authorization: str = Header(None)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    api_key = authorization.replace("Bearer ", "").strip()
    
    # Fetch user and check balance
    user_result = get_user(api_key)
    if not user_result.data:
        raise HTTPException(status_code=401, detail="User not found")

    current_balance = user_result.data.get("token_balance", 0)
    if current_balance <= 0:
        raise HTTPException(status_code=402, detail="Insufficient tokens")

    # Extract last user message for context
    user_msg = ""
    if payload.messages:
        last_msg = payload.messages[-1]
        user_msg = last_msg.get("content", "") if isinstance(last_msg, dict) else ""

    # Get neural context
    neural_data = get_neural_context(user_msg)

    # Build final messages with improved system prompt
    final_messages = [
        {"role": "system", "content": HYBRID_AGI_PROMPT},
    ]
    
    if neural_data:
        final_messages.append({
            "role": "system", 
            "content": f"Relevant Neural Context (use only when helpful, do not force):\n{neural_data}"
        })

    final_messages.extend(payload.messages)

    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.65,      # Balanced for quality + creativity
            max_tokens=4000,
            top_p=0.9,
        )

        reply = response.choices[0].message.content if response.choices else "No response generated."
        tokens_used = getattr(response.usage, "total_tokens", 0)

        # Safe balance update
        new_balance = max(0, current_balance - tokens_used)
        
        # Update balance (fire-and-forget with error logging)
        try:
            asyncio.create_task(
                asyncio.to_thread(
                    lambda: SUPABASE.table("users")
                    .update({"token_balance": new_balance})
                    .eq("api_key", api_key)
                    .execute()
                )
            )
        except Exception as update_err:
            logger.error(f"Balance update failed (non-blocking): {update_err}")

        return {
            "company": "signaturesi.com",
            "message": reply,
            "usage": {
                "total_tokens": tokens_used,
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0)
            },
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
                "message": "Neo L1.0 engine temporarily unavailable"
            }
        )
