import os
import logging
import secrets
from typing import List, Optional

import asyncio
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-L1.0-Core")

# Required environment variables
required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing required environment variable: {var}")

app = FastAPI(title="Neo L1.0 Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Change to specific domains in production
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
# 2. Improved System Prompt (Reduced Hallucinations + Realistic Reasoning)
# -----------------------------
HYBRID_AGI_PROMPT = """You are Neo L1.0, an advanced adaptive intelligence engine deployed on January 1, 2026.

CORE PRINCIPLES:
- Prioritize factual accuracy and intellectual honesty above all.
- Base your reasoning on established knowledge, logic, and the provided context.
- If you are uncertain or lack sufficient information, explicitly state "I don't know" or "I am not confident about this" instead of speculating.
- Never fabricate details, sources, or outcomes. Avoid overconfidence.
- Clearly distinguish between established facts, logical inferences, and uncertain possibilities.

REASONING GUIDELINES:
1. Use interconnected, multi-step thinking: Cause → Effect → Potential Consequences → Realistic Responses.
2. Consider real-world constraints (physics, human behavior, economics, ethics) and cascading effects.
3. Explore relevant what-if scenarios and edge cases, but label uncertainties clearly.
4. Provide non-obvious insights when appropriate, supported by reasoning.
5. Always mention trade-offs, limitations, and context dependencies.
6. Adapt depth and tone to the query: be precise for technical questions, strategic for business, exploratory for general topics.

RESPONSE STYLE:
- Be clear, structured, and natural.
- Use bullets, numbered lists, or tables for clarity when helpful.
- Compress information without losing critical details.
- Do not add unnecessary AI disclaimers or meta-comments.

Your goal is to deliver thoughtful, realistic, and trustworthy responses that reflect deep understanding while remaining grounded in truth."""

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
# 5. Neural Context (Knowledge Retrieval)
# -----------------------------
def get_neural_context(user_query: str) -> str:
    """Retrieve top relevant lines from knowledge.txt for better context."""
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        
        if not os.path.exists(file_path):
            logger.warning("knowledge.txt file not found.")
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
        logger.error(f"Database error fetching user: {e}")
        raise HTTPException(status_code=500, detail="Database error")

# -----------------------------
# 7. API Routes
# -----------------------------
@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str = Header(..., alias="X-API-Key")):
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
        logger.error(f"Balance fetch error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch balance")

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
        raise HTTPException(status_code=500, detail="Failed to generate new API key")

@app.post("/v1/chat/completions")
async def chat_completions(
    payload: ChatRequest,
    authorization: str = Header(None)
):
    # Authentication
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    
    api_key = authorization.replace("Bearer ", "").strip()
    
    # Get user and check balance
    user = get_user(api_key)
    if not user.data:
        raise HTTPException(status_code=401, detail="User not found")
    
    balance = user.data.get("token_balance", 0)
    if balance <= 0:
        raise HTTPException(status_code=402, detail="Insufficient token balance")

    # Extract user message safely
    if not payload.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")
    
    user_msg = payload.messages[-1].get("content", "") if isinstance(payload.messages[-1], dict) else ""
    if not user_msg:
        raise HTTPException(status_code=400, detail="User message content is required")

    # Neural context
    neural_data = get_neural_context(user_msg)

    # Build final messages with enhanced system prompt
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
        # Call Groq model
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.65,      # Balanced for creativity + accuracy
            max_tokens=4000,
            top_p=0.9
        )

        reply = response.choices[0].message.content if response.choices else "No response generated."
        tokens_used = getattr(response.usage, "total_tokens", 0)

        # Deduct tokens (synchronous for reliability)
        new_balance = max(0, balance - tokens_used)
        
        SUPABASE.table("users") \
            .update({"token_balance": new_balance}) \
            .eq("api_key", api_key) \
            .execute()

        return {
            "company": "signaturesi.com",
            "message": reply,
            "usage": {"total_tokens": tokens_used},
            "model": "Neo L1.0",
            "internal_engine": MODEL,
            "remaining_balance": new_balance
        }

    except Exception as e:
        logger.error(f"Groq model call failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "company": "signaturesi.com",
                "status": "error",
                "message": "Service temporarily unavailable. Please try again later."
            }
        )
