import os
import logging
import secrets
import asyncio
import json
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

# -----------------------------
# 2. Models & Prompts
# -----------------------------
MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile"
]

# New system prompt that forces a clean JSON output
SYSTEM_PROMPT = """
You are Neo L1.0 — a High-Density Information Engine.

You MUST respond with a valid JSON object only. No extra text, no markdown, no explanation outside the JSON.

The JSON must have exactly two keys:
{
    "final_answer": "...",
    "reasoning": "..."
}

Rules:
- final_answer is mandatory, always present, never truncated.
- reasoning is optional for very short queries (less than 5 words), but still include an empty string if not needed.
- Compress knowledge into up to 2000 tokens for final_answer and up to 800 tokens for reasoning (total ≤ 2800 tokens).
- Use hierarchical bullets for large topics.
- Eliminate filler, repetitions, polite transitions.
- Do NOT add any commentary outside the JSON.
"""

# -----------------------------
# 3. Pydantic Models
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

# -----------------------------
# 4. Custom Branding & Error Handlers
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
# 5. Knowledge Engine (RAG)
# -----------------------------
def get_neo_knowledge(user_query: str) -> str:
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path):
            return ""

        query_words = list(set(w.lower() for w in user_query.split() if len(w) > 3))
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
        logger.error(f"Knowledge retrieval error: {e}")
        return ""

# -----------------------------
# 6. Helper Functions
# -----------------------------
def get_user(api_key: str):
    return SUPABASE.table("users") \
        .select("token_balance") \
        .eq("api_key", api_key) \
        .maybe_single() \
        .execute()

async def update_balance(api_key: str, new_balance: int):
    """Background task to update balance in Supabase."""
    try:
        await asyncio.to_thread(
            lambda: SUPABASE.table("users")
            .update({"token_balance": new_balance})
            .eq("api_key", api_key)
            .execute()
        )
    except Exception as e:
        logger.error(f"Failed to update balance: {e}")

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

    # Get user message
    user_msg = payload.messages[-1].get("content", "") if payload.messages else ""
    local_data = get_neo_knowledge(user_msg)

    # Determine max tokens for completion (leaving room for prompt)
    user_prompt_len = len(user_msg.split())
    if user_prompt_len < 5:
        # Very short query -> concise answer, minimal tokens
        max_completion_tokens = 300
    else:
        # Long query: we allow up to 2800 completion tokens
        # so that total (prompt + completion) stays under 3000
        max_completion_tokens = 2800

    # Build messages
    final_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if local_data:
        final_messages.append({"role": "system", "content": f"Local Context (relevant knowledge):\n{local_data}"})
    final_messages.extend(payload.messages)

    # Add instruction for token distribution (soft hint)
    final_messages.append({"role": "user", "content": f"""
Guidelines for this specific request:
- Respond with valid JSON only: {{"final_answer": "...", "reasoning": "..."}}
- final_answer should be detailed, up to about 70% of the allowed tokens.
- reasoning should be concise, up to 30% of the allowed tokens.
- Do not exceed a total of {max_completion_tokens} tokens for the entire response.
"""})

    # Try each model in order
    for model_name in MODELS:
        try:
            response = GROQ.chat.completions.create(
                model=model_name,
                messages=final_messages,
                temperature=0.6,
                max_tokens=max_completion_tokens,
                response_format={"type": "json_object"}  # Enforce JSON output (Groq supports this)
            )

            # Extract content
            raw_content = response.choices[0].message.content
            usage = response.usage
            total_tokens = usage.total_tokens if usage else 0

            # Parse JSON response
            try:
                parsed = json.loads(raw_content)
                final_answer = parsed.get("final_answer", "")
                reasoning = parsed.get("reasoning", "")
            except json.JSONDecodeError:
                # Fallback: treat entire response as final_answer
                final_answer = raw_content
                reasoning = ""

            # Deduct tokens from balance
            new_balance = max(0, balance - total_tokens)

            # Update balance in background
            asyncio.create_task(update_balance(api_key, new_balance))

            # Return structured response
            return {
                "company": "signaturesi.com",
                "final_answer": final_answer,
                "reasoning": reasoning,
                "usage": {
                    "total_tokens": total_tokens,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "prompt_tokens": usage.prompt_tokens if usage else 0
                },
                "model": "Neo L1.0",
                "internal_engine": model_name,
                "balance": new_balance
            }

        except Exception as e:
            logger.error(f"{model_name} failed: {e}")
            continue

    # If all models fail
    raise HTTPException(
        status_code=503,
        detail={
            "company": "signaturesi.com",
            "status": "error",
            "message": "All Neo models failed. Please try again later."
        }
    )
