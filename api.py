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

# Validation check for environment variables
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

# Model Shifting: If the first fails, it moves to the next.
# Note: Added valid Groq models to ensure it doesn't fail.
MODELS = [
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "llama-3.1-8b-instant"
]

# Optimized Prompt to force JSON and strict Token Budget
SYSTEM_PROMPT = """Identity: Neo L1.0. Deployment: Jan 1, 2026.
Style: High-Density Reasoning. No filler.

Rules:
- Return ONLY valid JSON in this exact format: {"final_answer": "...", "reasoning": "..."}
- Keep "reasoning" extremely short (max 50 tokens) to save cost.
- Maximize the length and quality of "final_answer".
- Use provided Local Context strictly.
- Do NOT hallucinate outside knowledge.
"""

# -----------------------------
# 2. Pydantic Models
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

# -----------------------------
# 3. Custom Branding & Error Handlers
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
# 4. Knowledge Engine (RAG)
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
# 5. Helper Functions
# -----------------------------
def get_user(api_key: str):
    return SUPABASE.table("users") \
        .select("token_balance") \
        .eq("api_key", api_key) \
        .maybe_single() \
        .execute()

# -----------------------------
# 6. API Routes
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
    local_data = get_neo_knowledge(user_msg)

    final_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if local_data:
        final_messages.append({"role": "system", "content": f"Local Context:\n{local_data}"})
        
    # TOKEN SAVER: Only include the last 2 messages from history to prevent snowballing costs
    clean_history = payload.messages[-2:] if len(payload.messages) > 2 else payload.messages
    final_messages.extend(clean_history)

    # -----------------------------
    # MODEL SHIFTING (Fallback Loop)
    # -----------------------------
    for model_name in MODELS:
        try:
            response = GROQ.chat.completions.create(
                model=model_name,
                messages=final_messages,
                temperature=0.6,
                max_tokens=4096, # High limit for long research papers
                response_format={"type": "json_object"}
            )

            # Extract raw string content
            raw_content = response.choices[0].message.content or "{}"
            
            # Safe JSON Parsing
            try:
                parsed_data = json.loads(raw_content)
            except json.JSONDecodeError:
                parsed_data = {"final_answer": raw_content, "reasoning": "Output was too long for JSON format."}

            # Real Groq Usage
            real_usage = response.usage
            tokens_used = getattr(real_usage, "total_tokens", 0)
            
            new_balance = max(0, balance - tokens_used)

            # Async background update for database
            asyncio.create_task(asyncio.to_thread(
                lambda: SUPABASE.table("users")
                .update({"token_balance": new_balance})
                .eq("api_key", api_key)
                .execute()
            ))

            # Matched with frontend expectations
            return {
                "company": "signaturesi.com",
                "final_answer": parsed_data.get("final_answer", "No answer generated."),
                "reasoning": parsed_data.get("reasoning", ""),
                "usage": {
                    "total": tokens_used,
                    "prompt": getattr(real_usage, "prompt_tokens", 0),
                    "completion": getattr(real_usage, "completion_tokens", 0)
                },
                "model": "Neo L1.0",
                "internal_engine": model_name,
                "balance": new_balance
            }

        except Exception as e:
            logger.error(f"{model_name} failed: {e}")
            continue # Shift to the next model in the list

    # If all models fail
    raise HTTPException(
        status_code=503, 
        detail={"company": "signaturesi.com", "status": "error", "message": "All Neo models failed or are overloaded."}
    )
