import os
import logging
import secrets
import asyncio
from typing import List
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# -----------------------------
# Setup
# -----------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-L1.0-Core")

app = FastAPI(title="Neo L1.0 Engine")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase client
SUPABASE: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# GROQ client
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Models
MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant"
]

# Weaponized Prompt
SYSTEM_PROMPT = """Identity: Neo L1.0. Deployment: Jan 1, 2026.
Style: High-Density Reasoning. No filler. Max 2000 tokens.
Use 'Local Context' for all recent facts."""

# -----------------------------
# Models
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

# -----------------------------
# Helpers
# -----------------------------
def get_neo_knowledge(user_query: str) -> str:
    """Fetch local knowledge from knowledge.txt"""
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        
        if not os.path.exists(file_path):
            return ""
        
        query_words = [w.lower().strip(".,/") for w in user_query.split() if len(w) > 3]
        matches = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_lower = line.lower()
                if any(word in line_lower for word in query_words):
                    matches.append(line.strip())
                if len(matches) >= 5:  # max 5 lines per query
                    break
        return " | ".join(matches)
    except Exception as e:
        logger.error(f"Knowledge retrieval error: {e}")
        return ""

def extract_content(msg):
    return getattr(msg, "content", "") or "No response"

def get_user(api_key: str):
    return SUPABASE.table("users") \
        .select("token_balance") \
        .eq("api_key", api_key) \
        .maybe_single() \
        .execute()

async def update_balance_async(api_key: str, new_balance: int):
    await asyncio.to_thread(
        SUPABASE.table("users")
        .update({"token_balance": new_balance})
        .eq("api_key", api_key)
        .execute
    )

# -----------------------------
# Root Route
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def root_html():
    return """
    <html>
        <head><title>Neo L1.0 Engine</title></head>
        <body>
            <h1>Neo L1.0 Engine is Running</h1>
            <p>Use the API endpoints:</p>
            <ul>
                <li>POST /v1/chat/completions</li>
                <li>GET /v1/user/balance?api_key=YOUR_KEY</li>
                <li>POST /v1/user/new-key</li>
            </ul>
        </body>
    </html>
    """

# -----------------------------
# API Routes
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
        return {"api_key": api_key}
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

    # --- Knowledge Injection ---
    user_msg = payload.messages[-1]["content"]
    local_data = get_neo_knowledge(user_msg)
    
    final_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if local_data:
        final_messages.append({"role": "system", "content": f"Local Context: {local_data}"})
    
    final_messages.extend(payload.messages)

    for model_name in MODELS:
        try:
            response = GROQ.chat.completions.create(
                model=model_name,
                messages=final_messages,
                temperature=0.6,
                max_completion_tokens=2000
            )

            reply = extract_content(response.choices[0].message)
            tokens_used = response.usage.total_tokens
            new_balance = max(0, balance - tokens_used)

            # async update
            asyncio.create_task(update_balance_async(api_key, new_balance))

            return {
                "message": reply,
                "usage": {"total_tokens": tokens_used},
                "model": "Neo L1.0",
                "internal_engine": model_name,
                "balance": new_balance
            }

        except Exception as e:
            logger.error(f"{model_name} failed: {e}")
            continue

    raise HTTPException(503, "All Neo models failed")
