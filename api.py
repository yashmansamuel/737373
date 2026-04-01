import os
import logging
import secrets
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq
import asyncio

load_dotenv()

# -----------------------------
# Logger Setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# FastAPI Init
# -----------------------------
app = FastAPI(title="Signaturesi Neo L1.0 API")

# -----------------------------
# CORS Middleware
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Environment Variables
# -----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# -----------------------------
# Clients Init
# -----------------------------
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    groq_client = Groq()
    logger.info("Connected to Supabase and Groq successfully.")
except Exception as e:
    logger.error(f"Initialization Error: {e}")
    raise RuntimeError("Cannot connect to Supabase or Groq")

# -----------------------------
# System Prompt (Shortened)
# -----------------------------
SYSTEM_PROMPT = "Concise reply in 1-2 short sentences. Max 30 tokens."

# -----------------------------
# Models for automatic fallback (order matters)
# -----------------------------
GROQ_MODELS = [
    "llama3-8b-8192",           # efficient, lower token usage
    "mixtral-8x7b-32768",
    "openai/gpt-oss-20b",       # fallback
]

# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def home():
    return {"status": "Online", "brand": "Signaturesi", "model": "Neo L1.0"}

# -----------------------------
# Get User Balance
# -----------------------------
@app.get("/v1/user/balance")
def get_balance(api_key: str):
    try:
        response = supabase.table("users").select("token_balance").eq("api_key", api_key).execute()
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=404, detail="API Key not found")
        balance = response.data[0].get("token_balance", 0)
        return {"api_key": api_key, "balance": balance}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Supabase Error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

# -----------------------------
# Generate New API Key
# -----------------------------
@app.post("/v1/user/new-key")
async def generate_key(request: Request):
    new_key = "sig-live-" + secrets.token_urlsafe(16)
    user_country = request.headers.get("x-vercel-ip-country") or request.headers.get("cf-ipcountry") or "Unknown"

    try:
        supabase.table("users").insert({
            "api_key": new_key,
            "token_balance": 1000,
            "country": user_country
        }).execute()
        return {"api_key": new_key, "balance": 1000, "country": user_country}
    except Exception as e:
        logger.error(f"Supabase Insert Error: {e}")
        raise HTTPException(status_code=500, detail="Cannot create new API key")

# -----------------------------
# Groq call with automatic model switching
# -----------------------------
async def call_groq_with_fallback(messages):
    """Try each model in GROQ_MODELS, with per‑model retries."""
    per_model_retries = 2

    for model in GROQ_MODELS:
        for attempt in range(per_model_retries):
            try:
                completion = groq_client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=0.7,
                    max_completion_tokens=100,   # Reduced from 2048
                    top_p=1,
                    stream=False,
                )
                logger.info(f"Success with model {model} on attempt {attempt+1}")
                return completion
            except Exception as e:
                logger.warning(f"Model {model}, attempt {attempt+1} failed: {e}")
                await asyncio.sleep(1)

        logger.warning(f"Model {model} failed after {per_model_retries} attempts, switching to next model.")

    raise HTTPException(status_code=500, detail="All Groq models failed after retries")

# -----------------------------
# Chat Endpoint with Groq + Auto Model Switching
# -----------------------------
@app.post("/v1/chat/completions")
async def chat_proxy(request: Request, authorization: str = Header(None)):
    # 1️⃣ Validate API Key
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API Key")
    user_api_key = authorization.replace("Bearer ", "")

    # 2️⃣ Parse JSON & Validate Model
    body = await request.json()
    model_name = body.get("model")
    if model_name != "Neo-L1.0":
        raise HTTPException(status_code=400, detail="Invalid model. Use 'Neo-L1.0'")
    user_messages = body.get("messages")
    if not user_messages or not isinstance(user_messages, list):
        raise HTTPException(status_code=400, detail="Missing or invalid 'messages' array")

    # 3️⃣ Check User Balance
    try:
        response = supabase.table("users").select("token_balance").eq("api_key", user_api_key).execute()
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=401, detail="Invalid API Key")
        current_balance = response.data[0].get("token_balance", 0)
    except Exception as e:
        logger.error(f"Supabase Error: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    if current_balance <= 0:
        raise HTTPException(status_code=402, detail="Insufficient Balance")

    # 4️⃣ Call Groq with fallback models
    messages_for_groq = [{"role": "system", "content": SYSTEM_PROMPT}] + user_messages
    ai_response = await call_groq_with_fallback(messages_for_groq)

    # 5️⃣ Deduct tokens (only completion tokens, not input)
    tokens_used = getattr(ai_response.usage, "completion_tokens", 50)
    new_balance = max(0, current_balance - tokens_used)
    supabase.table("users").update({"token_balance": new_balance}).eq("api_key", user_api_key).execute()

    # 6️⃣ Customize Response
    ai_response.model = "Neo-L1.0"
    return ai_response
