import os
import logging
import secrets
import re
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq
import asyncio

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Signaturesi Neo L1.0 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    groq_client = Groq()
    logger.info("Connected to Supabase and Groq successfully.")
except Exception as e:
    logger.error(f"Initialization Error: {e}")
    raise RuntimeError("Cannot connect to Supabase or Groq")

SYSTEM_PROMPT = """Mode: Think. Triggers: [M]=MathHints, [C]=CodeSnippet, [H]=Health, [G]=General. Default:[G]. Format: ≤2 telegraphic sentences or 3 short bullets. No intro/outro/tags. Max 60 tokens.
Your knowledge was last updated on July 27, 2025. You are a 2025-era AI model."""

# Three models: 120B (primary) -> 20B (fallback) -> efficient (last resort)
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]

async def call_groq_with_fallback(messages):
    """Try each model with reasoning_effort='none' for the first two."""
    per_model_retries = 2
    for model in GROQ_MODELS:
        for attempt in range(per_model_retries):
            try:
                # Common parameters
                kwargs = {
                    "messages": messages,
                    "model": model,
                    "temperature": 0.7,
                    "max_completion_tokens": 200,
                    "top_p": 1,
                    "stream": False,
                }
                # Add reasoning_effort only for gpt-oss models
                if "gpt-oss" in model:
                    kwargs["reasoning_effort"] = "none"   # This disables reasoning!
                
                completion = groq_client.chat.completions.create(**kwargs)
                logger.info(f"Success with model {model} on attempt {attempt+1}")
                return completion, model
            except Exception as e:
                logger.warning(f"Model {model}, attempt {attempt+1} failed: {e}")
                await asyncio.sleep(1)
        logger.warning(f"Model {model} failed, switching to next.")
    raise HTTPException(status_code=500, detail="All models failed")

@app.get("/")
def home():
    return {"status": "Online", "brand": "Signaturesi", "model": "Neo L1.0 (2025)"}

@app.get("/v1/user/balance")
def get_balance(api_key: str):
    try:
        response = supabase.table("users").select("token_balance").eq("api_key", api_key).execute()
        if not response.data:
            raise HTTPException(404, "API Key not found")
        return {"api_key": api_key, "balance": response.data[0]["token_balance"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Supabase Error: {e}")
        raise HTTPException(500, "Database error")

@app.post("/v1/user/new-key")
async def generate_key(request: Request):
    new_key = "sig-live-" + secrets.token_urlsafe(16)
    country = request.headers.get("x-vercel-ip-country") or request.headers.get("cf-ipcountry") or "Unknown"
    try:
        supabase.table("users").insert({"api_key": new_key, "token_balance": 1000, "country": country}).execute()
        return {"api_key": new_key, "balance": 1000, "country": country}
    except Exception as e:
        logger.error(f"Insert Error: {e}")
        raise HTTPException(500, "Cannot create key")

@app.post("/v1/chat/completions")
async def chat_proxy(request: Request, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing API Key")
    user_api_key = authorization.replace("Bearer ", "")

    body = await request.json()
    if body.get("model") != "Neo-L1.0":
        raise HTTPException(400, "Invalid model. Use 'Neo-L1.0'")
    user_messages = body.get("messages")
    if not user_messages:
        raise HTTPException(400, "Missing messages")

    # Check balance
    try:
        resp = supabase.table("users").select("token_balance").eq("api_key", user_api_key).execute()
        if not resp.data:
            raise HTTPException(401, "Invalid API Key")
        balance = resp.data[0]["token_balance"]
        if balance <= 0:
            raise HTTPException(402, "Insufficient Balance")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Balance error: {e}")
        raise HTTPException(500, "Database error")

    # Prepare messages
    messages_for_groq = [{"role": "system", "content": SYSTEM_PROMPT}] + user_messages
    ai_response, used_model = await call_groq_with_fallback(messages_for_groq)

    # Extract content – with reasoning_effort="none", it's always in content
    assistant_content = ai_response.choices[0].message.content
    if not assistant_content or not assistant_content.strip():
        # Fallback: try to get from reasoning (should not happen, but safe)
        if hasattr(ai_response.choices[0].message, 'reasoning'):
            assistant_content = ai_response.choices[0].message.reasoning or "No response."
        else:
            assistant_content = "I'm unable to generate a response."

    tokens_used = ai_response.usage.completion_tokens
    new_balance = max(0, balance - tokens_used)
    supabase.table("users").update({"token_balance": new_balance}).eq("api_key", user_api_key).execute()

    return {
        "message": assistant_content.strip(),
        "usage": {
            "completion_tokens": tokens_used,
            "total_tokens": ai_response.usage.total_tokens
        },
        "model": "Neo-L1.0 (2025)",
        "internal_model": used_model
    }
