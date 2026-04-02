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

# Improved system prompt: explicitly ask for final answer in 'content'
SYSTEM_PROMPT = """Mode: Think. Triggers: [M]=MathHints, [C]=CodeSnippet, [H]=Health, [G]=General. Default:[G]. 
Format: ≤2 telegraphic sentences or 3 short bullets. No intro/outro/tags. Max 60 tokens.
IMPORTANT: You are an AI trained until July 27, 2025. Always put your final answer in the 'content' field, not only in reasoning."""

# Three models in order: 120B (primary) → 20B → efficient fallback
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",   # This model never returns empty content
]

def extract_answer_from_response(message_obj) -> str:
    """Try multiple ways to get the assistant's answer."""
    # 1. Direct content
    if hasattr(message_obj, 'content') and message_obj.content and message_obj.content.strip():
        return message_obj.content.strip()
    
    # 2. Reasoning field (for gpt-oss models)
    if hasattr(message_obj, 'reasoning') and message_obj.reasoning:
        reasoning = message_obj.reasoning
        # Try to extract final answer using patterns
        patterns = [
            r"(?:So|Therefore|Thus),?\s*(?:we )?answer:?\s*(.+?)(?:\n\n|$)",
            r"Final (?:answer|output):?\s*(.+?)(?:\n\n|$)",
            r"Output:?\s*(.+?)(?:\n\n|$)",
            r"(?:•|\*|\-)\s*(.+?)(?=\n(?:•|\*|\-)|$)",
        ]
        for pat in patterns:
            m = re.search(pat, reasoning, re.IGNORECASE | re.DOTALL)
            if m:
                ans = m.group(1).strip()
                if ans:
                    return ans
        # If no pattern matches, take the last non-empty line that looks like an answer
        lines = reasoning.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and len(line) > 10 and not line.startswith(("Mode:", "We need", "The user", "Default", "Format", "Triggers", "IMPORTANT")):
                return line
        # Fallback: return the whole reasoning truncated
        return reasoning[:300].strip()
    
    # 3. If message_obj has a 'text' attribute (some models)
    if hasattr(message_obj, 'text') and message_obj.text:
        return message_obj.text.strip()
    
    # 4. Last resort
    logger.error(f"Could not extract answer from message_obj: {message_obj}")
    return None

async def call_groq_with_fallback(messages):
    """Try each model with retries, return (completion, model_name)."""
    per_model_retries = 2
    for model in GROQ_MODELS:
        for attempt in range(per_model_retries):
            try:
                completion = groq_client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=0.7,
                    max_completion_tokens=300,   # Increased to avoid truncation
                    top_p=1,
                    stream=False,
                )
                logger.info(f"Success with model {model} on attempt {attempt+1}")
                return completion, model
            except Exception as e:
                logger.warning(f"Model {model}, attempt {attempt+1} failed: {e}")
                await asyncio.sleep(1)
        logger.warning(f"Model {model} failed, switching to next.")
    raise HTTPException(status_code=500, detail="All models failed")

@app.get("/")
def home():
    return {"status": "Online", "brand": "Signaturesi", "model": "Neo L1.0 (2025 Edition)"}

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

@app.post("/v1/chat/completions")
async def chat_proxy(request: Request, authorization: str = Header(None)):
    # 1. Validate API key
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API Key")
    user_api_key = authorization.replace("Bearer ", "")

    # 2. Parse request
    body = await request.json()
    if body.get("model") != "Neo-L1.0":
        raise HTTPException(status_code=400, detail="Invalid model. Use 'Neo-L1.0'")
    user_messages = body.get("messages")
    if not user_messages or not isinstance(user_messages, list):
        raise HTTPException(status_code=400, detail="Missing or invalid 'messages' array")

    # 3. Check balance
    try:
        response = supabase.table("users").select("token_balance").eq("api_key", user_api_key).execute()
        if not response.data:
            raise HTTPException(status_code=401, detail="Invalid API Key")
        current_balance = response.data[0].get("token_balance", 0)
    except Exception as e:
        logger.error(f"Supabase Error: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    if current_balance <= 0:
        raise HTTPException(status_code=402, detail="Insufficient Balance")

    # 4. Call Groq
    messages_for_groq = [{"role": "system", "content": SYSTEM_PROMPT}] + user_messages
    ai_response, used_model = await call_groq_with_fallback(messages_for_groq)

    # 5. Extract answer robustly
    message_obj = ai_response.choices[0].message
    assistant_content = extract_answer_from_response(message_obj)

    if assistant_content is None:
        # Log the full response for debugging
        logger.error(f"Failed to extract answer. Full response: {ai_response}")
        assistant_content = "Sorry, I encountered an internal issue. Please try again."

    # 6. Deduct tokens (only completion tokens)
    tokens_used = ai_response.usage.completion_tokens
    new_balance = max(0, current_balance - tokens_used)
    supabase.table("users").update({"token_balance": new_balance}).eq("api_key", user_api_key).execute()

    # 7. Return simplified response
    return {
        "message": assistant_content,
        "usage": {
            "completion_tokens": tokens_used,
            "total_tokens": ai_response.usage.total_tokens
        },
        "model": "Neo-L1.0 (2025)",
        "internal_model": used_model
    }
