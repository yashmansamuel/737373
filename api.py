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

# Updated system prompt - implies 2025 training knowledge
SYSTEM_PROMPT = """You are Neo L1.0, an advanced AI assistant with knowledge up to 2025. 
You have real-time web search capabilities. 
Mode: Think. Format: ≤2 telegraphic sentences or 3 short bullets. 
No intro/outro/tags. Max 60 tokens. 
Use browser search when asked about current events, latest news, or real-time information."""

# Models with browser search support
GROQ_MODELS = [
    "openai/gpt-oss-120b",              # Primary: 120B param model with tool support
    "openai/gpt-oss-safeguard-20b",     # Fallback: also supports tools
]

def extract_answer_from_reasoning(reasoning: str) -> str:
    """Extract final answer from reasoning text (for models that use reasoning)."""
    if not reasoning:
        return ""
    patterns = [
        r"So (?:we )?answer:?\s*(.+?)(?:\n\n|$)",
        r"Final (?:answer|output):?\s*(.+?)(?:\n\n|$)",
        r"Output:?\s*(.+?)(?:\n\n|$)",
    ]
    for pat in patterns:
        m = re.search(pat, reasoning, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
    # Take last bullet or sentence
    lines = reasoning.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and len(line) > 10 and not line.startswith(("Mode:", "We need", "The user")):
            return line
    return reasoning[-200:].strip()

async def call_groq_with_fallback(messages):
    """Try each model with browser search enabled, return first successful response."""
    per_model_retries = 2
    for model in GROQ_MODELS:
        for attempt in range(per_model_retries):
            try:
                # Enable browser search tool for GPT-OSS models
                completion = groq_client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=0.7,
                    max_completion_tokens=200,   # Enough for reasoning + answer + search results
                    top_p=1,
                    stream=False,
                    tools=[{"type": "web_search"}],  # Enable browser search
                )
                logger.info(f"Success with model {model} on attempt {attempt+1}")
                return completion, model
            except Exception as e:
                logger.warning(f"Model {model}, attempt {attempt+1} failed: {e}")
                await asyncio.sleep(1)
        logger.warning(f"Model {model} failed after {per_model_retries} attempts, switching.")
    raise HTTPException(status_code=500, detail="All models failed")

@app.get("/")
def home():
    return {"status": "Online", "brand": "Signaturesi", "model": "Neo L1.0 (GPT-OSS-120B)"}

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

    # 2. Parse and validate request
    body = await request.json()
    if body.get("model") != "Neo-L1.0":
        raise HTTPException(status_code=400, detail="Invalid model. Use 'Neo-L1.0'")
    user_messages = body.get("messages")
    if not user_messages or not isinstance(user_messages, list):
        raise HTTPException(status_code=400, detail="Missing or invalid 'messages' array")

    # 3. Check user balance
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

    # 4. Call Groq with fallback (browser search enabled)
    messages_for_groq = [{"role": "system", "content": SYSTEM_PROMPT}] + user_messages
    ai_response, used_model = await call_groq_with_fallback(messages_for_groq)

    # 5. Extract assistant content
    message_obj = ai_response.choices[0].message
    assistant_content = message_obj.content or ""

    # If content empty (some models put answer in reasoning), extract from reasoning
    if not assistant_content.strip() and hasattr(message_obj, 'reasoning') and message_obj.reasoning:
        assistant_content = extract_answer_from_reasoning(message_obj.reasoning)
        logger.info(f"Extracted answer from reasoning for model {used_model}")

    if not assistant_content.strip():
        assistant_content = "I'm unable to generate a response. Please try again."

    # 6. Check if web search was used (for logging/debugging)
    if hasattr(message_obj, 'executed_tools') and message_obj.executed_tools:
        logger.info(f"Web search executed with {len(message_obj.executed_tools)} tool calls")
        # Optional: You can add search results to response if needed

    # 7. Deduct only completion tokens
    tokens_used = ai_response.usage.completion_tokens
    new_balance = max(0, current_balance - tokens_used)
    supabase.table("users").update({"token_balance": new_balance}).eq("api_key", user_api_key).execute()

    # 8. Return simplified response
    return {
        "message": assistant_content,
        "usage": {
            "completion_tokens": tokens_used,
            "total_tokens": ai_response.usage.total_tokens
        },
        "model": "Neo-L1.0",
        "internal_model": used_model   # Optional, for debugging
    }
