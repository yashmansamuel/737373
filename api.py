import os
import logging
import secrets
import re
import asyncio
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

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
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    logger.info("Connected to Supabase and Groq successfully.")
except Exception as e:
    logger.error(f"Initialization Error: {e}")
    raise RuntimeError("Cannot connect to Supabase or Groq")

SYSTEM_PROMPT = """Mode: Think. Triggers: [M]=MathHints, [C]=CodeSnippet, [H]=Health, [G]=General. Default:[G]. Format: ≤2 telegraphic sentences or 3 short bullets. No intro/outro/tags. Max 60 tokens.
Your knowledge was last updated on July 27, 2025. You are a 2025-era AI model."""

# Note: Ensure these model IDs are correct for your Groq access
GROQ_MODELS = [
    "llama-3.3-70b-versatile", 
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
]

def extract_answer_from_reasoning(reasoning: str) -> str:
    if not reasoning:
        return ""
    patterns = [
        r"(?:So|Therefore|Thus),?\s*(?:we )?answer:?\s*(.+?)(?:\n\n|$)",
        r"Final (?:answer|output):?\s*(.+?)(?:\n\n|$)",
        r"Output:?\s*(.+?)(?:\n\n|$)",
    ]
    for pat in patterns:
        m = re.search(pat, reasoning, re.IGNORECASE | re.DOTALL)
        if m:
            ans = m.group(1).strip()
            if ans: return ans
    
    lines = [l.strip() for l in reasoning.split('\n') if l.strip()]
    for line in reversed(lines):
        if len(line) > 10 and not line.startswith(("Mode:", "We need")):
            return line
    return reasoning[:200].strip()

async def call_groq_with_fallback(messages):
    loop = asyncio.get_event_loop()
    for model in GROQ_MODELS:
        for attempt in range(2):
            try:
                # Running sync call in thread to avoid blocking
                completion = await loop.run_in_executor(
                    None, 
                    lambda: groq_client.chat.completions.create(
                        messages=messages,
                        model=model,
                        temperature=0.7,
                        max_tokens=2048, # Changed from max_completion_tokens
                        top_p=1,
                        stream=False
                    )
                )
                logger.info(f"Success with model {model}")
                return completion, model
            except Exception as e:
                logger.warning(f"Model {model}, attempt {attempt+1} failed: {e}")
                await asyncio.sleep(1)
    raise HTTPException(500, "All models failed")

@app.get("/")
def home():
    return {"status": "Online", "brand": "Signaturesi", "model": "Neo L1.0 (2025)"}

@app.get("/v1/user/balance")
def get_balance(api_key: str):
    resp = supabase.table("users").select("token_balance").eq("api_key", api_key).execute()
    if not resp.data:
        raise HTTPException(404, "API Key not found")
    return {"api_key": api_key, "balance": resp.data[0]["token_balance"]}

@app.post("/v1/user/new-key")
async def generate_key(request: Request):
    new_key = "sig-live-" + secrets.token_urlsafe(16)
    country = request.headers.get("x-vercel-ip-country") or "Unknown"
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

    # Check balance
    resp = supabase.table("users").select("token_balance").eq("api_key", user_api_key).execute()
    if not resp.data:
        raise HTTPException(401, "Invalid API Key")
    
    balance = resp.data[0]["token_balance"]
    if balance <= 0:
        raise HTTPException(402, "Insufficient Balance")

    user_messages = body.get("messages", [])
    messages_for_groq = [{"role": "system", "content": SYSTEM_PROMPT}] + user_messages
    
    ai_response, used_model = await call_groq_with_fallback(messages_for_groq)

    message_obj = ai_response.choices[0].message
    assistant_content = message_obj.content or ""

    # Check for reasoning if content is empty (for specific models)
    if not assistant_content.strip() and hasattr(message_obj, 'reasoning'):
        assistant_content = extract_answer_from_reasoning(message_obj.reasoning)

    if not assistant_content.strip():
        assistant_content = "I'm unable to generate a response."

    # Update Balance
    tokens_used = ai_response.usage.completion_tokens
    new_balance = max(0, balance - tokens_used)
    try:
        supabase.table("users").update({"token_balance": new_balance}).eq("api_key", user_api_key).execute()
    except Exception as e:
        logger.error(f"Balance update failed: {e}")

    return {
        "message": assistant_content.strip(),
        "usage": {
            "completion_tokens": tokens_used,
            "total_tokens": ai_response.usage.total_tokens
        },
        "model": "Neo-L1.0 (2025)",
        "internal_model": used_model
    }
