import os
import logging
import secrets
import re
import asyncio
from typing import List, Tuple, Optional
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# Load Environment
load_dotenv()

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SignaturesiNeo")

app = FastAPI(title="Signaturesi Neo L1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clients Initialization
SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Model Configuration (Triple Triple Fallback)
MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b"
]

# Surgical System Prompt
SYSTEM_PROMPT = (
    "Mode: Neo-Surgical. Goal: Precision & Token Efficiency. "
    "Rule 1: Reasoning <150 tokens. Rule 2: No citations or search tags. "
    "Rule 3: Native adaptive language. Rule 4: Max 4 bullets or 3 sentences. "
    "Search: Only for live facts. No search for logic/code."
)

# Data Models
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

# Content Cleaning Logic
def clean_neo_content(message_obj) -> str:
    content = getattr(message_obj, 'content', "") or ""
    # Remove search artifacts
    content = re.sub(r'【.*?】', '', content)
    content = re.sub(r'\[\d+\]', '', content)
    content = re.sub(r'†\w+', '', content)
    return content.strip()

# Core AI Logic with Anti-Spam Delays
async def call_ai_surgical(messages: List[dict]) -> Tuple[object, str]:
    # Prune context to save 8k/min limit
    optimized_context = [{"role": "system", "content": SYSTEM_PROMPT}, messages[-1]]
    
    # Check if query needs search (Saves Search Quota)
    query_lower = messages[-1]['content'].lower()
    needs_search = any(k in query_lower for k in ["latest", "today", "news", "price", "vs", "weather"])

    for i, model in enumerate(MODELS):
        try:
            # 7-Second Cooldown between switches to prevent "All Exhausted"
            if i > 0:
                logger.warning(f"Cooldown: Waiting 7s before switching to {model}")
                await asyncio.sleep(7)

            kwargs = {
                "model": model,
                "messages": optimized_context,
                "temperature": 0.3,
                "max_completion_tokens": 600,
                "stream": False,
            }
            
            # Disable reasoning tokens to prevent 10k burn
            if "gpt-oss" in model:
                kwargs["reasoning_effort"] = None 
                if needs_search:
                    kwargs["tools"] = [{"type": "browser_search"}]

            completion = GROQ_CLIENT.chat.completions.create(**kwargs)
            logger.info(f"Successfully processed with {model}")
            return completion, model

        except Exception as e:
            logger.error(f"Engine {model} failed: {e}")
            if "rate_limit" in str(e).lower():
                await asyncio.sleep(2) # Extra buffer for rate limits
            continue

    raise HTTPException(503, "❌ Error: All surgical engines exhausted. Please wait 30 seconds.")

# --- Endpoints ---

@app.get("/")
def health():
    return {"status": "online", "model": "Neo L1.0", "engine": "Surgical-v3"}

@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    resp = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    if not resp.data:
        raise HTTPException(404, "Invalid API Key")
    return {"api_key": api_key, "balance": resp.data["token_balance"]}

@app.post("/v1/user/new-key")
async def generate_key(request: Request):
    new_key = f"sig-neo-{secrets.token_urlsafe(16)}"
    country = request.headers.get("cf-ipcountry", "Global")
    SUPABASE.table("users").insert({"api_key": new_key, "token_balance": 2000, "country": country}).execute()
    return {"api_key": new_key, "balance": 2000, "status": "Key Activated"}

@app.post("/v1/chat/completions")
async def neo_chat_proxy(payload: ChatRequest, authorization: str = Header(None)):
    # 1. Auth & Balance
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Auth Failed")
    api_key = authorization.replace("Bearer ", "")

    user = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    if not user.data or user.data["token_balance"] <= 0:
        raise HTTPException(402, "💰 Balance: 0 tokens. Please recharge.")

    # 2. AI Call
    ai_response, used_model = await call_ai_surgical(payload.messages)

    # 3. Clean Output
    final_text = clean_neo_content(ai_response.choices[0].message)
    if not final_text:
        final_text = "Analysis complete. No output generated. Please rephrase."

    # 4. Token Billing
    tokens_used = ai_response.usage.total_tokens
    new_balance = max(0, user.data["token_balance"] - tokens_used)

    # 5. Async Database Update
    asyncio.create_task(
        asyncio.to_thread(
            lambda: SUPABASE.table("users").update({"token_balance": new_balance}).eq("api_key", api_key).execute()
        )
    )

    return {
        "id": f"neo_{secrets.token_hex(4)}",
        "object": "chat.completion",
        "model": "Neo-L1.0",
        "choices": [{
            "message": {"role": "assistant", "content": final_text},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": ai_response.usage.prompt_tokens,
            "completion_tokens": ai_response.usage.completion_tokens,
            "total_tokens": tokens_used
        },
        "balance_remaining": new_balance,
        "engine": used_model
    }
