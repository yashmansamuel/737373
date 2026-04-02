import os
import logging
import secrets
import re
import asyncio
from typing import List, Tuple
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# -----------------------------
# 1. Logging & App Setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-L1.0-Final-Production")

app = FastAPI(title="Signaturesi Neo L1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 2. Service Clients
# -----------------------------
# Ensure SUPABASE_URL, SUPABASE_KEY, and GROQ_API_KEY are in your .env
SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# 3. GPT-Style Adaptive Settings
# -----------------------------
MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-safeguard-20b", "openai/gpt-oss-120b"]

# System Prompt: Native fluency and strict formatting rules
SYSTEM_PROMPT = (
    "Act: Neo-Adaptive Intelligence. Task: Provide expert, high-quality responses. "
    "Language: Respond natively in the language used by the user (Urdu, Hindi, English, etc.). "
    "Style: Professional, fluid, and natural like GPT-4. No robotic or repetitive phrasing. "
    "Rules: Strictly NO internal reasoning text. Strictly NO search citations or source tags like [1] or 【†】. "
    "Completeness: Never leave a list item or sentence unfinished. Ensure full closure. "
    "Constraint: Maintain high information density. Focus only on the current query to save tokens."
)

# -----------------------------
# 4. Helper: Final Citation & Artifact Killer
# -----------------------------
def clean_neo_output(text: str) -> str:
    """Removes all search citations, source tags, and technical artifacts."""
    if not text: return ""
    # Remove citations like 【1†L168-L178】, [1], [2], etc.
    text = re.sub(r'【.*?】', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'†\w+', '', text)
    # Remove any stray headers/markdown artifacts
    text = re.sub(r'(Source|Citations|Reference):.*', '', text, flags=re.IGNORECASE)
    return text.strip()

# -----------------------------
# 5. Data Models
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

# -----------------------------
# 6. Token-Saver AI Engine
# -----------------------------
async def call_neo_engine(messages: List[dict]) -> Tuple[object, str]:
    # 🔥 TOKEN SAVER: We only send System Prompt + the very last User Message.
    # This keeps input costs at ~150-200 tokens instead of 1000+.
    if len(messages) > 0:
        optimized_context = [{"role": "system", "content": SYSTEM_PROMPT}, messages[-1]]
    else:
        raise HTTPException(400, "No messages found in request.")

    for model_name in MODELS:
        try:
            params = {
                "model": model_name,
                "messages": optimized_context,
                "temperature": 0.5,           # Optimal for natural flow
                "max_completion_tokens": 500,  # Prevents truncated/incomplete answers
                "stream": False,
            }
            
            # Disable reasoning effort for models that support it to save billing
            if "gpt-oss" in model_name:
                params["reasoning_effort"] = None 
                params["tools"] = [{"type": "browser_search"}]

            completion = GROQ_CLIENT.chat.completions.create(**params)
            return completion, model_name

        except Exception as e:
            err = str(e).lower()
            logger.warning(f"Engine {model_name} failed: {err}")
            if "rate_limit" in err:
                await asyncio.sleep(0.7) # Anti-spam delay
            continue

    raise HTTPException(503, "All Neo Engines are currently busy. Please retry.")

# -----------------------------
# 7. Main Proxy Endpoint
# -----------------------------
@app.post("/v1/chat/completions")
async def neo_chat_proxy(payload: ChatRequest, authorization: str = Header(None)):
    # 1. Auth & Balance Verification
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Authorization header missing or malformed.")
    
    api_key = authorization.replace("Bearer ", "")
    user_query = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    
    if not user_query.data:
        raise HTTPException(401, "Invalid API Key.")
    
    current_balance = user_query.data["token_balance"]
    if current_balance <= 0:
        raise HTTPException(402, "Token balance exhausted.")

    # 2. AI Execution (Token-Saver Mode)
    ai_raw_res, engine_id = await call_neo_engine(payload.messages)

    # 3. Clean and Polish Output
    raw_content = ai_raw_res.choices[0].message.content or ""
    final_content = clean_neo_output(raw_content)

    # 4. Accurate Token Billing
    usage_stats = ai_raw_res.usage
    total_spent = usage_stats.total_tokens
    updated_balance = max(0, current_balance - total_spent)

    # Update database in background to maintain speed
    asyncio.create_task(
        asyncio.to_thread(
            lambda: SUPABASE.table("users").update({"token_balance": updated_balance}).eq("api_key", api_key).execute()
        )
    )

    # 5. Return Response in OpenAI Standard Format
    return {
        "id": f"neo_comp_{secrets.token_hex(6)}",
        "object": "chat.completion",
        "model": "Neo-L1.0",
        "choices": [{
            "message": {
                "role": "assistant", 
                "content": final_content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": usage_stats.prompt_tokens,
            "completion_tokens": usage_stats.completion_tokens,
            "total_tokens": total_spent
        }
    }

# -----------------------------
# 8. Utility Endpoints
# -----------------------------
@app.get("/v1/user/balance")
def get_balance(api_key: str):
    res = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    if not res.data:
        raise HTTPException(404, "Key not found.")
    return {"api_key": api_key, "balance": res.data["token_balance"]}

@app.post("/v1/user/new-key")
async def generate_key(request: Request):
    new_key = f"sig-neo-{secrets.token_urlsafe(20)}"
    country = request.headers.get("cf-ipcountry", "Global")
    SUPABASE.table("users").insert({
        "api_key": new_key, 
        "token_balance": 2000, 
        "country": country
    }).execute()
    return {"api_key": new_key, "balance": 2000, "status": "Neo Key Activated"}

@app.get("/")
def health_check():
    return {"status": "online", "model": "Neo L1.0", "engine": "Stateless/Token-Saver"}
