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
logger = logging.getLogger("Neo-L1.0-Final")

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
SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# 3. GPT-Style Adaptive Settings
# -----------------------------
MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-safeguard-20b", "openai/gpt-oss-120b"]

# Updated for Native Fluidity & Citation Removal
SYSTEM_PROMPT = (
    "Act: Neo-Adaptive Intelligence. Task: Provide expert, high-quality responses. "
    "Language: Respond natively in the language used by the user (e.g., Urdu, Hindi, English). "
    "Style: Professional, fluid, and natural like GPT-4. No robotic phrasing. "
    "Rules: Strictly no internal reasoning text. Strictly NO search citations or source tags like [1] or 【†】. "
    "Completeness: Ensure every sentence is grammatically finished and polished. "
    "Constraint: Maintain high information density. Concise but fully helpful."
)

# -----------------------------
# 4. Helper: Advanced Cleaning
# -----------------------------
def clean_neo_output(text: str) -> str:
    """Removes all search citations, source tags, and technical artifacts."""
    if not text: return ""
    # Remove citations like 【1†L168-L178】, [1], [2], etc.
    text = re.sub(r'【.*?】', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'†\w+', '', text)
    # Remove any stray markdown artifacts from search tools
    text = text.replace("Source:", "").replace("Citations:", "")
    return text.strip()

# -----------------------------
# 5. Data Models
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

# -----------------------------
# 6. AI Engine Logic
# -----------------------------
async def call_neo_engine(messages: List[dict]) -> Tuple[object, str]:
    # Context Optimization: System Prompt + Current Query
    optimized_context = [messages[0], messages[-1]] 

    for model_name in MODELS:
        try:
            params = {
                "model": model_name,
                "messages": optimized_context,
                "temperature": 0.5,           # Increased for GPT-like natural flow
                "max_completion_tokens": 300,  # Enough for detailed native responses
                "stream": False,
            }
            
            if "gpt-oss" in model_name:
                params["reasoning_effort"] = None 
                params["tools"] = [{"type": "browser_search"}]

            completion = GROQ_CLIENT.chat.completions.create(**params)
            return completion, model_name

        except Exception as e:
            err = str(e).lower()
            logger.warning(f"Engine {model_name} error: {err}")
            if "rate_limit" in err: await asyncio.sleep(0.6)
            continue

    raise HTTPException(503, "All Neo-Engines exhausted.")

# -----------------------------
# 7. Main Proxy Endpoint
# -----------------------------
@app.post("/v1/chat/completions")
async def neo_chat_proxy(payload: ChatRequest, authorization: str = Header(None)):
    # Auth Check
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Auth Required")
    
    api_key = authorization.replace("Bearer ", "")
    user = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    
    if not user.data or user.data["token_balance"] <= 0:
        raise HTTPException(402, "Insufficient tokens")

    # AI Execution
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + payload.messages
    ai_raw, engine_id = await call_neo_engine(full_messages)

    # Content Refinement (The Citation Killer)
    raw_text = ai_raw.choices[0].message.content or ""
    final_text = clean_neo_output(raw_text)

    # Billing Update
    total_tokens = ai_raw.usage.total_tokens
    new_bal = max(0, user.data["token_balance"] - total_tokens)

    asyncio.create_task(
        asyncio.to_thread(
            lambda: SUPABASE.table("users").update({"token_balance": new_bal}).eq("api_key", api_key).execute()
        )
    )

    # Standard OpenAI Format Response
    return {
        "id": f"neo_{secrets.token_hex(4)}",
        "object": "chat.completion",
        "model": "Neo-L1.0",
        "choices": [{
            "message": {"role": "assistant", "content": final_text},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": ai_raw.usage.prompt_tokens,
            "completion_tokens": ai_raw.usage.completion_tokens,
            "total_tokens": total_tokens
        },
        "engine_used": engine_id
    }

# -----------------------------
# 8. Utility Endpoints
# -----------------------------
@app.get("/v1/user/balance")
def get_balance(api_key: str):
    res = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    if not res.data: raise HTTPException(404, "Invalid Key")
    return {"api_key": api_key, "balance": res.data["token_balance"]}

@app.post("/v1/user/new-key")
async def generate_key(request: Request):
    new_key = f"sig-neo-{secrets.token_urlsafe(16)}"
    country = request.headers.get("cf-ipcountry", "Global")
    SUPABASE.table("users").insert({"api_key": new_key, "token_balance": 2000, "country": country}).execute()
    return {"api_key": new_key, "balance": 2000, "status": "Neo Active"}

@app.get("/")
def health(): return {"status": "online", "model": "Neo L1.0"}
