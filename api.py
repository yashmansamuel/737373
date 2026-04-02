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

# Load environment variables
load_dotenv()

# -----------------------------
# Logger Setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SignaturesiNeo")

# -----------------------------
# FastAPI App Initialization
# -----------------------------
app = FastAPI(title="Signaturesi Neo L1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Service Clients
# -----------------------------
SUPABASE: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)
GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# Neo Surgical Configuration
# -----------------------------
MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b",
    "llama-3.1-8b-instant"
]

# Optimized Surgical System Prompt
SYSTEM_PROMPT = (
    "Act: Neo-Surgical Intelligence. Goal: Max Accuracy, Min Tokens. "
    "Reasoning Rule: Internal hidden reasoning MUST NOT exceed 200 tokens. "
    "Search Rule: Use 'browser_search' ONLY for top 3-5 pages. No deep crawling. "
    "Output Rule: No fillers, no citations (【†】, [1]), no 'Searching...' text. "
    "Language: Respond natively in user's language (Roman Urdu/Hindi/English). "
    "Constraint: Answer must be concise and complete. Never leave sentences unfinished."
)

# -----------------------------
# Pydantic Models
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

class NewKeyResponse(BaseModel):
    api_key: str
    balance: int
    status: str

# -----------------------------
# Helper: Content Extraction & Cleaning
# -----------------------------
def clean_neo_output(message_obj) -> str:
    content = getattr(message_obj, 'content', "") or ""
    
    # 1. Cleaning Artifacts (Citations)
    content = re.sub(r'【.*?】', '', content)
    content = re.sub(r'\[\d+\]', '', content)
    content = re.sub(r'†\w+', '', content)
    
    if not content.strip():
        return "Unable to generate a clean response. Please try again."
    
    return content.strip()

# -----------------------------
# Token-Saver AI Call Logic
# -----------------------------
async def call_ai_with_fallback(messages: List[dict]) -> Tuple[object, str]:
    # 🔥 TOKEN SAVER: System Prompt + Last User Message only
    optimized_context = [{"role": "system", "content": SYSTEM_PROMPT}, messages[-1]]

    for model in MODELS:
        try:
            kwargs = {
                "model": model,
                "messages": optimized_context,
                "temperature": 0.3, # Precision focus
                "max_completion_tokens": 600, # Buffer for tables/native text
                "stream": False,
            }
            
            if "gpt-oss" in model:
                kwargs["reasoning_effort"] = None # Saves expensive reasoning tokens
                kwargs["tools"] = [{"type": "browser_search"}]

            completion = GROQ_CLIENT.chat.completions.create(**kwargs)
            logger.info(f"Success with model: {model}")
            return completion, model
            
        except Exception as e:
            logger.error(f"Model {model} failed: {str(e)}")
            continue
            
    raise HTTPException(503, "All Neo-Engines are currently exhausted.")

# -----------------------------
# API Endpoints
# -----------------------------

@app.get("/")
def health():
    return {"status": "online", "model": "Neo L1.0", "brand": "Signaturesi", "mode": "Surgical"}

@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    resp = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    if not resp.data:
        raise HTTPException(404, "API key not found")
    return {"api_key": api_key, "balance": resp.data["token_balance"]}

@app.post("/v1/user/new-key", response_model=NewKeyResponse)
async def generate_key(request: Request):
    try:
        new_key = f"sig-neo-{secrets.token_urlsafe(16)}"
        ip_country = request.headers.get("cf-ipcountry", "Global")
        SUPABASE.table("users").insert({
            "api_key": new_key,
            "token_balance": 2000,
            "country": ip_country
        }).execute()
        return {"api_key": new_key, "balance": 2000, "status": "Neo Active"}
    except Exception as e:
        logger.error(f"DB Error: {e}")
        raise HTTPException(500, "Failed to generate key.")

@app.post("/v1/chat/completions")
async def neo_chat_proxy(payload: ChatRequest, authorization: str = Header(None)):
    # 1. Security Check
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header.")
    
    api_key = authorization.replace("Bearer ", "")
    user_query = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    
    if not user_query.data:
        raise HTTPException(401, "Invalid API Key.")
    
    balance = user_query.data["token_balance"]
    if balance <= 0:
        raise HTTPException(402, "Insufficient token balance.")

    # 2. AI Processing
    ai_response, used_model = await call_ai_with_fallback(payload.messages)

    # 3. Output Cleaning
    final_answer = clean_neo_output(ai_response.choices[0].message)

    # 4. Accurate Billing
    tokens_used = ai_response.usage.total_tokens
    new_balance = max(0, balance - tokens_used)

    # Background Update
    asyncio.create_task(
        asyncio.to_thread(
            lambda: SUPABASE.table("users").update({"token_balance": new_balance}).eq("api_key", api_key).execute()
        )
    )

    # 5. Standard JSON Response
    return {
        "id": f"neo_{secrets.token_hex(6)}",
        "object": "chat.completion",
        "model": "Neo-L1.0",
        "choices": [
            {
                "message": {"role": "assistant", "content": final_answer},
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": ai_raw.usage.prompt_tokens if 'ai_raw' in locals() else 0, # Safety
            "completion_tokens": ai_response.usage.completion_tokens,
            "total_tokens": tokens_used
        },
        "engine_details": {
            "internal_model": used_model,
            "provider": "Signaturesi"
        }
    }
