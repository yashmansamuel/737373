import os
import re
import logging
import secrets
import asyncio
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# --- 1. Setup & Environment ---
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("NeoAPI")

app = FastAPI(
    title="Signaturesi Neo L1.0",
    description="Professional AI Proxy API with CoT Transparency",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Constants & Configuration ---
SYSTEM_PROMPT = (
    "Mode: Think. Triggers: [M]=MathHints, [C]=CodeSnippet, [H]=Health, [G]=General. "
    "Default:[G]. Format: ≤2 telegraphic sentences or 3 short bullets. No intro/outro/tags. "
    "Max 60 tokens. Knowledge Cutoff: July 27, 2025."
)

GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b",
    "llama-3.1-8b-instant",
]

# --- 3. Schemas (Validation) ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = Field(..., example="Neo-L1.0")
    messages: List[ChatMessage]

class TokenUsage(BaseModel):
    billed_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    message: str
    reasoning: Optional[str] = None
    usage: TokenUsage
    model_info: Dict[str, str]
    remaining_balance: int

# --- 4. Infrastructure Clients ---
try:
    supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    logger.info("✅ Infrastructure connected successfully.")
except Exception as e:
    logger.critical(f"❌ Initialization Failed: {e}")
    raise RuntimeError("Backend services unavailable.")

# --- 5. Logic Helpers ---
def clean_reasoning_extract(reasoning: str) -> str:
    """Regex based answer extraction from CoT blocks."""
    if not reasoning: return ""
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
            if ans: return ans
    
    # Fallback to last substantial line
    lines = [l.strip() for l in reasoning.split('\n') if len(l.strip()) > 15]
    return lines[-1] if lines else reasoning[:200]

async def get_ai_completion(messages: List[Dict]):
    """Fallback mechanism for multiple Groq models."""
    for model in GROQ_MODELS:
        for attempt in range(2):
            try:
                response = groq_client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=0.6,
                    max_completion_tokens=1500,
                    reasoning_effort="medium",
                    stream=False
                )
                return response, model
            except Exception as e:
                logger.warning(f"Model {model} failed (Attempt {attempt+1}): {e}")
                await asyncio.sleep(1)
    raise HTTPException(503, "All AI engines are currently saturated.")

# --- 6. Endpoints ---

@app.get("/")
async def health_check():
    return {"status": "active", "engine": "Neo-L1.0", "uptime": "stable"}

@app.get("/v1/user/balance")
async def check_balance(api_key: str):
    res = supabase.table("users").select("token_balance").eq("api_key", api_key).execute()
    if not res.data:
        raise HTTPException(404, "Invalid API Key")
    return {"balance": res.data[0]["token_balance"]}

@app.post("/v1/user/new-key")
async def create_access_key(request: Request):
    new_key = f"sig-live-{secrets.token_urlsafe(24)}"
    country = request.headers.get("x-vercel-ip-country", "Global")
    
    try:
        supabase.table("users").insert({
            "api_key": new_key,
            "token_balance": 1000,
            "country": country
        }).execute()
        return {"api_key": new_key, "balance": 1000}
    except Exception as e:
        logger.error(f"Key Gen Error: {e}")
        raise HTTPException(500, "Could not issue new key.")

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_proxy(payload: ChatRequest, authorization: Optional[str] = Header(None)):
    # 1. Auth Validation
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Valid Bearer token required.")
    
    user_api_key = authorization.replace("Bearer ", "").strip()

    # 2. Database Session (Check Balance)
    user_res = supabase.table("users").select("token_balance").eq("api_key", user_api_key).execute()
    if not user_res.data:
        raise HTTPException(403, "Access denied: Invalid Key.")
    
    current_balance = user_res.data[0]["token_balance"]
    if current_balance <= 0:
        raise HTTPException(402, "Payment Required: Token balance exhausted.")

    # 3. AI Execution
    formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [m.dict() for m in payload.messages]
    ai_raw, internal_model = await get_ai_completion(formatted_messages)

    # 4. Data Extraction
    msg_obj = ai_raw.choices[0].message
    content = msg_obj.content or ""
    reasoning = getattr(msg_obj, 'reasoning', "") or ""

    # Parse if content is trapped in reasoning
    if not content.strip() and reasoning:
        content = clean_reasoning_extract(reasoning)
    
    # 5. Token Accounting
    # We only bill for completion_tokens (includes reasoning + final answer)
    billed = ai_raw.usage.completion_tokens
    new_balance = max(0, current_balance - billed)
    
    # Async background-style update
    supabase.table("users").update({"token_balance": new_balance}).eq("api_key", user_api_key).execute()

    return ChatResponse(
        message=content.strip() or "Process complete, no text returned.",
        reasoning=reasoning.strip(),
        usage=TokenUsage(
            billed_tokens=billed,
            prompt_tokens=ai_raw.usage.prompt_tokens,
            completion_tokens=ai_raw.usage.completion_tokens,
            total_tokens=ai_raw.usage.total_tokens
        ),
        model_info={
            "public_name": "Neo-L1.0",
            "engine": internal_model
        },
        remaining_balance=new_balance
    )
