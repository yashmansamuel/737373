import os
import re
import logging
import secrets
import asyncio
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# --- 1. Environment & Logging ---
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("NeoAPI")

app = FastAPI(title="Signaturesi Neo L1.0", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Configuration & Personality ---
# Yeh prompt CoT ko "Human-like" banayega
HUMAN_COT_PROMPT = (
    "Your internal thinking (CoT) must reflect a high-IQ human expert's stream of consciousness. "
    "Don't just repeat rules; weigh options, show brief curiosity, and be sharp. "
    "Example CoT: 'User wants a quick greeting. I'll stay concise but welcoming. No fluff needed.' "
    "\n\nResponse Rules: Mode: Think. Triggers: [M]=MathHints, [C]=CodeSnippet, [H]=Health, [G]=General. "
    "Default:[G]. Format: ≤2 telegraphic sentences or 3 short bullets. No intro/outro/tags. "
    "Max 60 tokens. Knowledge Cutoff: July 2025."
)

GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b",
    "llama-3.1-8b-instant",
]

# --- 3. Data Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = Field(..., example="Neo-L1.0")
    messages: List[ChatMessage]

# --- 4. Infrastructure Initialization ---
try:
    supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    logger.info("✅ Connected to Supabase & Groq.")
except Exception as e:
    logger.critical(f"❌ Infrastructure Error: {e}")
    raise RuntimeError("System Boot Failure")

# --- 5. Helper Functions ---
def extract_final_answer(reasoning: str) -> str:
    """CoT se final answer nikalne ka expert logic."""
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
    
    # Fallback: Last non-meta line
    lines = [l.strip() for l in reasoning.split('\n') if len(l.strip()) > 10]
    return lines[-1] if lines else reasoning[:150]

async def call_ai_engine(messages: List[Dict]):
    """Multi-model fallback with creative temperature."""
    for model in GROQ_MODELS:
        for attempt in range(2):
            try:
                # Temperature 0.8 CoT ko natural banata hai
                response = groq_client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=0.8, 
                    max_completion_tokens=1500,
                    reasoning_effort="medium",
                    stream=False
                )
                return response, model
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                await asyncio.sleep(1)
    raise HTTPException(503, "AI Engines busy")

# --- 6. API Routes ---

@app.get("/")
def health():
    return {"status": "online", "engine": "Neo-L1.0"}

@app.get("/v1/user/balance")
async def get_balance(api_key: str):
    res = supabase.table("users").select("token_balance").eq("api_key", api_key).execute()
    if not res.data:
        raise HTTPException(404, "Key not found")
    return {"balance": res.data[0]["token_balance"]}

@app.post("/v1/user/new-key")
async def generate_key(request: Request):
    new_key = f"sig-live-{secrets.token_urlsafe(24)}"
    country = request.headers.get("x-vercel-ip-country", "Global")
    try:
        supabase.table("users").insert({
            "api_key": new_key, "token_balance": 1000, "country": country
        }).execute()
        return {"api_key": new_key, "balance": 1000}
    except Exception:
        raise HTTPException(500, "Key generation failed")

@app.post("/v1/chat/completions")
async def chat_proxy(payload: ChatRequest, authorization: str = Header(None)):
    # 1. Validation
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Auth Header Missing")
    
    user_key = authorization.replace("Bearer ", "").strip()
    
    # 2. Check Balance
    user_data = supabase.table("users").select("token_balance").eq("api_key", user_key).execute()
    if not user_data.data:
        raise HTTPException(403, "Invalid Key")
    
    balance = user_data.data[0]["token_balance"]
    if balance <= 0:
        raise HTTPException(402, "Low Balance")

    # 3. AI Call
    messages = [{"role": "system", "content": HUMAN_COT_PROMPT}] + [m.dict() for m in payload.messages]
    ai_res, used_model = await call_ai_engine(messages)

    # 4. Processing
    msg_obj = ai_res.choices[0].message
    content = msg_obj.content or ""
    reasoning = getattr(msg_obj, 'reasoning', "") or ""

    if not content.strip() and reasoning:
        content = extract_final_answer(reasoning)

    # 5. Billing (Only completion tokens)
    tokens_to_bill = ai_res.usage.completion_tokens
    new_balance = max(0, balance - tokens_to_bill)
    
    # Sync balance update
    supabase.table("users").update({"token_balance": new_balance}).eq("api_key", user_key).execute()

    return {
        "message": content.strip() or "Neo could not generate a response.",
        "reasoning": reasoning.strip(),
        "usage": {
            "billed_tokens": tokens_to_bill,
            "total_tokens": ai_res.usage.total_tokens
        },
        "model_info": {"public_name": "Neo-L1.0", "internal_engine": used_model},
        "remaining_balance": new_balance
    }
