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
# 1. Logger & App Setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-Surgical-v2")

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
# 3. Strict Surgical Settings
# -----------------------------
# Only 3 Models for Maximum Control
MODELS = [
    "openai/gpt-oss-120b", 
    "openai/gpt-oss-20b", 
    "openai/gpt-oss-safeguard-20b"
]

# THE TOKEN GUIDER PROMPT
SYSTEM_PROMPT = (
    "Act: Neo-Surgical Token Guider. Goal: Precise answer with MINIMUM token burn. "
    "STRICT RULE 1: Internal reasoning must be under 100 tokens. No deep thinking. "
    "STRICT RULE 2: Use 'browser_search' ONLY for live facts (prices, news). "
    "NEVER search for logic, coding, or general knowledge. "
    "STRICT RULE 3: Output must be direct. No intro, no 'Searching...', no citations. "
    "Language: Native adaptive (Roman Urdu/Hindi/English). "
    "Constraint: Max 150 words per response. Finish all sentences."
)

# -----------------------------
# 4. Helper: Cleaning & Artifact Removal
# -----------------------------
def clean_neo_output(text: str) -> str:
    if not text: return ""
    # Remove all types of citations and brackets
    text = re.sub(r'【.*?】', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'†\w+', '', text)
    return text.strip()

# -----------------------------
# 5. Data Models
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

# -----------------------------
# 6. Surgical Engine Logic
# -----------------------------
async def call_neo_engine(messages: List[dict]) -> Tuple[object, str]:
    # 🔥 TOKEN SAVER: Only send System Prompt + Last User Query
    user_query = messages[-1]['content']
    optimized_context = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_query}]

    # Smart Search Detection: Search sirf tab jab zaroori ho
    query_lower = user_query.lower()
    needs_search = any(x in query_lower for x in ["price", "latest", "today", "news", "weather", "vs"])

    for model_name in MODELS:
        try:
            params = {
                "model": model_name,
                "messages": optimized_context,
                "temperature": 0.2,            # Low temp = Less tokens/No fluff
                "max_completion_tokens": 500,   # Safe limit for complete answers
                "stream": False,
            }
            
            if "gpt-oss" in model_name:
                params["reasoning_effort"] = None  # 🔥 STOPS 10K TOKEN BURN
                if needs_search:
                    params["tools"] = [{"type": "browser_search"}]
                # If no search needed, 'tools' is not added to save overhead

            completion = GROQ_CLIENT.chat.completions.create(**params)
            return completion, model_name

        except Exception as e:
            logger.warning(f"Model {model_name} failed: {e}")
            await asyncio.sleep(0.5)
            continue

    raise HTTPException(503, "All surgical engines exhausted.")

# -----------------------------
# 7. Main API Endpoint
# -----------------------------
@app.post("/v1/chat/completions")
async def neo_chat_proxy(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Auth Required")
    
    api_key = authorization.replace("Bearer ", "")
    user = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    
    if not user.data or user.data["token_balance"] <= 0:
        raise HTTPException(402, "Balance Empty")

    # AI Execution
    ai_raw, engine_id = await call_neo_engine(payload.messages)

    # Output Refinement
    raw_content = ai_raw.choices[0].message.content or ""
    final_content = clean_neo_output(raw_content)

    # Billing (Total tokens including prompt/search)
    total_spent = ai_raw.usage.total_tokens
    new_bal = max(0, user.data["token_balance"] - total_spent)

    # DB Update
    asyncio.create_task(
        asyncio.to_thread(
            lambda: SUPABASE.table("users").update({"token_balance": new_bal}).eq("api_key", api_key).execute()
        )
    )

    return {
        "id": f"surg_{secrets.token_hex(4)}",
        "object": "chat.completion",
        "model": "Neo-L1.0",
        "choices": [{
            "message": {"role": "assistant", "content": final_content},
            "finish_reason": "stop"
        }],
        "usage": {
            "total_tokens": total_spent,
            "completion_tokens": ai_raw.usage.completion_tokens
        },
        "engine": engine_id
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
    SUPABASE.table("users").insert({"api_key": new_key, "token_balance": 2000}).execute()
    return {"api_key": new_key, "balance": 2000, "status": "Neo Active"}

@app.get("/")
def health(): return {"status": "online", "mode": "Strict-Surgical"}
