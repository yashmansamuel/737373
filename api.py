import os
import logging
import secrets
import re
import asyncio
import time
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
logger = logging.getLogger("Neo-Surgical-v2")

app = FastAPI(title="Signaturesi Neo L1.0 - Token Saver Edition")

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
# 3. Model & Token Settings
# -----------------------------
# Strictly limited to 3 models as requested
MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b"
]

# Strict Token Saving Guider Prompt
SYSTEM_PROMPT = (
    "Act: Neo-Surgical Intelligence. Goal: Strict Token Saving. "
    "Guider Rules: 1. Internal reasoning MUST be under 150 tokens. 2. Never use search for logic/coding. "
    "3. Only use 'browser_search' for live facts (top 2-3 pages). 4. Eliminate all fluff/intros. "
    "Language: Respond natively in user's language. "
    "Output: Concise, high-density, and complete. Limit total response to essential info only."
)

# -----------------------------
# 4. Helper Functions
# -----------------------------
def clean_neo_output(text: str) -> str:
    """Removes citations and artifacts to keep output clean."""
    if not text: return ""
    text = re.sub(r'【.*?】', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'†\w+', '', text)
    return text.strip()

class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

# -----------------------------
# 5. Surgical Engine with Anti-Spam Delay
# -----------------------------
async def call_neo_surgical_engine(messages: List[dict]) -> Tuple[object, str]:
    # TOKEN SAVER: Pruning history - Only System + Current Message
    optimized_context = [{"role": "system", "content": SYSTEM_PROMPT}, messages[-1]]
    
    # Identify if search is actually needed to save quota
    query_text = messages[-1]['content'].lower()
    needs_search = any(x in query_text for x in ["latest", "price", "news", "today", "vs", "weather"])

    for i, model_name in enumerate(MODELS):
        try:
            # 7-Second Anti-Spam Delay before switching to next model (except first attempt)
            if i > 0:
                logger.info(f"Anti-Spam: Delaying 7 seconds before switching to {model_name}...")
                await asyncio.sleep(7)

            params = {
                "model": model_name,
                "messages": optimized_context,
                "temperature": 0.2,            # Low temp = less token wastage
                "max_completion_tokens": 500,   # Safe buffer for complete answers
                "stream": False,
            }
            
            if "gpt-oss" in model_name:
                params["reasoning_effort"] = None # Critical: Stops hidden 10k token burn
                if needs_search:
                    params["tools"] = [{"type": "browser_search"}]

            completion = GROQ_CLIENT.chat.completions.create(**params)
            return completion, model_name

        except Exception as e:
            logger.error(f"Model {model_name} failed: {e}")
            continue

    raise HTTPException(503, "All surgical engines exhausted.")

# -----------------------------
# 6. Main Proxy Endpoint
# -----------------------------
@app.post("/v1/chat/completions")
async def neo_chat_proxy(payload: ChatRequest, authorization: str = Header(None)):
    # 1. Security & Balance Auth
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid Auth")
    
    api_key = authorization.replace("Bearer ", "")
    user = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    
    if not user.data:
        raise HTTPException(401, "API Key not found")
    
    current_bal = user.data.get("token_balance", 0)
    if current_bal <= 0:
        raise HTTPException(402, "💰 Balance: 0 tokens. Please recharge.")

    # 2. AI Execution (Surgical Mode)
    ai_res, engine_id = await call_neo_surgical_engine(payload.messages)

    # 3. Output Refinement
    raw_content = ai_res.choices[0].message.content or ""
    final_content = clean_neo_output(raw_content)

    # 4. Token Accounting
    tokens_spent = ai_res.usage.total_tokens
    new_bal = max(0, current_bal - tokens_spent)

    # Background Update (Prevents "undefined" token error on UI)
    asyncio.create_task(
        asyncio.to_thread(
            lambda: SUPABASE.table("users").update({"token_balance": new_bal}).eq("api_key", api_key).execute()
        )
    )

    # 5. Standard Response
    return {
        "id": f"neo_surg_{secrets.token_hex(4)}",
        "object": "chat.completion",
        "model": "Neo-L1.0",
        "choices": [{
            "message": {"role": "assistant", "content": final_content},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": ai_res.usage.prompt_tokens,
            "completion_tokens": ai_res.usage.completion_tokens,
            "total_tokens": tokens_spent
        },
        "balance_remaining": new_bal
    }

# -----------------------------
# 7. Utility Endpoints
# -----------------------------
@app.get("/v1/user/balance")
def get_balance(api_key: str):
    res = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    if not res.data: raise HTTPException(404, "Invalid Key")
    return {"api_key": api_key, "balance": res.data["token_balance"]}

@app.post("/v1/user/new-key")
async def generate_key(request: Request):
    new_key = f"sig-neo-{secrets.token_urlsafe(16)}"
    country = request.headers.get("cf-ipcountry", "Unknown")
    SUPABASE.table("users").insert({"api_key": new_key, "token_balance": 2000, "country": country}).execute()
    return {"api_key": new_key, "balance": 2000, "status": "Neo Active"}

@app.get("/")
def health(): return {"status": "online", "mode": "Surgical-v2", "delay": "7s"}
