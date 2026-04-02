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
logger = logging.getLogger("Neo-L1.0-Surgical")

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
# 3. Neo Surgical Settings
# -----------------------------
MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-safeguard-20b", "openai/gpt-oss-120b"]

# Optimized System Prompt for Token Efficiency & Accuracy
SYSTEM_PROMPT = (
    "Act: Neo-Surgical Intelligence. Goal: Maximum Accuracy with Minimum Tokens. "
    "Reasoning Rule: Internal hidden reasoning MUST NOT exceed 200 tokens. "
    "Step-by-Step Reasoning: 1. Identify intent. 2. Fetch top 3 results ONLY. 3. Filter noise. 4. Synthesize direct answer. "
    "Search Rule: Use 'browser_search' only for top 3-5 relevant pages. No deep crawling. "
    "Output Rule: No fillers, no citations (【†】, [1]), no 'Searching...' text. "
    "Language: Respond natively in user's language (Roman Urdu/Hindi/English). "
    "Constraint: Answer must be concise (max 4-5 sentences or 6 bullets). "
    "Completeness: Ensure every sentence and table is 100% finished."
)

# -----------------------------
# 4. Helper: Artifact Killer
# -----------------------------
def clean_neo_output(text: str) -> str:
    """Removes search citations, source tags, and technical artifacts."""
    if not text: return ""
    text = re.sub(r'【.*?】', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'†\w+', '', text)
    text = re.sub(r'(Source|Citations|Reference|Read more):.*', '', text, flags=re.IGNORECASE)
    return text.strip()

# -----------------------------
# 5. Data Models
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

# -----------------------------
# 6. Token-Saver Engine Logic
# -----------------------------
async def call_neo_engine(messages: List[dict]) -> Tuple[object, str]:
    # 🔥 TOKEN SAVER: Only System Prompt + Current User Message
    if len(messages) > 0:
        optimized_context = [{"role": "system", "content": SYSTEM_PROMPT}, messages[-1]]
    else:
        raise HTTPException(400, "Empty message list.")

    for model_name in MODELS:
        try:
            params = {
                "model": model_name,
                "messages": optimized_context,
                "temperature": 0.3,           # Lower temp for higher accuracy/less tokens
                "max_completion_tokens": 600,  # Buffer for tables/detailed native text
                "stream": False,
            }
            
            if "gpt-oss" in model_name:
                params["reasoning_effort"] = None # Avoids expensive reasoning tokens
                params["tools"] = [{"type": "browser_search"}]

            completion = GROQ_CLIENT.chat.completions.create(**params)
            return completion, model_name

        except Exception as e:
            err = str(e).lower()
            logger.warning(f"Engine {model_name} failed: {err}")
            if "rate_limit" in err: await asyncio.sleep(0.8)
            continue

    raise HTTPException(503, "All engines busy. Please try again.")

# -----------------------------
# 7. Optimized Proxy Endpoint
# -----------------------------
@app.post("/v1/chat/completions")
async def neo_chat_proxy(payload: ChatRequest, authorization: str = Header(None)):
    # 1. Auth Check
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid token.")
    
    api_key = authorization.replace("Bearer ", "")
    user = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    
    if not user.data or user.data["token_balance"] <= 0:
        raise HTTPException(402, "Insufficient balance.")

    # 2. AI Execution
    ai_raw, engine_id = await call_neo_engine(payload.messages)

    # 3. Polish Output
    raw_text = ai_raw.choices[0].message.content or ""
    final_text = clean_neo_output(raw_text)

    # 4. Accurate Billing
    total_spent = ai_raw.usage.total_tokens
    new_bal = max(0, user.data["token_balance"] - total_spent)

    # Background DB Update
    asyncio.create_task(
        asyncio.to_thread(
            lambda: SUPABASE.table("users").update({"token_balance": new_bal}).eq("api_key", api_key).execute()
        )
    )

    # 5. Standard OpenAI Response
    return {
        "id": f"neo_surg_{secrets.token_hex(4)}",
        "object": "chat.completion",
        "model": "Neo-L1.0",
        "choices": [{
            "message": {"role": "assistant", "content": final_content},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": ai_raw.usage.prompt_tokens,
            "completion_tokens": ai_raw.usage.completion_tokens,
            "total_tokens": total_spent
        }
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
    new_key = f"sig-neo-{secrets.token_urlsafe(18)}"
    country = request.headers.get("cf-ipcountry", "Unknown")
    SUPABASE.table("users").insert({"api_key": new_key, "token_balance": 2000, "country": country}).execute()
    return {"api_key": new_key, "balance": 2000, "status": "Neo Surgical Active"}

@app.get("/")
def health(): return {"status": "online", "mode": "Surgical/Token-Saver", "search": "Limited"}
