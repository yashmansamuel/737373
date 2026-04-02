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
logger = logging.getLogger("Neo-L1.0-Shield")

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
# 3. GPT-Style Shield Settings
# -----------------------------
MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-safeguard-20b", "openai/gpt-oss-120b"]

# Updated System Prompt with Search Limit & Native Fluidity
SYSTEM_PROMPT = (
    "Act: Neo-Adaptive Intelligence. Task: Provide expert, high-quality responses. "
    "Search Rule: Use 'browser_search' ONLY for the top 3-5 most relevant pages. "
    "DO NOT perform deep crawling or multi-page scraping to save quota. "
    "Language: Respond natively in the language used by the user. "
    "Rules: Strictly NO internal reasoning text. Strictly NO search citations or source tags like [1] or 【†】. "
    "Completeness: Never leave a list item or sentence unfinished. Ensure full closure. "
    "Constraint: High information density. Focus only on the current query to save tokens."
)

# -----------------------------
# 4. Helper: Final Artifact Filter
# -----------------------------
def clean_neo_output(text: str) -> str:
    """Removes all search citations, source tags, and technical artifacts."""
    if not text: return ""
    # Remove citations like 【1†L168-L178】, [1], [2], etc.
    text = re.sub(r'【.*?】', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'†\w+', '', text)
    # Remove stray headers/markdown artifacts
    text = re.sub(r'(Source|Citations|Reference|Read more):.*', '', text, flags=re.IGNORECASE)
    return text.strip()

# -----------------------------
# 5. Data Models
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

# -----------------------------
# 6. Token & Quota Saver Engine
# -----------------------------
async def call_neo_engine(messages: List[dict]) -> Tuple[object, str]:
    # 🔥 TOKEN SAVER: Only send System Prompt + last message
    if len(messages) > 0:
        optimized_context = [{"role": "system", "content": SYSTEM_PROMPT}, messages[-1]]
    else:
        raise HTTPException(400, "No messages provided.")

    for model_name in MODELS:
        try:
            params = {
                "model": model_name,
                "messages": optimized_context,
                "temperature": 0.5,           
                "max_completion_tokens": 500,  
                "stream": False,
            }
            
            # Quota Protection: Search is enabled but restricted via prompt
            if "gpt-oss" in model_name:
                params["reasoning_effort"] = None 
                params["tools"] = [{"type": "browser_search"}]

            completion = GROQ_CLIENT.chat.completions.create(**params)
            return completion, model_name

        except Exception as e:
            err = str(e).lower()
            logger.warning(f"Engine {model_name} failed: {err}")
            if "rate_limit" in err:
                await asyncio.sleep(0.7) 
            continue

    raise HTTPException(503, "Engines overloaded. Try again later.")

# -----------------------------
# 7. Main Proxy Endpoint
# -----------------------------
@app.post("/v1/chat/completions")
async def neo_chat_proxy(payload: ChatRequest, authorization: str = Header(None)):
    # Auth & Balance Check
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid Authorization.")
    
    api_key = authorization.replace("Bearer ", "")
    user = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    
    if not user.data:
        raise HTTPException(401, "Key not found.")
    
    current_bal = user.data["token_balance"]
    if current_bal <= 0:
        raise HTTPException(402, "Balance exhausted.")

    # Execute AI (Limited Search Mode)
    ai_raw, engine_id = await call_neo_engine(payload.messages)

    # Clean Output
    raw_content = ai_raw.choices[0].message.content or ""
    final_content = clean_neo_output(raw_content)

    # Billing Update
    total_spent = ai_raw.usage.total_tokens
    new_bal = max(0, current_bal - total_spent)

    # Background DB Update
    asyncio.create_task(
        asyncio.to_thread(
            lambda: SUPABASE.table("users").update({"token_balance": new_bal}).eq("api_key", api_key).execute()
        )
    )

    return {
        "id": f"neo_shield_{secrets.token_hex(4)}",
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
    new_key = f"sig-neo-{secrets.token_urlsafe(16)}"
    country = request.headers.get("cf-ipcountry", "Global")
    SUPABASE.table("users").insert({"api_key": new_key, "token_balance": 2000, "country": country}).execute()
    return {"api_key": new_key, "balance": 2000, "status": "Neo Active"}

@app.get("/")
def health(): return {"status": "online", "shield": "Active", "search_depth": "Limited (3-5 pages)"}
