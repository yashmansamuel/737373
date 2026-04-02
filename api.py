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

load_dotenv()

# -----------------------------
# 1. Surgical Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-Surgical-L1")

app = FastAPI(title="Signaturesi Neo L1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 2. Clients & Models (3 Model Limit)
# -----------------------------
SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Strict 3-Model Set for better Guider Control
MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b"
]

# -----------------------------
# 3. Strict Token-Saving Guider (System Prompt)
# -----------------------------
# Is prompt mein humne reasoning aur output dono ko zanjeer pehna di hai.
SYSTEM_PROMPT = (
    "Mode: Neo-Surgical-Saver. Objective: Answer in <150 tokens. "
    "Reasoning: Hidden thinking MUST be under 100 tokens. Step-by-step logic only. "
    "Search Rule: Use 'browser_search' ONLY for live facts (news, prices). "
    "Strictly NO search for coding, logic, or general knowledge. "
    "Formatting: No intros, no citations (【†】), no fillers. "
    "Language: Native adaptive (Roman Urdu/Hindi/English)."
)

# -----------------------------
# 4. Pydantic Models
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

# -----------------------------
# 5. Engine Logic (Reasoning & Search Control)
# -----------------------------
async def call_ai_surgical(messages: List[dict]) -> Tuple[object, str]:
    # 🔥 TOKEN SAVER: History Pruning (Only System + Last Msg)
    optimized_context = [{"role": "system", "content": SYSTEM_PROMPT}, messages[-1]]
    
    # Check if Search is actually needed to save Quota
    query_text = messages[-1]['content'].lower()
    live_keywords = ["price", "today", "news", "latest", "weather", "vs", "match"]
    needs_search = any(word in query_text for word in live_keywords)

    for model in MODELS:
        try:
            kwargs = {
                "model": model,
                "messages": optimized_context,
                "temperature": 0.2,            # Low temp = Less rambling = Fewer tokens
                "max_completion_tokens": 400,   # Strict cap for output
                "stream": False,
            }
            
            if "gpt-oss" in model:
                # ❌ ELIMINATE REASONING BILL: Setting to None saves thousands of tokens
                kwargs["reasoning_effort"] = None 
                
                if needs_search:
                    kwargs["tools"] = [{"type": "browser_search"}]
                    # Note: Prompt handles the 3-5 page limit
            
            completion = GROQ.chat.completions.create(**kwargs)
            return completion, model
        except Exception as e:
            logger.error(f"Fallback: {model} failed: {e}")
            continue
    raise HTTPException(503, "All surgical engines busy.")

# -----------------------------
# 6. Main Endpoint
# -----------------------------
@app.post("/v1/chat/completions")
async def neo_chat_proxy(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid Key")
    
    api_key = authorization.replace("Bearer ", "")
    user = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    
    if not user.data or user.data["token_balance"] <= 0:
        raise HTTPException(402, "Insufficient Balance")

    # Call Surgical Engine
    ai_res, used_model = await call_ai_surgical(payload.messages)

    # Content Cleaning (Final Citation Killer)
    raw_content = ai_res.choices[0].message.content or ""
    final_answer = re.sub(r'【.*?】|\[\d+\]|†\w+', '', raw_content).strip()

    # Billing (Based on actual usage)
    tokens_spent = ai_res.usage.total_tokens
    new_bal = max(0, user.data["token_balance"] - tokens_spent)

    # Async Update
    asyncio.create_task(
        asyncio.to_thread(
            lambda: SUPABASE.table("users").update({"token_balance": new_bal}).eq("api_key", api_key).execute()
        )
    )

    return {
        "id": f"neo_surg_{secrets.token_hex(4)}",
        "object": "chat.completion",
        "model": "Neo-L1.0",
        "choices": [{"message": {"role": "assistant", "content": final_answer}, "finish_reason": "stop"}],
        "usage": ai_res.usage
    }

@app.get("/")
def health(): return {"status": "online", "mode": "Ultra-Surgical", "models": 3}
