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
# 1. Setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-Surgical-v3")
app = FastAPI(title="Signaturesi Neo L1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# 2. Optimized Config
# -----------------------------
MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b"
]

SYSTEM_PROMPT = (
    "Act: Neo-Surgical. Goal: Accurate, Low-Token. "
    "Rules: Max 150 words. Use 'browser_search' for top 2 results only. "
    "No reasoning, no citations, no fluff. Native language response."
)

# -----------------------------
# 3. Surgical AI Engine
# -----------------------------
async def call_neo_surgical_engine(messages: List[dict]) -> Tuple[object, str]:
    # 🔥 TOKEN SAVER: System + ONLY Last Message
    optimized_context = [{"role": "system", "content": SYSTEM_PROMPT}, messages[-1]]
    
    query_text = messages[-1]['content'].lower()
    # Sirf zaroori queries par search chalayein taaki 8k limit bache
    needs_search = any(x in query_text for x in ["latest", "price", "news", "today", "vs"])

    for i, model_name in enumerate(MODELS):
        try:
            # Har switch se pehle delay taaki TPM limit refresh ho jaye
            if i > 0:
                await asyncio.sleep(5) 

            params = {
                "model": model_name,
                "messages": optimized_context,
                "temperature": 0.2,
                "max_completion_tokens": 400, # Chota response = No Rate Limit
                "stream": False,
            }
            
            if "gpt-oss" in model_name:
                params["reasoning_effort"] = None
                if needs_search:
                    # Search ko restrict kar rahe hain
                    params["tools"] = [{"type": "browser_search"}]

            completion = GROQ_CLIENT.chat.completions.create(**params)
            return completion, model_name

        except Exception as e:
            err_msg = str(e).lower()
            logger.error(f"Model {model_name} Error: {err_msg}")
            
            # Agar Rate Limit hit ho jaye toh zyada wait karein
            if "rate_limit" in err_msg or "429" in err_msg:
                await asyncio.sleep(10)
            continue

    raise HTTPException(503, "All engines busy or Rate Limit reached (8k TPM). Try in 1 minute.")

# -----------------------------
# 4. Proxy Endpoint
# -----------------------------
@app.post("/v1/chat/completions")
async def neo_chat_proxy(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Auth Required")
    
    api_key = authorization.replace("Bearer ", "")
    user = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    
    if not user.data:
        raise HTTPException(401, "Invalid Key")
    
    current_bal = user.data.get("token_balance", 0)
    if current_bal <= 0:
        raise HTTPException(402, "💰 Balance Exhausted")

    # Call Engine
    ai_res, engine_id = await call_neo_surgical_engine(payload.messages)

    # Content & Billing
    final_content = re.sub(r'【.*?】|\[\d+\]|†\w+', '', ai_res.choices[0].message.content or "").strip()
    tokens_spent = ai_res.usage.total_tokens
    new_bal = max(0, current_bal - tokens_spent)

    # Update DB
    asyncio.create_task(
        asyncio.to_thread(
            lambda: SUPABASE.table("users").update({"token_balance": new_bal}).eq("api_key", api_key).execute()
        )
    )

    return {
        "id": f"neo_{secrets.token_hex(4)}",
        "object": "chat.completion",
        "model": "Neo-L1.0",
        "choices": [{"message": {"role": "assistant", "content": final_content}}],
        "usage": ai_res.usage,
        "balance": new_bal
    }

# FastAPI model setup at the end to avoid reference errors
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]
