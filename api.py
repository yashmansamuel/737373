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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-Surgical-v3")

app = FastAPI(title="Signaturesi Neo L1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clients
SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Final 3 Models only
MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b"
]

SYSTEM_PROMPT = (
    "Mode: Neo-Surgical. Rule 1: No hidden reasoning (>100 tokens). "
    "Rule 2: No citations/source tags. Rule 3: Native adaptive language. "
    "Rule 4: Max 5 bullets or 3 sentences. Rule 5: Tool search for LIVE facts only."
)

class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

def clean_output(text: str) -> str:
    if not text: return ""
    text = re.sub(r'【.*?】|\[\d+\]|†\w+', '', text)
    return text.strip()

async def call_ai_surgical(messages: List[dict]) -> Tuple[object, str]:
    # Shield 1: Context Pruning
    optimized_context = [{"role": "system", "content": SYSTEM_PROMPT}, messages[-1]]
    
    # Shield 3: Tool Gating Logic
    query = messages[-1]['content'].lower()
    search_needed = any(k in query for k in ["price", "today", "latest", "news", "vs", "weather", "gold"])

    for i, model in enumerate(MODELS):
        try:
            # Shield 4: Anti-Spam 7s Jitter
            if i > 0:
                logger.warning(f"Spam Shield: Waiting 7s before switching to {model}")
                await asyncio.sleep(7)

            kwargs = {
                "model": model,
                "messages": optimized_context,
                "temperature": 0.2, # Accuracy focus
                "max_completion_tokens": 600,
                "stream": False,
            }
            
            # Shield 2: Reasoning Kill
            if "gpt-oss" in model:
                kwargs["reasoning_effort"] = None 
                if search_needed:
                    kwargs["tools"] = [{"type": "browser_search"}]

            completion = GROQ_CLIENT.chat.completions.create(**kwargs)
            return completion, model

        except Exception as e:
            logger.error(f"Engine {model} Error: {e}")
            continue

    raise HTTPException(503, "All engines exhausted. Please wait 30 seconds.")

@app.post("/v1/chat/completions")
async def neo_chat_proxy(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Auth Error")
    
    api_key = authorization.replace("Bearer ", "")
    user = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    
    if not user.data or user.data["token_balance"] <= 0:
        raise HTTPException(402, "💰 Balance: 0 tokens. Recharge required.")

    # Execute Surgical AI
    ai_res, engine_id = await call_ai_surgical(payload.messages)
    
    # Clean & Bill
    final_text = clean_output(ai_res.choices[0].message.content)
    tokens_spent = ai_res.usage.total_tokens
    new_bal = max(0, user.data["token_balance"] - tokens_spent)

    # Async Update
    asyncio.create_task(
        asyncio.to_thread(
            lambda: SUPABASE.table("users").update({"token_balance": new_bal}).eq("api_key", api_key).execute()
        )
    )

    return {
        "id": f"neo_{secrets.token_hex(4)}",
        "object": "chat.completion",
        "model": "Neo-L1.0",
        "choices": [{"message": {"role": "assistant", "content": final_text}, "finish_reason": "stop"}],
        "usage": {"total_tokens": tokens_spent},
        "balance_remaining": new_bal,
        "engine": engine_id
    }

@app.get("/")
def health(): return {"status": "online", "engine": "Neo-Surgical-v3"}
