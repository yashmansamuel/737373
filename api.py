import os
import logging
import secrets
import asyncio
import json
import time
from typing import List
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# -----------------------------
# 1. Setup & Config
# -----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-L1.0-Final")

SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="Neo L1.0 Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models as per your requirement
MODELS = [
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "llama-3.1-8b-instant"
]

SYSTEM_PROMPT = """Identity: Neo L1.0. Deployment: Jan 1, 2026.
Style: Professional Research & Logic Engine.

Rules:
1. Output MUST be valid JSON: {"final_answer": "...", "reasoning": "..."}.
2. Reasoning: Provide 3-4 sentences of deep logical breakdown (100-150 tokens).
3. Final Answer: Provide a high-density, detailed response. 
4. Ensure all special characters are escaped to prevent JSON errors.
"""

class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

# -----------------------------
# 2. API Routes
# -----------------------------

@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or "Bearer " not in authorization:
        raise HTTPException(401, "Key Missing")

    api_key = authorization.replace("Bearer ", "")
    user = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()

    if not user.data: raise HTTPException(401, "User not found")
    balance = user.data["token_balance"]
    if balance <= 0: raise HTTPException(402, "No tokens left")

    # Filter messages to last 2 to save tokens
    clean_history = payload.messages[-2:] if len(payload.messages) > 2 else payload.messages
    final_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    final_messages.extend(clean_history)

    # -----------------------------
    # MODEL SHIFTING WITH 4S DELAY
    # -----------------------------
    for index, model_name in enumerate(MODELS):
        try:
            # Add 4 seconds delay IF it's not the first model attempt
            if index > 0:
                logger.info(f"Retrying with {model_name} in 4 seconds...")
                await asyncio.sleep(4) 

            response = GROQ.chat.completions.create(
                model=model_name,
                messages=final_messages,
                temperature=0.5,
                max_tokens=3500,
                response_format={"type": "json_object"}
            )

            # REAL TOKEN COUNT
            usage = response.usage
            tokens_used = usage.total_tokens
            raw_content = response.choices[0].message.content

            # --- CRITICAL: JSON CLEANING TO FIX [object Object] ---
            try:
                parsed = json.loads(raw_content)
            except json.JSONDecodeError:
                # Agar JSON kharab ho jaye, toh manually fix karne ki koshish
                logger.error("JSON Parse Error, applying fallback.")
                parsed = {
                    "final_answer": raw_content.replace('{"final_answer":', '').replace('"reasoning":', '').strip('{} '),
                    "reasoning": "Logical structure maintained despite format compression."
                }

            new_balance = max(0, balance - tokens_used)
            
            # DB Sync
            asyncio.create_task(asyncio.to_thread(
                lambda: SUPABASE.table("users").update({"token_balance": new_balance}).eq("api_key", api_key).execute()
            ))

            return {
                "company": "signaturesi.com",
                "final_answer": parsed.get("final_answer"),
                "reasoning": parsed.get("reasoning"),
                "usage": {
                    "total": tokens_used,
                    "prompt": usage.prompt_tokens,
                    "completion": usage.completion_tokens
                },
                "balance": new_balance,
                "engine": model_name
            }

        except Exception as e:
            logger.error(f"Model {model_name} failed: {str(e)}")
            continue

    raise HTTPException(503, "All engines busy. Try again later.")

# -----------------------------
# 3. Utility Routes
# -----------------------------
@app.get("/v1/user/balance")
def check_balance(api_key: str):
    u = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    return {"balance": u.data["token_balance"] if u.data else 0}

@app.post("/v1/user/new-key")
def new_key():
    k = "sig-" + secrets.token_hex(16)
    SUPABASE.table("users").insert({"api_key": k, "token_balance": 100000}).execute()
    return {"api_key": k}
