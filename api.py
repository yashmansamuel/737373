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
# 1. Setup & Environment
# -----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-L1.0-Ultra")

SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="Neo L1.0 Ultra-Stable")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Models List - Fixed for Groq Compatibility
MODELS = [
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "llama-3.1-8b-instant"
]

# Deeper System Prompt for Balanced Combo
SYSTEM_PROMPT = """Identity: Neo L1.0 (signaturesi.com). 
Role: Deep-Logic Research Engine.
Task: Provide a high-density 'reasoning' (100-150 tokens) and an exhaustive 'final_answer'.

JSON Rules:
- Return ONLY valid JSON: {"final_answer": "...", "reasoning": "..."}
- Escape all double quotes and newlines within the strings.
- Reasoning must explain the underlying scientific or logical framework.
- Final answer must be detailed, academic, and direct.
"""

# -----------------------------
# 2. Data Models
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

# -----------------------------
# 3. Logic & Helpers
# -----------------------------
def get_neo_knowledge(query: str) -> str:
    try:
        file_path = os.path.join(os.path.dirname(__file__), "knowledge.txt")
        if not os.path.exists(file_path): return ""
        words = [w.lower() for w in query.split() if len(w) > 3]
        matches = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if any(w in line.lower() for w in words):
                    matches.append(line.strip())
                if len(matches) >= 5: break
        return "\n".join(matches)
    except: return ""

async def update_balance_safe(api_key: str, tokens: int, current_bal: int):
    """Accurately deducts real tokens used by Groq."""
    new_bal = max(0, current_bal - tokens)
    await asyncio.to_thread(lambda: SUPABASE.table("users").update({"token_balance": new_bal}).eq("api_key", api_key).execute())
    return new_bal

# -----------------------------
# 4. Core Engine (With Shifting & Delay)
# -----------------------------
@app.post("/v1/chat/completions")
async def chat_engine(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization: raise HTTPException(401, "API Key Required")
    api_key = authorization.replace("Bearer ", "")

    # Fetch User
    user_res = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    if not user_res.data: raise HTTPException(401, "Invalid API Key")
    balance = user_res.data["token_balance"]

    if balance <= 0: raise HTTPException(402, "Insufficient Tokens")

    # Prepare Context & History (Locked to last 2 to save tokens)
    user_query = payload.messages[-1].get("content", "")
    context = get_neo_knowledge(user_query)
    
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context: msgs.append({"role": "system", "content": f"Local Context: {context}"})
    
    # Strictly take only last 2 messages for stable token burn
    msgs.extend(payload.messages[-2:] if len(payload.messages) > 2 else payload.messages)

    # MODEL SHIFTING LOOP WITH 4s DELAY
    for index, model_id in enumerate(MODELS):
        try:
            # If it's not the first model, wait 4 seconds before trying
            if index > 0:
                logger.info(f"Shifting to {model_id}... Sleeping 4s to avoid rate limits.")
                await asyncio.sleep(4)

            response = GROQ.chat.completions.create(
                model=model_id,
                messages=msgs,
                temperature=0.4,
                max_tokens=3500, # Supports large output
                response_format={"type": "json_object"}
            )

            # Accurate Token Retrieval
            usage = response.usage
            total_burned = usage.total_tokens
            content_raw = response.choices[0].message.content

            # Deep JSON Parsing (Anti-Crash)
            try:
                data = json.loads(content_raw)
            except:
                # If JSON fails, manually wrap content to avoid [object Object]
                data = {"final_answer": content_raw, "reasoning": "Direct extraction performed."}

            # Update DB and return
            new_bal = await update_balance_safe(api_key, total_burned, balance)

            return {
                "company": "signaturesi.com",
                "final_answer": data.get("final_answer"),
                "reasoning": data.get("reasoning"),
                "usage": {
                    "total": total_burned,
                    "prompt": usage.prompt_tokens,
                    "completion": usage.completion_tokens
                },
                "balance": new_bal,
                "engine": model_id
            }

        except Exception as e:
            logger.error(f"Engine {model_id} Error: {e}")
            continue

    raise HTTPException(503, "Critical: All Neo engines are currently overloaded.")

# -----------------------------
# 5. Admin Routes
# -----------------------------
@app.post("/v1/user/new-key")
def create_key():
    k = f"sig-{secrets.token_hex(16)}"
    SUPABASE.table("users").insert({"api_key": k, "token_balance": 100000}).execute()
    return {"api_key": k}

@app.get("/health")
def health(): return {"status": "operational", "version": "Neo-L1.0-Ultra"}
