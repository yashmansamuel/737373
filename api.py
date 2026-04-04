import os
import logging
import secrets
import asyncio
import json
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
logger = logging.getLogger("Neo-L1.0-Stable")

# Initialize Clients
SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="Neo L1.0 Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# FIXED MODELS (As per your requirement)
MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant"
]

# BALANCED PROMPT: Deep Reasoning + Detailed Answer
SYSTEM_PROMPT = """Identity: Neo L1.0. Deployment: Jan 1, 2026.
Style: Professional Research & Logic Engine.

Rules:
1. Output MUST be a valid JSON: {"final_answer": "...", "reasoning": "..."}.
2. Reasoning: Provide a 3-4 sentence logical breakdown (100-150 tokens). 
   Explain the 'why' and 'how' behind your conclusion.
3. Final Answer: Provide a high-density, detailed, and complete response.
4. Use provided Local Context strictly. No filler text.
"""

# -----------------------------
# 2. Models & Helpers
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

def get_neo_knowledge(user_query: str) -> str:
    try:
        file_path = os.path.join(os.path.dirname(__file__), "knowledge.txt")
        if not os.path.exists(file_path): return ""
        query_words = [w.lower() for w in user_query.split() if len(w) > 3]
        matches = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if any(word in line.lower() for word in query_words):
                    matches.append(line.strip())
                if len(matches) >= 5: break
        return "\n".join(matches)
    except: return ""

# -----------------------------
# 3. Main Chat Logic
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

    # TOKEN SAVER: Memory limited to last 2 messages
    user_msg = payload.messages[-1].get("content", "")
    local_data = get_neo_knowledge(user_msg)

    final_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if local_data:
        final_messages.append({"role": "system", "content": f"Local Context:\n{local_data}"})
    
    # Last 2 messages only (Input Token Lock)
    final_messages.extend(payload.messages[-2:] if len(payload.messages) > 2 else payload.messages)

    # MODEL FALLBACK LOOP
    for model_name in MODELS:
        try:
            # Using a reliable Groq mapping for your custom names
            # (Note: Groq usually uses IDs like llama-3.3-70b-versatile)
            target_model = model_name if "/" not in model_name else "llama-3.3-70b-versatile"
            
            response = GROQ.chat.completions.create(
                model=target_model,
                messages=final_messages,
                temperature=0.5,
                max_tokens=3000,
                response_format={"type": "json_object"}
            )

            # REAL TOKEN COUNT
            real_usage = response.usage
            tokens_used = real_usage.total_tokens
            
            content = response.choices[0].message.content
            parsed = json.loads(content)

            new_balance = max(0, balance - tokens_used)
            
            # Real-time DB Update
            asyncio.create_task(asyncio.to_thread(
                lambda: SUPABASE.table("users").update({"token_balance": new_balance}).eq("api_key", api_key).execute()
            ))

            return {
                "company": "signaturesi.com",
                "final_answer": parsed.get("final_answer"),
                "reasoning": parsed.get("reasoning"),
                "usage": {
                    "total": tokens_used,
                    "prompt": real_usage.prompt_tokens,
                    "completion": real_usage.completion_tokens
                },
                "balance": new_balance,
                "model": "Neo L1.0"
            }

        except Exception as e:
            logger.error(f"Shifted from {model_name} due to: {e}")
            continue

    raise HTTPException(503, "All engines busy")

# -----------------------------
# 4. Utility Routes
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
