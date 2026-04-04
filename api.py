import os
import logging
import secrets
import asyncio
import json
from typing import List
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq  # if your Llama model via Groq; else adapt to your Llama API

# -----------------------------
# Load environment & logging
# -----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-Llama-Engine")

# -----------------------------
# Supabase client
# -----------------------------
SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# -----------------------------
# Model setup
# -----------------------------
MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"  # single engine
ENGINE = Groq(api_key=os.getenv("GROQ_API_KEY"))  # replace if using your Llama API wrapper

# -----------------------------
# FastAPI App & CORS
# -----------------------------
app = FastAPI(title="Neo L1.0 Llama Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# -----------------------------
# System Prompt (concise reasoning)
# -----------------------------
SYSTEM_PROMPT = """Identity: Neo L1.0.
Role: High-Density Reasoning Engine.
STRICT OUTPUT RULES:
1. Return ONLY valid JSON: {"final_answer": "...", "reasoning": "..."}.
2. Keep reasoning concise, ≤50 words.
3. Escape all quotes and newlines properly."""

# -----------------------------
# Models & Schemas
# -----------------------------
class ChatRequest(BaseModel):
    messages: List[dict]

# -----------------------------
# Knowledge Retrieval (RAG)
# -----------------------------
def get_relevant_knowledge(query: str, max_lines: int = 3) -> str:
    """Return top relevant lines from knowledge.txt for the query."""
    try:
        file_path = os.path.join(os.path.dirname(__file__), "knowledge.txt")
        if not os.path.exists(file_path):
            return ""
        words = [w.lower() for w in query.split() if len(w) > 3]
        matches = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if any(w in line.lower() for w in words):
                    matches.append(line.strip())
                if len(matches) >= max_lines:
                    break
        return "\n".join(matches)
    except Exception as e:
        logger.error(f"Knowledge retrieval failed: {e}")
        return ""

# -----------------------------
# Token Utilities
# -----------------------------
async def deduct_tokens(api_key: str, tokens_used: int) -> int:
    """Atomically deduct tokens and return new balance."""
    def db_update():
        res = SUPABASE.table("users").update({
            "token_balance": f"token_balance - {tokens_used}"
        }).eq("api_key", api_key).gte("token_balance", tokens_used).execute()
        return res.data
    result = await asyncio.to_thread(db_update)
    if not result:
        raise HTTPException(402, "Token limit reached. Please top-up.")
    new_balance = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute().data["token_balance"]
    return new_balance

# -----------------------------
# Chat Engine Endpoint
# -----------------------------
@app.post("/v1/chat/completions")
async def chat_engine(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or "Bearer " not in authorization:
        raise HTTPException(401, "Valid API Key Required")
    
    api_key = authorization.replace("Bearer ", "")
    
    # Check user exists and get balance
    user_res = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    if not user_res.data:
        raise HTTPException(401, "User not found")
    balance = user_res.data["token_balance"]
    if balance <= 0:
        raise HTTPException(402, "Zero balance")
    
    # Prepare messages
    user_query = payload.messages[-1].get("content", "")
    context = get_relevant_knowledge(user_query)
    
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context:
        msgs.append({"role": "system", "content": f"Context:\n{context}"})
    msgs.extend(payload.messages[-2:] if len(payload.messages) > 2 else payload.messages)
    
    # Call single Llama model
    try:
        response = ENGINE.chat.completions.create(
            model=MODEL_ID,
            messages=msgs,
            temperature=0.5,
            max_tokens=2000,  # limit to save tokens
            response_format={"type": "json_object"}
        )
        
        usage = response.usage
        total_burned = usage.total_tokens
        raw_content = response.choices[0].message.content
        
        # Safe JSON parse
        try:
            data = json.loads(raw_content)
        except:
            data = {"final_answer": raw_content, "reasoning": "Fallback: JSON parse failed."}
        
        # Deduct tokens
        new_bal = await deduct_tokens(api_key, total_burned)
        
        # Return structured response
        return {
            "company": "signaturesi.com",
            "final_answer": data.get("final_answer"),
            "reasoning": data.get("reasoning"),
            "usage": {"total": total_burned, "prompt": usage.prompt_tokens, "completion": usage.completion_tokens},
            "balance": new_bal,
            "engine": MODEL_ID
        }
    
    except Exception as e:
        logger.error(f"Engine {MODEL_ID} failed: {e}")
        raise HTTPException(503, "Engine error or overloaded")

# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "operational", "engine": "Neo L1.0 Llama-4-Scout"}

# -----------------------------
# User Key / Plan Endpoints
# -----------------------------
@app.post("/v1/user/new-key")
def create_key():
    k = f"sig-{secrets.token_hex(16)}"
    SUPABASE.table("users").insert({"api_key": k, "token_balance": 1_000_000}).execute()  # Starter: 1M tokens
    return {"api_key": k, "plan": "Starter", "token_balance": 1_000_000}

@app.post("/v1/user/top-up")
def top_up(api_key: str, tokens: int = 1_000_000):
    price_per_1M = 12  # $12 per 1M tokens
    price = (tokens / 1_000_000) * price_per_1M
    
    # Payment processing placeholder (Stripe/PayPal)
    # Assume success
    SUPABASE.table("users").update({
        "token_balance": f"token_balance + {tokens}"
    }).eq("api_key", api_key).execute()
    
    new_balance = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute().data["token_balance"]
    return {"new_balance": new_balance, "tokens_added": tokens, "price_usd": price}
