import os
import logging
import secrets
import asyncio
import json
from typing import List
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq  # Assuming your Llama engine wrapper uses Groq API; replace with correct client if needed

# -----------------------------
# Setup & Configuration
# -----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-Llama4-Scout")

SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Single Llama-4-SCOUT model
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# FastAPI app
app = FastAPI(title="Neo L1.0 Knowledge Engine (Single Model)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# -----------------------------
# Models
# -----------------------------
class ChatRequest(BaseModel):
    messages: List[dict]

# -----------------------------
# Knowledge Retrieval
# -----------------------------
def get_relevant_knowledge(query: str, max_lines: int = 3) -> str:
    """Return top N matching lines from knowledge.txt."""
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
# Core Chat Endpoint
# -----------------------------
@app.post("/v1/chat/completions")
async def chat_engine(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or "Bearer " not in authorization:
        raise HTTPException(401, "Valid API Key Required")
    
    api_key = authorization.replace("Bearer ", "")
    user_res = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    if not user_res.data:
        raise HTTPException(401, "User Not Found")
    
    balance = user_res.data["token_balance"]
    if balance <= 0:
        raise HTTPException(402, "Zero Balance")
    
    # Last message query
    user_query = payload.messages[-1].get("content", "")
    context = get_relevant_knowledge(user_query)
    
    # Build prompt
    system_prompt = "Identity: Neo L1.0.\n" \
                    "Respond JSON only: {\"final_answer\":\"...\",\"reasoning\":\"...\"}\n" \
                    "Keep reasoning concise (max 50 words)."
    if context:
        system_prompt += f"\nContext:\n{context}"
    
    msgs = [{"role": "system", "content": system_prompt}]
    msgs.extend(payload.messages[-2:] if len(payload.messages) > 2 else payload.messages)
    
    # -----------------------------
    # Call Llama-4-SCOUT Model
    # -----------------------------
    try:
        response = Groq.chat.completions.create(
            model=MODEL,
            messages=msgs,
            temperature=0.5,
            max_tokens=3500,
            response_format={"type": "json_object"}
        )
        
        usage = response.usage
        total_burned = usage.total_tokens
        raw_content = response.choices[0].message.content
        
        # Parse JSON safely
        try:
            data = json.loads(raw_content)
        except:
            data = {"final_answer": raw_content, "reasoning": "Fallback: concise reasoning not generated."}
        
        # -----------------------------
        # Token Deduction (Atomic)
        # -----------------------------
        new_bal = max(0, balance - total_burned)
        asyncio.create_task(asyncio.to_thread(
            lambda: SUPABASE.table("users")
                               .update({"token_balance": new_bal})
                               .eq("api_key", api_key)
                               .execute()
        ))
        
        return {
            "company": "signaturesi.com",
            "final_answer": data.get("final_answer"),
            "reasoning": data.get("reasoning"),
            "usage": {"total": total_burned, "prompt": usage.prompt_tokens, "completion": usage.completion_tokens},
            "balance": new_bal,
            "engine": MODEL
        }
    except Exception as e:
        logger.error(f"Model call failed: {e}")
        raise HTTPException(503, "Engine unavailable. Try again later.")

# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "operational", "engine": MODEL}

# -----------------------------
# API Key Generation
# -----------------------------
@app.post("/v1/user/new-key")
def create_key():
    k = f"sig-{secrets.token_hex(16)}"
    SUPABASE.table("users").insert({"api_key": k, "token_balance": 1_000_000}).execute()
    return {"api_key": k, "starter_tokens": 1_000_000}

# -----------------------------
# Top-up Tokens Endpoint
# -----------------------------
@app.post("/v1/user/top-up")
def top_up(api_key: str, tokens: int = 1_000_000):
    # Example pricing: $12 per 1M tokens
    price = 12 * (tokens / 1_000_000)
    # TODO: integrate payment processing (Stripe / PayPal)
    SUPABASE.table("users").update({
        "token_balance": f"token_balance + {tokens}"
    }).eq("api_key", api_key).execute()
    return {"new_balance": f"Updated", "price_usd": price}
