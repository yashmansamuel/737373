import os
import secrets
import asyncio
import json
from typing import List
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- OPTIMIZED SYSTEM PROMPT ---
# Isme humne tokens ki strict instruction di hai
SYSTEM_PROMPT = (
    "You are Neo L1.0. Role: Technical Researcher. "
    "CONSTRAINT: Your reasoning must be ultra-short (max 50 tokens). "
    "Your final_answer should be detailed and high-quality. "
    "Output ONLY JSON: {\"final_answer\": \"...\", \"reasoning\": \"...\"}. "
    "Total response must stay under 3000 tokens."
)

MODELS = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]

class ChatRequest(BaseModel):
    messages: List[dict]

@app.post("/v1/chat/completions")
async def chat_engine(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization: raise HTTPException(401, "Key Missing")
    
    api_key = authorization.replace("Bearer ", "")
    
    # 1. Fetch Real User Balance
    user_data = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    if not user_data.data: raise HTTPException(401, "Invalid Key")
    
    balance = user_data.data["token_balance"]
    if balance <= 0: raise HTTPException(402, "Balance Exhausted")

    # 2. Token Saving: Filter History (Only keep last 2 messages)
    # Isse Input Burn 90% kam ho jayega
    clean_history = payload.messages[-2:] if len(payload.messages) > 2 else payload.messages
    
    final_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    final_messages.extend(clean_history)

    for model_id in MODELS:
        try:
            # 3. Groq API Call
            response = GROQ.chat.completions.create(
                model=model_id,
                messages=final_messages,
                temperature=0.4,
                max_tokens=3000, # Full response limit
                response_format={"type": "json_object"}
            )
            
            # --- REAL TOKEN COUNT FROM GROQ ---
            real_usage = response.usage
            total_burned = real_usage.total_tokens # Exactly what Groq charged
            prompt_tokens = real_usage.prompt_tokens
            completion_tokens = real_usage.completion_tokens
            
            content = response.choices[0].message.content
            parsed_data = json.loads(content)

            # 4. Accurate DB Update
            new_balance = max(0, balance - total_burned)
            SUPABASE.table("users").update({"token_balance": new_balance}).eq("api_key", api_key).execute()

            return {
                "company": "signaturesi.com",
                "final_answer": parsed_data.get("final_answer"),
                "reasoning": parsed_data.get("reasoning"),
                "usage": {
                    "total": total_burned,
                    "input_burn": prompt_tokens,
                    "output_gen": completion_tokens
                },
                "balance": new_balance,
                "model": model_id
            }

        except Exception as e:
            print(f"Model {model_id} Error: {e}")
            continue

    raise HTTPException(503, "All Engines Busy")

# Helper to create keys
@app.post("/v1/user/new-key")
def create_key():
    k = f"sig-{secrets.token_hex(16)}"
    SUPABASE.table("users").insert({"api_key": k, "token_balance": 50000}).execute()
    return {"key": k}
