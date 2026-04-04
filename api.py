import os
import secrets
import asyncio
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Clients
SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Minimal System Prompt for low overhead
SYSTEM_PROMPT = "You are Neo L1.0. Output ONLY JSON: {\"final_answer\": \"...\", \"reasoning\": \"...\"}."

class ChatRequest(BaseModel):
    messages: List[dict]

@app.post("/v1/chat/completions")
async def chat_engine(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or "Bearer " not in authorization:
        raise HTTPException(401, "API Key Missing")
    
    api_key = authorization.split(" ")[1]
    
    # 1. Fetch User from Supabase
    user_res = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    if not user_res.data:
        raise HTTPException(401, "Invalid Key")
    
    current_balance = user_res.data["token_balance"]
    if current_balance <= 0:
        raise HTTPException(402, "Balance Exhausted")

    # 2. Prepare Minimal Messages (Latest 2 for context saving)
    input_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    input_messages.extend(payload.messages[-2:])

    try:
        # 3. Call Groq
        response = GROQ.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=input_messages,
            temperature=0.5,
            response_format={"type": "json_object"}
        )

        # 4. GET ACTUAL TOKENS FROM GROQ (No estimation here)
        # Groq returns exact usage: prompt_tokens + completion_tokens
        actual_usage = response.usage
        total_tokens_used = actual_usage.total_tokens 
        
        # 5. Extract Content
        raw_content = response.choices[0].message.content
        try:
            data = json.loads(raw_content)
        except:
            data = {"final_answer": raw_content, "reasoning": "N/A"}

        # 6. Update Database with EXACT count
        new_balance = max(0, current_balance - total_tokens_used)
        
        # Background task for DB sync to keep API fast
        SUPABASE.table("users").update({"token_balance": new_balance}).eq("api_key", api_key).execute()

        # 7. Return exact figures to User
        return {
            "company": "signaturesi.com",
            "final_answer": data.get("final_answer"),
            "reasoning": data.get("reasoning"),
            "usage": {
                "total": total_tokens_used, # REAL count from Groq
                "prompt": actual_usage.prompt_tokens,
                "completion": actual_usage.completion_tokens
            },
            "balance": new_balance
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(500, "Neo Engine Error")

@app.post("/v1/user/new-key")
def create_key():
    new_key = f"sig-{secrets.token_hex(16)}"
    SUPABASE.table("users").insert({"api_key": new_key, "token_balance": 50000}).execute()
    return {"api_key": new_key, "balance": 50000}
