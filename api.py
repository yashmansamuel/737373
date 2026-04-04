import os
import json
import secrets
import asyncio
from typing import List
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from groq import Groq

# Setup
SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# System Prompt: Strict Instruction for AGI and Token usage
SYSTEM_PROMPT = (
    "You are Neo L1.0, a high-capacity Research AI. "
    "Your task is to provide exhaustive, deep-dive academic content. "
    "Structure: { \"final_answer\": \"(Detailed 3000+ word content here)\", \"reasoning\": \"(Short 50-token logic)\" }. "
    "Ensure the JSON is valid and escape all newlines. Maximize the final_answer length."
)

class ChatRequest(BaseModel):
    messages: List[dict]

@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization: raise HTTPException(401)
    api_key = authorization.replace("Bearer ", "")
    
    # 1. Real Balance Check
    user = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    if not user.data: raise HTTPException(401)
    balance = user.data["token_balance"]

    # 2. Token Saving History (Limit to last 2 for accuracy)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.extend(payload.messages[-2:])

    try:
        # 3. Groq Call with Real Usage Tracking
        response = GROQ.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=msgs,
            temperature=0.6,
            max_tokens=4096, # Increased for long research papers
            response_format={"type": "json_object"}
        )
        
        # Real stats from Groq
        usage = response.usage
        total_tokens = usage.total_tokens
        content = response.choices[0].message.content
        
        # Validating JSON to avoid [object Object] error
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Fallback if JSON is malformed due to length
            data = {"final_answer": content, "reasoning": "Output was too large for standard JSON parsing."}

        # 4. Accurate Deduction
        new_balance = max(0, balance - total_tokens)
        asyncio.create_task(update_balance(api_key, new_balance))

        return {
            "final_answer": data.get("final_answer"),
            "reasoning": data.get("reasoning"),
            "usage": {
                "total": total_tokens,
                "input": usage.prompt_tokens,
                "output": usage.completion_tokens
            },
            "balance": new_balance
        }

    except Exception as e:
        raise HTTPException(500, str(e))

async def update_balance(api_key, val):
    SUPABASE.table("users").update({"token_balance": val}).eq("api_key", api_key).execute()
