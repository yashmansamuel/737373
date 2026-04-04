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

# 1. Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-Core-V2")

SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="Neo L1.0 Lean Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 2. Ultra-Lean Prompt (Saves ~150 tokens per call)
SYSTEM_PROMPT = "You are Neo. Reply ONLY JSON: {\"final_answer\": \"...\", \"reasoning\": \"...\"}. Be direct."

MODELS = ["llama-3.3-70b-versatile", "qwen-qwq-32b-preview", "gemma2-9b-it"]

class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

# 3. Helpers
def get_knowledge(query: str) -> str:
    if len(query.split()) < 5: return "" # Don't burn tokens for short chat
    try:
        path = os.path.join(os.path.dirname(__file__), "knowledge.txt")
        if not os.path.exists(path): return ""
        words = [w.lower() for w in query.split() if len(w) > 3]
        matches = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if any(word in line.lower() for word in words):
                    matches.append(line.strip())
                if len(matches) >= 2: break # Limit context to save tokens
        return "\n".join(matches)
    except: return ""

async def update_bal(api_key: str, new_bal: int):
    await asyncio.to_thread(lambda: SUPABASE.table("users").update({"token_balance": new_bal}).eq("api_key", api_key).execute())

# 4. Main Chat Logic
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or "Bearer " not in authorization:
        raise HTTPException(401, "Key missing")
    
    api_key = authorization.split(" ")[1]
    user = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    
    if not user.data: raise HTTPException(401, "User not found")
    balance = user.data["token_balance"]
    if balance <= 0: raise HTTPException(402, "No Balance")

    user_query = payload.messages[-1].get("content", "")
    context = get_knowledge(user_query)

    # TOKEN SAVER: Build minimal message list
    # Sirf System Prompt + Context + Aakhri 2 Messages (Latest Context)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context:
        msgs.append({"role": "system", "content": f"Context: {context}"})
    
    # Sirf aakhri 2 messages bhej rahe hain taake history 10k tokens na khaye
    msgs.extend(payload.messages[-2:]) 

    for model_name in MODELS:
        try:
            response = GROQ.chat.completions.create(
                model=model_name,
                messages=msgs,
                temperature=0.5,
                max_tokens=2500, # Aapki 2k ki requirement puri hogi
                response_format={"type": "json_object"}
            )
            
            usage = response.usage
            total_burned = usage.total_tokens
            raw_content = response.choices[0].message.content
            
            try:
                parsed = json.loads(raw_content)
            except:
                parsed = {"final_answer": raw_content, "reasoning": "Parse Error"}

            new_balance = max(0, balance - total_burned)
            asyncio.create_task(update_bal(api_key, new_balance))

            return {
                "company": "signaturesi.com",
                "final_answer": parsed.get("final_answer"),
                "reasoning": parsed.get("reasoning"),
                "usage": {
                    "total": total_burned,
                    "prompt": usage.prompt_tokens,
                    "completion": usage.completion_tokens
                },
                "balance": new_balance
            }
        except Exception as e:
            logger.error(f"Fail: {model_name} | {e}")
            continue

    raise HTTPException(503, "All models failed")

# Standard Routes
@app.get("/")
def home(): return {"status": "Neo L1.0 Online"}

@app.get("/v1/user/balance")
def check_bal(api_key: str):
    u = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    return {"balance": u.data["token_balance"] if u.data else 0}

@app.post("/v1/user/new-key")
def new_key():
    k = "sig-" + secrets.token_hex(16)
    SUPABASE.table("users").insert({"api_key": k, "token_balance": 100000}).execute()
    return {"api_key": k}
