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

# 1. Setup & Config
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-L1.0-Core")

SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="Neo L1.0 Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 2. Optimized System Prompt (Short & Direct = Cheap)
SYSTEM_PROMPT = "You are Neo L1.0. Output ONLY JSON: {\"final_answer\": \"...\", \"reasoning\": \"...\"}. Be concise. No filler."

MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3-32b", "llama-3.3-70b-versatile"]

# 3. Models
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

# 4. Helpers
def get_neo_knowledge(user_query: str) -> str:
    """Only search if query is meaningful (>3 words)"""
    if len(user_query.split()) < 4: return ""
    try:
        file_path = os.path.join(os.path.dirname(__file__), "knowledge.txt")
        if not os.path.exists(file_path): return ""
        
        words = [w.lower() for w in user_query.split() if len(w) > 3]
        matches = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if any(word in line.lower() for word in words):
                    matches.append(line.strip())
                if len(matches) >= 3: break # Limit matches to save tokens
        return "\n".join(matches)
    except: return ""

def get_user_data(api_key: str):
    return SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()

async def update_balance(api_key: str, new_balance: int):
    await asyncio.to_thread(lambda: SUPABASE.table("users").update({"token_balance": new_balance}).eq("api_key", api_key).execute())

# 5. API Routes
@app.get("/")
async def root():
    return {"company": "signaturesi.com", "engine": "Neo L1.0", "status": "active"}

@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    user = get_user_data(api_key)
    return {"api_key": api_key, "balance": user.data.get("token_balance", 0) if user.data else 0}

@app.post("/v1/user/new-key")
def generate_key():
    new_key = "sig-" + secrets.token_hex(16)
    SUPABASE.table("users").insert({"api_key": new_key, "token_balance": 100000}).execute()
    return {"api_key": new_key, "company": "signaturesi.com"}

@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or "Bearer " not in authorization:
        raise HTTPException(401, "Invalid Key")
    
    api_key = authorization.split(" ")[1]
    user = get_user_data(api_key)
    if not user.data: raise HTTPException(401, "User not found")
    
    balance = user.data["token_balance"]
    if balance <= 0: raise HTTPException(402, "Insufficient Balance")

    user_msg = payload.messages[-1].get("content", "")
    context = get_neo_knowledge(user_msg)

    # Build Minimal Message List
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context:
        msgs.append({"role": "system", "content": f"Fact: {context}"})
    
    # Send only last 8 messages for history (Token saving)
    msgs.extend(payload.messages[-8:])

    for model_name in MODELS:
        try:
            res = GROQ.chat.completions.create(
                model=model_name,
                messages=msgs,
                temperature=0.4,
                response_format={"type": "json_object"}
            )
            
            content = res.choices[0].message.content
            usage = res.usage
            t_tokens = usage.total_tokens
            
            # Parse Output
            try:
                out = json.loads(content)
            except:
                out = {"final_answer": content, "reasoning": ""}

            new_bal = max(0, balance - t_tokens)
            asyncio.create_task(update_balance(api_key, new_bal))

            return {
                "company": "signaturesi.com",
                "final_answer": out.get("final_answer"),
                "reasoning": out.get("reasoning"),
                "usage": {"total": t_tokens, "prompt": usage.prompt_tokens, "completion": usage.completion_tokens},
                "model": "Neo L1.0",
                "balance": new_bal
            }
        except Exception as e:
            logger.error(f"Error with {model_name}: {e}")
            continue

    raise HTTPException(503, "All engines busy")
