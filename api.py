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
logger = logging.getLogger("Neo-Final-Core")

SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="Neo L1.0 Final")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Models List
MODELS = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "llama-3.1-8b-instant"]

SYSTEM_PROMPT = """Identity: Neo L1.0. Deployment: 2026.
Role: Professional Researcher.
Format: ALWAYS return a valid JSON object with keys "final_answer" and "reasoning".
Reasoning: Provide 3-5 sentences of deep logical breakdown (~120 tokens).
Final Answer: Provide an exhaustive, high-quality response.
Constraint: Escape all special characters and newlines for valid JSON.
"""

class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

# 2. Knowledge Retrieval
def get_neo_context(query: str) -> str:
    try:
        path = os.path.join(os.path.dirname(__file__), "knowledge.txt")
        if not os.path.exists(path): return ""
        words = [w.lower() for w in query.split() if len(w) > 3]
        matches = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if any(w in line.lower() for w in words): matches.append(line.strip())
                if len(matches) >= 3: break
        return "\n".join(matches)
    except: return ""

# 3. Core Chat Engine
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization: raise HTTPException(401, "No API Key")
    api_key = authorization.replace("Bearer ", "")
    
    # Check Balance
    user = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    if not user.data: raise HTTPException(401, "Invalid Key")
    balance = user.data["token_balance"]
    if balance <= 0: raise HTTPException(402, "Out of tokens")

    # Build Messages (History Locked to Last 2)
    context = get_neo_context(payload.messages[-1].get("content", ""))
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context: msgs.append({"role": "system", "content": f"Context: {context}"})
    msgs.extend(payload.messages[-2:])

    for model_name in MODELS:
        try:
            response = GROQ.chat.completions.create(
                model=model_name,
                messages=msgs,
                temperature=0.5,
                max_tokens=2500, # Balanced for 15req/10k tokens
                response_format={"type": "json_object"}
            )

            # Accurate Token Tracking
            usage = response.usage
            total_burned = usage.total_tokens
            
            raw_content = response.choices[0].message.content
            parsed = json.loads(raw_content)

            # Update DB
            new_bal = max(0, balance - total_burned)
            asyncio.create_task(asyncio.to_thread(
                lambda: SUPABASE.table("users").update({"token_balance": new_bal}).eq("api_key", api_key).execute()
            ))

            return {
                "company": "signaturesi.com",
                "final_answer": parsed.get("final_answer"),
                "reasoning": parsed.get("reasoning"),
                "usage": {"total": total_burned},
                "balance": new_bal
            }

        except Exception as e:
            logger.error(f"Error with {model_name}: {e}. Retrying next model in 4s...")
            await asyncio.sleep(4) # 4 Second Delay to prevent Groq spam
            continue

    raise HTTPException(503, "All engines overloaded.")

# 4. Utilities
@app.post("/v1/user/new-key")
def create_key():
    k = "sig-" + secrets.token_hex(16)
    SUPABASE.table("users").insert({"api_key": k, "token_balance": 100000}).execute()
    return {"api_key": k}
