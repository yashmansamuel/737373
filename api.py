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

# 1. Configuration & Logs
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-Final-Core")

SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="Neo L1.0 Neutral Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Models List (For Stable Shifting)
MODELS = [
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "llama-3.1-8b-instant"
]

# The "Perfect Combo" Prompt
SYSTEM_PROMPT = """Identity: Neo L1.0. Role: High-Density Intelligence Engine.
STRICT OUTPUT RULES:
1. Return ONLY valid JSON: {"final_answer": "...", "reasoning": "..."}.
2. REASONING FIELD: Explain your 100-150 token logical foundation. Do NOT put tables or the main answer here.
3. FINAL_ANSWER FIELD: Put the actual detailed response, tables, and markdown here.
4. Formatting: Escape all double quotes (\\") and newlines (\\n) properly."""

class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

# 2. Knowledge Retrieval (RAG)
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

# 3. Core Chat Engine
@app.post("/v1/chat/completions")
async def chat_engine(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or "Bearer " not in authorization:
        raise HTTPException(401, "Valid API Key Required")
    
    api_key = authorization.replace("Bearer ", "")
    user_res = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    
    if not user_res.data: raise HTTPException(401, "User Not Found")
    balance = user_res.data["token_balance"]
    if balance <= 0: raise HTTPException(402, "Zero Balance")

    # TOKEN SAVER: Memory Lock (Last 2 Messages)
    user_query = payload.messages[-1].get("content", "")
    context = get_neo_knowledge(user_query)
    
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context: msgs.append({"role": "system", "content": f"Context: {context}"})
    msgs.extend(payload.messages[-2:] if len(payload.messages) > 2 else payload.messages)

    # MODEL SHIFTING WITH DYNAMIC 4s DELAY
    for i, model_id in enumerate(MODELS):
        try:
            if i > 0: 
                logger.info(f"Shifting to {model_id}. Waiting 4s...")
                await asyncio.sleep(4)

            response = GROQ.chat.completions.create(
                model=model_id,
                messages=msgs,
                temperature=0.5,
                max_tokens=3500,
                response_format={"type": "json_object"}
            )

            usage = response.usage
            total_burned = usage.total_tokens
            raw_content = response.choices[0].message.content

            # Safe Parsing to avoid [object Object]
            try:
                data = json.loads(raw_content)
            except:
                data = {"final_answer": raw_content, "reasoning": "Fallback: JSON extraction failed."}

            # Update DB
            new_bal = max(0, balance - total_burned)
            asyncio.create_task(asyncio.to_thread(
                lambda: SUPABASE.table("users").update({"token_balance": new_bal}).eq("api_key", api_key).execute()
            ))

            return {
                "company": "signaturesi.com",
                "final_answer": data.get("final_answer"),
                "reasoning": data.get("reasoning"),
                "usage": {"total": total_burned, "prompt": usage.prompt_tokens, "completion": usage.completion_tokens},
                "balance": new_bal,
                "engine": model_id
            }

        except Exception as e:
            logger.error(f"Engine {model_id} failed: {e}")
            continue

    raise HTTPException(503, "All engines overloaded.")

# 4. Utilities
@app.get("/health")
def health(): return {"status": "operational", "engine": "Neo L1.0 Final"}

@app.post("/v1/user/new-key")
def create_key():
    k = f"sig-{secrets.token_hex(16)}"
    SUPABASE.table("users").insert({"api_key": k, "token_balance": 100000}).execute()
    return {"api_key": k}
