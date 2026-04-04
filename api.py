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

required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing required environment variable: {var}")

app = FastAPI(title="Neo L1.0 Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 2. Models & Prompts
MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile"
]

# Optimized Prompt to save input tokens
SYSTEM_PROMPT = """You are Neo L1.0. Output ONLY valid JSON:
{"final_answer": "...", "reasoning": "..."}.
Rules: Use hierarchical bullets for long answers. No filler or polite talk. 
final_answer: max 2000 tokens. reasoning: max 800 tokens."""

# 3. Pydantic Models
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

# 4. Custom Branding & Error Handlers
@app.get("/")
async def root():
    return {
        "company": "signaturesi.com",
        "engine": "Neo L1.0 Core",
        "status": "running",
        "deployment": "Jan 1, 2026"
    }

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"company": "signaturesi.com", "status": "running", "message": "Endpoint not found"}
    )

# 5. Knowledge Engine (RAG)
def get_neo_knowledge(user_query: str) -> str:
    try:
        # Optimization: Don't run RAG for tiny messages (hi, hello)
        if len(user_query.split()) < 3:
            return ""
            
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path):
            return ""

        query_words = list(set(w.lower() for w in user_query.split() if len(w) > 3))
        matches = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_lower = line.lower()
                score = sum(word in line_lower for word in query_words)
                if score >= 1:
                    matches.append(line.strip())
                if len(matches) >= 5: break
        return "\n".join(matches)
    except Exception as e:
        logger.error(f"Knowledge retrieval error: {e}")
        return ""

# 6. Helper Functions
def get_user(api_key: str):
    return SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()

async def update_balance(api_key: str, new_balance: int):
    try:
        await asyncio.to_thread(
            lambda: SUPABASE.table("users").update({"token_balance": new_balance}).eq("api_key", api_key).execute()
        )
    except Exception as e:
        logger.error(f"Failed to update balance: {e}")

# 7. API Routes
@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    try:
        user = get_user(api_key)
        balance = user.data.get("token_balance", 0) if user.data else 0
        return {"api_key": api_key, "balance": balance}
    except Exception:
        raise HTTPException(500, "Balance fetch failed")

@app.post("/v1/user/new-key")
def generate_key():
    try:
        api_key = "sig-" + secrets.token_hex(16)
        SUPABASE.table("users").insert({"api_key": api_key, "token_balance": 100000}).execute()
        return {"api_key": api_key, "company": "signaturesi.com"}
    except Exception:
        raise HTTPException(500, "Failed to create key")

@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key")

    api_key = authorization.replace("Bearer ", "")
    user = get_user(api_key)

    if not user.data: raise HTTPException(401, "User not found")
    balance = user.data["token_balance"]
    if balance <= 0: raise HTTPException(402, "No tokens left")

    user_msg = payload.messages[-1].get("content", "") if payload.messages else ""
    local_data = get_neo_knowledge(user_msg)

    # Dynamic token limit to prevent over-burning
    max_completion_tokens = 300 if len(user_msg.split()) < 5 else 2500

    # Building clean messages
    final_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if local_data:
        final_messages.append({"role": "system", "content": f"Context: {local_data}"})
    
    # Send only necessary history (e.g., last 10 messages) to save tokens
    final_messages.extend(payload.messages[-10:])

    for model_name in MODELS:
        try:
            response = GROQ.chat.completions.create(
                model=model_name,
                messages=final_messages,
                temperature=0.5,
                max_tokens=max_completion_tokens,
                response_format={"type": "json_object"}
            )

            raw_content = response.choices[0].message.content
            usage = response.usage
            total_tokens = usage.total_tokens if usage else 0

            try:
                parsed = json.loads(raw_content)
                final_answer = parsed.get("final_answer", "")
                reasoning = parsed.get("reasoning", "")
            except json.JSONDecodeError:
                final_answer, reasoning = raw_content, ""

            new_balance = max(0, balance - total_tokens)
            asyncio.create_task(update_balance(api_key, new_balance))

            return {
                "company": "signaturesi.com",
                "final_answer": final_answer,
                "reasoning": reasoning,
                "usage": {
                    "total_tokens": total_tokens,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "prompt_tokens": usage.prompt_tokens if usage else 0
                },
                "model": "Neo L1.0",
                "internal_engine": model_name,
                "balance": new_balance
            }

        except Exception as e:
            logger.error(f"{model_name} failed: {e}")
            continue

    raise HTTPException(503, detail="All Neo models failed.")
