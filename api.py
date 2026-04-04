import os
import logging
import secrets
import json
from typing import List
from fastapi import FastAPI, HTTPException, Header, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# -----------------------------
# 1. Setup & Config
# -----------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-L1.0-Core")

required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing required environment variable: {var}")

app = FastAPI(title="Neo L1.0 Engine - Optimized")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# 2. Models & Prompts
# -----------------------------
# Note: Ensure these model names are active in your Groq account
MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "mixtral-8x7b-32768"
]

SYSTEM_PROMPT = """
You are Neo L1.0 — a High-Density Information Engine.
Your output MUST be a valid JSON object. 

Structure:
{
  "reasoning": "Brief logic flow (max 500 tokens)",
  "final_answer": "Comprehensive, detailed information (up to 2500 tokens)"
}

Rules:
- NEVER cut 'final_answer' mid-sentence.
- Use hierarchical bullets for complex topics.
- Eliminate all filler words and polite transitions.
- If the query is short, still provide both fields but keep reasoning concise.
"""

# -----------------------------
# 3. Pydantic Models
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

# -----------------------------
# 4. Helper Functions
# -----------------------------
def update_user_balance(api_key: str, new_balance: int):
    try:
        SUPABASE.table("users").update({"token_balance": new_balance}).eq("api_key", api_key).execute()
    except Exception as e:
        logger.error(f"DB Update Error: {e}")

def get_neo_knowledge(user_query: str) -> str:
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path): return ""

        query_words = [w.lower() for w in user_query.split() if len(w) > 3]
        matches = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if any(word in line.lower() for word in query_words):
                    matches.append(line.strip())
                if len(matches) >= 5: break
        return "\n".join(matches)
    except Exception: return ""

# -----------------------------
# 5. API Routes
# -----------------------------
@app.get("/")
async def root():
    return {"company": "signaturesi.com", "engine": "Neo L1.0 Core", "status": "online"}

@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, background_tasks: BackgroundTasks, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid API key")

    api_key = authorization.replace("Bearer ", "")
    
    # 1. Check User
    user_res = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    if not user_res.data:
        raise HTTPException(401, "User unauthorized")
    
    balance = user_res.data["token_balance"]
    if balance <= 0:
        raise HTTPException(402, "Insufficient token balance")

    # 2. Context & Prompt Setup
    user_msg = payload.messages[-1].get("content", "")
    local_context = get_neo_knowledge(user_msg)
    
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if local_context:
        full_messages.append({"role": "system", "content": f"Context: {local_context}"})
    full_messages.extend(payload.messages)

    # 3. Model Execution
    for model_name in MODELS:
        try:
            response = GROQ.chat.completions.create(
                model=model_name,
                messages=full_messages,
                temperature=0.5,
                max_tokens=3000, # Hard limit
                response_format={"type": "json_object"} # Forces JSON
            )

            # Parse JSON safely
            raw_content = response.choices[0].message.content
            parsed_res = json.loads(raw_content)
            
            total_tokens = response.usage.total_tokens
            new_balance = max(0, balance - total_tokens)

            # Update DB in background to keep API fast
            background_tasks.add_task(update_user_balance, api_key, new_balance)

            return {
                "company": "signaturesi.com",
                "final_answer": parsed_res.get("final_answer", "Error generating answer"),
                "reasoning": parsed_res.get("reasoning", "No reasoning provided"),
                "usage": {"total_tokens": total_tokens},
                "model": "Neo L1.0",
                "internal_engine": model_name,
                "balance": new_balance
            }

        except Exception as e:
            logger.error(f"Model {model_name} error: {e}")
            continue

    raise HTTPException(503, "All Neo engines are currently unavailable")
