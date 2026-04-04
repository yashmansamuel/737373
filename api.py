import os
import logging
import secrets
import asyncio
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# 1. Configuration & Logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-Core-V3")

# Initialize Clients
SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="Neo L1.0 High-Density Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Optimized Constants
# Short prompt = Low Input Cost
SYSTEM_PROMPT = (
    "You are Neo L1.0, a high-density intelligence engine. "
    "Role: Scientific Researcher. Output ONLY valid JSON: "
    "{\"final_answer\": \"...\", \"reasoning\": \"...\"}. "
    "Be technical, direct, and avoid fluff."
)

# Priority Models (Groq IDs)
MODELS = [
    "llama-3.3-70b-versatile", 
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

# 3. Request Models
class ChatRequest(BaseModel):
    messages: List[dict]
    model: Optional[str] = "Neo L1.0"

# 4. Core Logic Functions
def get_local_context(query: str) -> str:
    """Retrieves context from knowledge.txt only if query is complex."""
    if len(query.split()) < 4: return ""
    try:
        file_path = os.path.join(os.path.dirname(__file__), "knowledge.txt")
        if not os.path.exists(file_path): return ""
        
        keywords = [w.lower() for w in query.split() if len(w) > 3]
        matches = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if any(k in line.lower() for k in keywords):
                    matches.append(line.strip())
                if len(matches) >= 2: break 
        return "\n".join(matches)
    except Exception as e:
        logger.error(f"RAG Error: {e}")
        return ""

async def update_user_balance(api_key: str, cost: int):
    """Deducts tokens from Supabase balance."""
    try:
        user = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).single().execute()
        if user.data:
            new_bal = max(0, user.data["token_balance"] - cost)
            SUPABASE.table("users").update({"token_balance": new_bal}).eq("api_key", api_key).execute()
            return new_bal
    except Exception as e:
        logger.error(f"DB Balance Update Failed: {e}")
    return 0

# 5. Main API Endpoint
@app.post("/v1/chat/completions")
async def chat_engine(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or "Bearer " not in authorization:
        raise HTTPException(401, "Missing API Key")
    
    api_key = authorization.split(" ")[1]
    
    # Verify User
    user_res = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    if not user_res.data:
        raise HTTPException(401, "Invalid API Key")
    
    current_balance = user_res.data["token_balance"]
    if current_balance <= 0:
        raise HTTPException(402, "Insufficient Tokens")

    # Prepare Content
    user_query = payload.messages[-1]["content"]
    context = get_local_context(user_query)

    # Token-Saving Strategy: Only send System Prompt + Context + Last 2 Messages
    # Isse "Hidden Cost" kabhi bhi 300 tokens se upar nahi jayegi
    refined_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context:
        refined_messages.append({"role": "system", "content": f"Local Context: {context}"})
    
    # History Limit (Critical for balance)
    refined_messages.extend(payload.messages[-2:])

    # Model Fallback Loop
    for model_id in MODELS:
        try:
            response = GROQ.chat.completions.create(
                model=model_id,
                messages=refined_messages,
                temperature=0.4,
                max_tokens=2500, # Supports long answers up to 2k tokens
                response_format={"type": "json_object"}
            )
            
            usage = response.usage
            total_cost = usage.total_tokens
            raw_content = response.choices[0].message.content
            
            # Parse Response
            try:
                data = json.loads(raw_content)
            except:
                data = {"final_answer": raw_content, "reasoning": "Direct response generated."}

            # Update Balance
            remaining_bal = await update_user_balance(api_key, total_cost)

            return {
                "company": "signaturesi.com",
                "final_answer": data.get("final_answer"),
                "reasoning": data.get("reasoning"),
                "usage": {
                    "total": total_cost,
                    "prompt": usage.prompt_tokens,
                    "completion": usage.completion_tokens
                },
                "balance": remaining_bal,
                "engine": model_id
            }

        except Exception as e:
            logger.warning(f"Model {model_id} failed: {e}")
            continue

    raise HTTPException(503, "All Neo engines are currently offline.")

# 6. Utility Routes
@app.post("/v1/user/new-key")
def create_key():
    new_key = f"sig-{secrets.token_hex(16)}"
    SUPABASE.table("users").insert({"api_key": new_key, "token_balance": 100000}).execute()
    return {"api_key": new_key, "initial_balance": 100000}

@app.get("/health")
def health():
    return {"status": "operational", "engine": "Neo L1.0 Core"}
