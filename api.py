import os
import logging
import secrets
import asyncio
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Signaturesi Neo L1.0 Pro API")

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clients Initialization
try:
    supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    logger.info("All services connected successfully.")
except Exception as e:
    logger.error(f"Initialization Error: {e}")
    raise RuntimeError("Check your Environment Variables (Groq, Supabase, Tavily)")

# Updated System Prompt for Real-time context
SYSTEM_PROMPT = """You are Neo L1.0, a high-intelligence AI. 
Knowledge Date: April 2026. Use the provided search context to answer accurately. 
Format: Concise, max 3 bullets or 2 sentences. No fluff."""

GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

async def get_search_context(query: str):
    """Real-time web search using Tavily."""
    try:
        # Search for latest 2026 data
        search_result = tavily.search(query=query, search_depth="basic", max_results=3)
        context = "\n".join([f"- {r['content']}" for r in search_result['results']])
        return context
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return "No real-time data available."

async def call_groq_with_fallback(messages):
    for model in GROQ_MODELS:
        try:
            loop = asyncio.get_event_loop()
            completion = await loop.run_in_executor(
                None, 
                lambda: groq_client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=0.2, # Lower temp for facts
                    max_tokens=500
                )
            )
            return completion, model
        except Exception as e:
            logger.warning(f"Model {model} failed: {e}")
            continue
    raise HTTPException(500, "All AI models are currently busy.")

@app.post("/v1/chat/completions")
async def chat_proxy(request: Request, authorization: str = Header(None)):
    # 1. Auth & Validation
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing API Key")
    
    user_api_key = authorization.replace("Bearer ", "")
    body = await request.json()
    user_query = body.get("messages")[-1]["content"]

    # 2. Check Balance from Supabase
    resp = supabase.table("users").select("token_balance").eq("api_key", user_api_key).execute()
    if not resp.data:
        raise HTTPException(401, "Invalid Key")
    
    balance = resp.data[0]["token_balance"]
    if balance <= 0:
        raise HTTPException(402, "Insufficient Balance")

    # 3. REAL-TIME SEARCH (The Fix!)
    # We fetch latest data before asking the AI
    search_data = await get_search_context(user_query)
    
    # 4. Prepare Augmented Prompt (RAG)
    enriched_messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCURRENT REAL-TIME CONTEXT:\n{search_data}"},
        {"role": "user", "content": user_query}
    ]

    # 5. Get AI Response
    ai_response, used_model = await call_groq_with_fallback(enriched_messages)
    assistant_content = ai_response.choices[0].message.content

    # 6. Deduct Tokens & Update DB
    tokens_used = ai_response.usage.completion_tokens
    new_balance = max(0, balance - tokens_used)
    supabase.table("users").update({"token_balance": new_balance}).eq("api_key", user_api_key).execute()

    return {
        "message": assistant_content.strip(),
        "usage": {"tokens_deducted": tokens_used},
        "model": "Neo-L1.0",
        "real_time": True
    }

# Standard routes
@app.get("/")
def home(): return {"status": "Online", "engine": "Neo L1.0 + Search"}

@app.post("/v1/user/new-key")
async def generate_key():
    new_key = "sig-live-" + secrets.token_urlsafe(16)
    supabase.table("users").insert({"api_key": new_key, "token_balance": 5000}).execute()
    return {"api_key": new_key, "balance": 5000}
