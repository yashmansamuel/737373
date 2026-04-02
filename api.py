import os
import logging
import secrets
import re
import asyncio
from typing import List, Tuple, Optional
from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# -----------------------------
# 1. Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Neo-L1.0-Core")

# -----------------------------
# 2. FastAPI Initialization
# -----------------------------
app = FastAPI(
    title="Signaturesi Neo L1.0",
    description="Universal AI Engine with Search & Logic Optimization",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change this to your specific domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 3. External Service Clients
# -----------------------------
try:
    SUPABASE: Client = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_KEY")
    )
    GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))
    logger.info("External services initialized successfully.")
except Exception as e:
    logger.error(f"Critical Service Failure: {e}")
    raise RuntimeError("Backend services failed to start.")

# -----------------------------
# 4. Global Settings & Prompts
# -----------------------------
MODELS = [
    "openai/gpt-oss-20b",           # Fast, stable, high rate limits
    "openai/gpt-oss-safeguard-20b", # Security-focused fallback
    "openai/gpt-oss-120b"           # High-intelligence fallback
]

# Universal System Prompt: Optimized for high-density, clean output.
SYSTEM_PROMPT = (
    "Act: Neo-Universal Assistant. Task: Provide high-quality, accurate, and direct responses. "
    "Guidelines: Use search for real-time facts. Maintain a neutral, professional tone. "
    "Rules: Strictly no internal reasoning text, no conversational fillers (e.g., 'Okay', 'I found'), "
    "and no search citations or technical tags (e.g., [1], 【†】). "
    "Ensure every sentence is grammatically complete. "
    "Constraint: Keep it under 4 concise sentences or 5 clear bullets for general queries. "
    "For code or technical tasks, provide the full necessary logic directly."
)

# -----------------------------
# 5. Data Models (Pydantic)
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

class NewKeyResponse(BaseModel):
    api_key: str
    balance: int
    status: str

# -----------------------------
# 6. Helper Functions
# -----------------------------
def clean_output(text: str) -> str:
    """Removes search citations and cleans up technical clutter."""
    # Remove citations like [1], [2], 【4†L308-L311】 etc.
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'【.*?】', '', text)
    return text.strip()

async def call_groq_engine(messages: List[dict]) -> Tuple[object, str]:
    """Handles logic for model fallback and API communication."""
    # Context Pruning: Keeps System Prompt + Last 2 interactions to save $
    optimized_context = [messages[0]] + messages[-2:] if len(messages) > 2 else messages

    for model_name in MODELS:
        try:
            logger.info(f"Attempting query with model: {model_name}")
            
            # Request Parameters
            kwargs = {
                "model": model_name,
                "messages": optimized_context,
                "temperature": 0.2,            # Balanced: Precise but natural
                "max_completion_tokens": 250,   # Longer cap for quality responses
                "stream": False,
            }

            # Search & Reasoning Logic
            if "gpt-oss" in model_name:
                kwargs["reasoning_effort"] = None  # Prevents reasoning-token billing
                kwargs["tools"] = [{"type": "browser_search"}]

            response = GROQ_CLIENT.chat.completions.create(**kwargs)
            return response, model_name

        except Exception as e:
            err_msg = str(e).lower()
            logger.warning(f"Model {model_name} failed: {err_msg}")
            
            if "rate_limit" in err_msg or "429" in err_msg:
                await asyncio.sleep(0.7) # Safety delay
            continue

    raise HTTPException(status_code=503, detail="All AI engines are currently at capacity.")

# -----------------------------
# 7. API Endpoints
# -----------------------------

@app.get("/")
async def root():
    return {"status": "active", "engine": "Neo-L1.0", "version": "1.0.0"}

@app.get("/v1/user/balance", response_model=BalanceResponse)
async def get_user_balance(api_key: str):
    result = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Invalid API Key provided.")
    return {"api_key": api_key, "balance": result.data["token_balance"]}

@app.post("/v1/user/new-key", response_model=NewKeyResponse)
async def create_api_key(request: Request):
    try:
        generated_key = f"sig-neo-{secrets.token_urlsafe(24)}"
        ip_country = request.headers.get("cf-ipcountry", "Unknown")
        
        SUPABASE.table("users").insert({
            "api_key": generated_key,
            "token_balance": 2000, # Starting credits
            "country": ip_country
        }).execute()
        
        return {"api_key": generated_key, "balance": 2000, "status": "Activated"}
    except Exception as e:
        logger.error(f"Key generation failed: {e}")
        raise HTTPException(status_code=500, detail="Database error during registration.")

@app.post("/v1/chat/completions")
async def chat_endpoint(payload: ChatRequest, authorization: str = Header(None)):
    # 1. Security Check
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization header.")
    
    user_api_key = authorization.replace("Bearer ", "")
    user_query = SUPABASE.table("users").select("token_balance").eq("api_key", user_api_key).maybe_single().execute()
    
    if not user_query.data:
        raise HTTPException(status_code=401, detail="Unauthorized: Key not found.")
    
    current_tokens = user_query.data["token_balance"]
    if current_tokens <= 0:
        raise HTTPException(status_code=402, detail="Token balance exhausted. Please top up.")

    # 2. AI Processing
    formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + payload.messages
    ai_raw_response, engine_id = await call_groq_engine(formatted_messages)

    # 3. Output Refinement
    raw_content = ai_raw_response.choices[0].message.content or ""
    final_content = clean_output(raw_content)

    # 4. Token Accounting
    usage_stats = ai_raw_response.usage
    total_spent = usage_stats.total_tokens
    new_bal = max(0, current_tokens - total_spent)

    # Background update for performance
    asyncio.create_task(
        asyncio.to_thread(
            lambda: SUPABASE.table("users").update({"token_balance": new_bal}).eq("api_key", user_api_key).execute()
        )
    )

    # 5. Return OpenAI-Compatible JSON
    return {
        "id": f"neo_gen_{secrets.token_hex(6)}",
        "object": "chat.completion",
        "created": 1712000000, # Placeholder timestamp
        "model": "Neo-L1.0",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": usage_stats.prompt_tokens,
            "completion_tokens": usage_stats.completion_tokens,
            "total_tokens": total_spent
        },
        "engine_details": engine_id
    }
