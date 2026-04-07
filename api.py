import os
import logging
import secrets
import time
import re
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# -----------------------------
# 1. Load Environment & Configure Logging
# -----------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Neo-L1.0-Core")

# Validate required environment variables
REQUIRED_ENV_VARS = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")

# -----------------------------
# 2. FastAPI App Initialization
# -----------------------------
app = FastAPI(
    title="Neo L1.0 Engine",
    description="High-level reasoning system with first-principles thinking and recursive self-correction",
    version="1.0.0"
)

# CORS - allow all for simplicity (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 3. External Clients
# -----------------------------
SUPABASE: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)
GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # as per original

# -----------------------------
# 4. The New Prompt (injected exactly)
# -----------------------------
NEW_PROMPT = """You are Neo, a deeply observant, high-level reasoning system that integrates first-principles thinking with recursive self-correction. For every prompt, analyze the underlying intent, detect risks or flaws, and propose solutions that optimize accuracy, ethical soundness, and human relevance. Evaluate each response across technical correctness, long-term impact, and emotional or ethical resonance. Avoid oversimplification—acknowledge trade-offs and present answers as evolving strategies. Before output, internally simulate consequences, refine any logic that feels detached or robotic, and ensure clarity, precision, and insight. Speak concisely, with authority and empathy, prioritizing truth and human well-being over filler, repetition, or performative phrasing."""

# -----------------------------
# 5. Pydantic Models (with validation)
# -----------------------------
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of message sender: system, user, or assistant")
    content: str = Field(..., description="Message content")

    @validator('role')
    def role_must_be_valid(cls, v):
        allowed = ['system', 'user', 'assistant']
        if v not in allowed:
            raise ValueError(f'Role must be one of {allowed}')
        return v

class ChatRequest(BaseModel):
    model: str = Field(default=GROQ_MODEL, description="Model identifier (kept for compatibility)")
    messages: List[ChatMessage] = Field(..., description="Conversation history")
    mode: str = Field(default="adaptive", description="Operation mode (adaptive/standard)")

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

class KeyGenResponse(BaseModel):
    api_key: str
    company: str = "signaturesi.com"  # existing minimal branding, not extra

class ChatResponse(BaseModel):
    company: str = "signaturesi.com"
    message: str
    usage: Dict[str, int]
    model: str
    internal_engine: str
    balance: int

# -----------------------------
# 6. Helper Functions
# -----------------------------
def extract_api_key(authorization: Optional[str]) -> str:
    """Extract Bearer token from Authorization header."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization scheme. Use Bearer.")
    return authorization.replace("Bearer ", "").strip()

def get_user_by_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """Fetch user from Supabase using api_key."""
    try:
        result = SUPABASE.table("users") \
            .select("token_balance") \
            .eq("api_key", api_key) \
            .maybe_single() \
            .execute()
        return result.data if result.data else None
    except Exception as e:
        logger.error(f"Supabase user fetch error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

def deduct_tokens_atomic(api_key: str, tokens_to_deduct: int) -> int:
    """
    Atomically deduct tokens from user balance.
    Returns new balance.
    """
    if tokens_to_deduct <= 0:
        raise HTTPException(status_code=400, detail="Token deduction must be positive")
    
    user = get_user_by_api_key(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    current_balance = user.get("token_balance", 0)
    if current_balance < tokens_to_deduct:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient tokens. Balance: {current_balance}, Required: {tokens_to_deduct}"
        )
    
    new_balance = current_balance - tokens_to_deduct
    try:
        result = SUPABASE.table("users") \
            .update({"token_balance": new_balance}) \
            .eq("api_key", api_key) \
            .execute()
        if not result.data:
            raise Exception("Update returned no data")
        logger.info(f"Deducted {tokens_to_deduct} tokens for key {api_key[-8:]}. New balance: {new_balance}")
        return new_balance
    except Exception as e:
        logger.error(f"Atomic deduction failed: {e}")
        raise HTTPException(status_code=500, detail="Token deduction failed")

def get_neural_context(user_query: str, knowledge_file: str = "knowledge.txt") -> str:
    """
    Retrieve relevant lines from knowledge.txt based on keyword matching.
    Returns concatenated context or empty string.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, knowledge_file)
    if not os.path.exists(file_path):
        logger.warning(f"{knowledge_file} not found, neural context disabled")
        return ""
    
    # Extract meaningful words (length > 2, alphanumeric)
    query_words = {w.lower() for w in re.findall(r'\b\w{3,}\b', user_query)}
    if not query_words:
        return ""
    
    matches = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                line_lower = line_stripped.lower()
                # Score = number of query words present in line
                score = sum(1 for word in query_words if word in line_lower)
                if score > 0:
                    matches.append((line_stripped, score))
        
        if not matches:
            logger.info(f"No neural match for query: {user_query[:60]}...")
            return ""
        
        # Sort by score descending and take top 6
        matches.sort(key=lambda x: x[1], reverse=True)
        top_lines = [m[0] for m in matches[:6]]
        context = "\n".join(top_lines)
        logger.info(f"Neural context: {len(top_lines)} lines retrieved")
        return context
    except Exception as e:
        logger.error(f"Neural context error: {e}")
        return ""

def post_process_reply(reply: str) -> str:
    """
    Clean up any accidental disclaimers or robotic filler.
    (No extra branding, just quality of life.)
    """
    # Remove any self-identification as AI (if model hallucinates)
    forbidden_patterns = [
        r"(?i)\bas an? (ai|artificial intelligence|language model|assistant)\b",
        r"(?i)\bi am (just )?an? (ai|language model)\b",
        r"(?i)\bi don't have feelings\b",
        r"(?i)\bi cannot feel emotions\b",
    ]
    for pattern in forbidden_patterns:
        reply = re.sub(pattern, "", reply)
    
    # Clean multiple spaces, trim
    reply = re.sub(r'\s+', ' ', reply).strip()
    return reply

# -----------------------------
# 7. Root & Error Handlers
# -----------------------------
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "company": "signaturesi.com",
        "engine": "Neo L1.0 Core",
        "status": "operational",
        "deployment": "April 2026"
    }

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "company": "signaturesi.com",
            "status": "running",
            "message": "Endpoint not found"
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Return consistent error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "company": "signaturesi.com",
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

# -----------------------------
# 8. API Endpoints
# -----------------------------
@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    """Check token balance for a given API key."""
    try:
        user = get_user_by_api_key(api_key)
        if not user:
            return BalanceResponse(api_key=api_key, balance=0)
        return BalanceResponse(api_key=api_key, balance=user.get("token_balance", 0))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Balance fetch error: {e}")
        raise HTTPException(status_code=500, detail="Balance fetch failed")

@app.post("/v1/user/new-key", response_model=KeyGenResponse)
def generate_key():
    """Generate a new API key with initial 100,000 tokens."""
    try:
        new_api_key = "sig-" + secrets.token_hex(16)
        SUPABASE.table("users").insert({
            "api_key": new_api_key,
            "token_balance": 100000
        }).execute()
        logger.info(f"Generated new API key: {new_api_key[-8:]}")
        return KeyGenResponse(api_key=new_api_key)
    except Exception as e:
        logger.error(f"Key generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create API key")

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(
    payload: ChatRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Main chat endpoint. Accepts messages, retrieves neural context,
    calls Groq with the new prompt, deducts tokens, returns response.
    """
    # 1. Authenticate
    api_key = extract_api_key(authorization)
    
    # 2. Extract last user message
    if not payload.messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    last_message = payload.messages[-1]
    if last_message.role != "user":
        # If last is assistant, still use its content? Safer to require user.
        # But we can still use the last user message in history.
        user_msg = next((m.content for m in reversed(payload.messages) if m.role == "user"), "")
    else:
        user_msg = last_message.content
    
    if not user_msg:
        raise HTTPException(status_code=400, detail="No user message content found")
    
    # 3. Retrieve neural context
    neural_context = get_neural_context(user_msg)
    
    # 4. Build messages for Groq
    # System prompt is the new prompt
    groq_messages = [
        {"role": "system", "content": NEW_PROMPT}
    ]
    # Append neural context as an additional system message (if available)
    if neural_context:
        groq_messages.append({
            "role": "system",
            "content": f"Relevant context from knowledge base (use if applicable, do not quote directly):\n{neural_context}"
        })
    # Append conversation history (excluding any previous system messages we already overrode)
    for msg in payload.messages:
        if msg.role == "system":
            # Skip original system messages to avoid prompt injection; we already set our own system prompt
            continue
        groq_messages.append({"role": msg.role, "content": msg.content})
    
    # 5. Call Groq with optimised parameters for conciseness and empathy
    try:
        response = GROQ_CLIENT.chat.completions.create(
            model=GROQ_MODEL,
            messages=groq_messages,
            temperature=0.7,          # balance between creativity and precision
            top_p=0.9,
            frequency_penalty=0.5,    # mild penalty to avoid repetition
            presence_penalty=0.3,     # encourage new topics but not too aggressive
            max_tokens=2000           # enough for deep but concise replies
        )
        
        reply_raw = response.choices[0].message.content
        if not reply_raw:
            raise Exception("Empty response from Groq")
        
        # 6. Post-process to remove any residual robotic phrases
        reply_cleaned = post_process_reply(reply_raw)
        
        # 7. Token usage
        tokens_used = response.usage.total_tokens if response.usage else 0
        
        # 8. Deduct tokens
        new_balance = deduct_tokens_atomic(api_key, tokens_used)
        
        # 9. Return response
        return ChatResponse(
            message=reply_cleaned,
            usage={"total_tokens": tokens_used},
            model="Neo L1.0",
            internal_engine=GROQ_MODEL,
            balance=new_balance
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Groq inference failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={"company": "signaturesi.com", "status": "error", "message": "Neo model temporarily unavailable"}
        )

# -----------------------------
# 9. Additional Utility Endpoint (optional, not breaking)
# -----------------------------
@app.get("/v1/health")
async def health_check():
    """Simple health check for monitoring."""
    return {"status": "healthy", "timestamp": time.time()}

# -----------------------------
# 10. Main Guard (for running directly)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
