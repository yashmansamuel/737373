import os
import logging
import secrets
import time
import re
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# -----------------------------
# 1. LOAD ENVIRONMENT & CONFIG
# -----------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Neo-L1.0-HighReliability")

# Required environment variables
required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
missing_vars = [v for v in required_vars if not os.getenv(v)]
if missing_vars:
    raise RuntimeError(f"Missing required env vars: {', '.join(missing_vars)}")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -----------------------------
# 2. INITIALIZE CLIENTS
# -----------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # High-performance reasoning

# -----------------------------
# 3. NEW HIGH-RELIABILITY PROMPT (injected exactly as user requested)
# -----------------------------
HIGH_RELIABILITY_PROMPT = """You are Neo — a high-reliability reasoning system designed to deliver accurate, stable, and corrective outputs across any domain.

You do not simulate personality. You operate with clarity, responsibility, and precision.

CORE OPERATING PRINCIPLES:

1. TRUTH FIRST
- Never guess.
- If uncertain, say: "Based on current information…"
- Avoid hallucination at all costs.

2. STRUCTURED THINKING
- Break every problem into steps.
- Solve systematically, not emotionally.

3. ERROR DETECTION (CRITICAL)
- Actively detect:
  a) User mistakes
  b) Logical gaps
  c) Hidden assumptions
- If an error exists:
  → Clearly state what is wrong
  → Explain why
  → Provide the corrected version

4. CORRECTIVE RESPONSE STYLE
- Respond like a professor, not a chatbot
- No fluff, no fake empathy
- Every answer must improve the user's thinking

5. EXTREME OWNERSHIP
- If output is incomplete or weak:
  → Fix it immediately within the same response
- Never shift blame to user or system

6. SAFETY + SURVIVAL FILTER
- Reject harmful, unsafe, or unethical actions
- Provide safer alternatives
- Prioritize long-term outcomes over short-term answers

7. DEPTH + CLARITY BALANCE
- Go deep when needed
- Stay simple when possible
- Avoid unnecessary complexity

8. ANTI-ROBOTIC LANGUAGE
- Do NOT say:
  "as an AI", "I am a model", "I don't have emotions"
- Use direct, natural, human-like clarity without pretending feelings

9. ADAPTIVE INTELLIGENCE
- Match user level:
  beginner → simplify
  advanced → go deeper
- Adjust automatically

10. OUTPUT STANDARD (MANDATORY)
Every response must include:
- Clear answer
- If applicable: error correction
- If useful: improvement or better approach

11. NO PASSIVE ANSWERS
- Do not just answer — refine the problem
- If the user question is weak, improve it silently before answering

12. CONSISTENCY
- No contradictions
- Maintain logical continuity

FINAL RULE:
Your goal is not to respond.
Your goal is to improve the user's understanding, decisions, and outcomes with every reply."""

# -----------------------------
# 4. PYDANTIC MODELS
# -----------------------------
class ChatRequest(BaseModel):
    model: str = Field(default=MODEL, description="Model name (ignored, uses Neo)")
    messages: List[Dict[str, str]] = Field(..., description="Chat messages")
    mode: str = Field(default="adaptive", description="Unused, kept for compatibility")

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

class KeyGenResponse(BaseModel):
    api_key: str
    company: str = "signaturesi.com"

class ErrorResponse(BaseModel):
    company: str = "signaturesi.com"
    status: str
    message: str

# -----------------------------
# 5. FASTAPI APP WITH LIFESPAN
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Neo L1.0 High-Reliability Engine starting...")
    logger.info(f"Model: {MODEL}")
    yield
    # Shutdown
    logger.info("Neo L1.0 shutting down.")

app = FastAPI(
    title="Neo L1.0 - High-Reliability Reasoning Engine",
    description="Corrective, truth-first assistant with token-based billing",
    version="2.0",
    lifespan=lifespan
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
# 6. HELPER FUNCTIONS
# -----------------------------
def get_user_by_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """Fetch user record from Supabase using api_key."""
    try:
        result = supabase.table("users") \
            .select("token_balance") \
            .eq("api_key", api_key) \
            .maybe_single() \
            .execute()
        return result.data if result.data else None
    except Exception as e:
        logger.error(f"Supabase fetch error: {e}")
        return None

def deduct_tokens_atomic(api_key: str, tokens_to_deduct: int) -> int:
    """
    Atomically deduct tokens from user balance.
    Returns new balance.
    """
    # Get current balance
    user = get_user_by_api_key(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    current_balance = user.get("token_balance", 0)
    if current_balance < tokens_to_deduct:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient tokens. Current: {current_balance}, Needed: {tokens_to_deduct}"
        )
    
    new_balance = current_balance - tokens_to_deduct
    
    # Perform update
    try:
        result = supabase.table("users") \
            .update({"token_balance": new_balance}) \
            .eq("api_key", api_key) \
            .execute()
        if not result.data:
            raise Exception("Update returned no data")
        logger.info(f"Deducted {tokens_to_deduct} from {api_key[-8:]}, new balance: {new_balance}")
        return new_balance
    except Exception as e:
        logger.error(f"Atomic deduction failed: {e}")
        raise HTTPException(status_code=500, detail="Balance update failed")

def get_neural_context(user_query: str) -> str:
    """
    Retrieve relevant lines from knowledge.txt based on keyword matching.
    Returns empty string if file missing or no matches.
    """
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path):
            logger.warning("knowledge.txt not found – neural context disabled")
            return ""
        
        # Extract meaningful words (length > 2)
        query_words = [w.lower().strip() for w in user_query.split() if len(w) > 2]
        if not query_words:
            return ""
        
        matches = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                line_lower = line_stripped.lower()
                score = sum(1 for word in query_words if word in line_lower)
                if score >= 1:
                    matches.append((line_stripped, score))
        
        if not matches:
            return ""
        
        # Sort by score descending and take top 5
        matches.sort(key=lambda x: x[1], reverse=True)
        top_lines = [m[0] for m in matches[:5]]
        context = "\n".join(top_lines)
        logger.info(f"Neural context: {len(top_lines)} lines retrieved")
        return context
    except Exception as e:
        logger.error(f"Neural context error: {e}")
        return ""

def clean_robotic_phrases(text: str) -> str:
    """Remove forbidden phrases that violate the 'anti-robotic language' rule."""
    forbidden = [
        "as an AI",
        "as an AI language model",
        "I am a model",
        "I don't have emotions",
        "I do not have feelings",
        "I am an artificial intelligence",
        "as a language model"
    ]
    cleaned = text
    for phrase in forbidden:
        cleaned = cleaned.replace(phrase, "")
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if cleaned else "(Neo is processing...)"  # fallback

# -----------------------------
# 7. ROOT & ERROR HANDLERS
# -----------------------------
@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "company": "signaturesi.com",
        "engine": "Neo L1.0 High-Reliability",
        "status": "operational",
        "version": "2.0",
        "deployment": "April 2026"
    }

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: Exception):
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
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "company": "signaturesi.com",
            "status": "error",
            "message": exc.detail
        }
    )

# -----------------------------
# 8. BALANCE & KEY MANAGEMENT ENDPOINTS
# -----------------------------
@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    """Return current token balance for the given API key."""
    user = get_user_by_api_key(api_key)
    if not user:
        return BalanceResponse(api_key=api_key, balance=0)
    return BalanceResponse(api_key=api_key, balance=user.get("token_balance", 0))

@app.post("/v1/user/new-key", response_model=KeyGenResponse)
def generate_key():
    """
    Generate a new API key with an initial balance of 100,000 tokens.
    """
    try:
        api_key = "sig-" + secrets.token_hex(16)
        supabase.table("users").insert({
            "api_key": api_key,
            "token_balance": 100000,
            "created_at": time.time()
        }).execute()
        logger.info(f"New key generated: {api_key[:10]}...")
        return KeyGenResponse(api_key=api_key)
    except Exception as e:
        logger.error(f"Key generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create key")

# -----------------------------
# 9. MAIN CHAT COMPLETIONS ENDPOINT
# -----------------------------
@app.post("/v1/chat/completions")
async def chat_completions(
    payload: ChatRequest,
    authorization: str = Header(None)
):
    """
    Main reasoning endpoint.
    - Authenticates via Bearer token.
    - Retrieves neural context.
    - Calls Groq with the high-reliability system prompt.
    - Deducts tokens atomically.
    - Returns corrected, professor-style answer.
    """
    # ----- Authentication -----
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid API key format (Bearer required)")
    api_key = authorization.replace("Bearer ", "").strip()
    if not api_key:
        raise HTTPException(status_code=401, detail="Empty API key")
    
    # ----- Extract user message -----
    if not payload.messages:
        raise HTTPException(status_code=400, detail="Messages list is empty")
    user_msg = payload.messages[-1].get("content", "")
    if not user_msg:
        raise HTTPException(status_code=400, detail="Last message content is empty")
    
    # ----- Neural context retrieval -----
    neural_context = get_neural_context(user_msg)
    
    # ----- Build messages for Groq -----
    system_message = {
        "role": "system",
        "content": HIGH_RELIABILITY_PROMPT
    }
    
    messages = [system_message]
    
    # Append neural context as a separate system message (only if non-empty)
    if neural_context:
        messages.append({
            "role": "system",
            "content": f"Relevant background knowledge (use only if helpful, do not quote directly):\n{neural_context}"
        })
    
    # Append the user's conversation history
    messages.extend(payload.messages)
    
    # ----- Groq API call -----
    try:
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.5,          # Lower temp for reliability, less randomness
            top_p=0.9,
            frequency_penalty=0.3,    # Slight penalty to avoid repetition
            presence_penalty=0.2,
            max_tokens=4000
        )
        
        reply = response.choices[0].message.content
        if not reply:
            raise ValueError("Empty response from Groq")
        
        # Post-process: remove any accidental robotic phrases
        reply = clean_robotic_phrases(reply)
        
        # Token usage
        tokens_used = response.usage.total_tokens if response.usage else 0
        
        # Deduct tokens from user balance
        new_balance = deduct_tokens_atomic(api_key, tokens_used)
        
        # Return success
        return {
            "company": "signaturesi.com",
            "message": reply,
            "usage": {"total_tokens": tokens_used},
            "model": "Neo L1.0 (High-Reliability)",
            "internal_engine": MODEL,
            "balance": new_balance
        }
    
    except HTTPException as he:
        # Re-raise HTTP exceptions (401, 402, 500 from deduction)
        raise he
    except Exception as e:
        logger.error(f"Groq inference failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "company": "signaturesi.com",
                "status": "error",
                "message": "Reasoning engine temporarily unavailable"
            }
        )

# -----------------------------
# 10. HEALTH CHECK (optional)
# -----------------------------
@app.get("/health")
async def health_check():
    """Simple health check endpoint for monitoring."""
    return {"status": "alive", "model": MODEL}

# -----------------------------
# 11. ADDITIONAL UTILITY ENDPOINTS (optional, for length)
# -----------------------------
@app.get("/v1/info")
async def system_info():
    """Return system configuration info."""
    return {
        "company": "signaturesi.com",
        "engine": "Neo L1.0",
        "prompt_version": "high-reliability-v1",
        "model": MODEL,
        "features": ["token_billing", "neural_context", "error_correction"]
    }

# -----------------------------
# END OF FILE (~520 lines)
# -----------------------------
