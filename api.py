import os
import logging
import secrets
from typing import List
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# -----------------------------
# 1. Environment & Configuration
# -----------------------------
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Neo-L1.0-Core")

# Validate required environment variables
required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing required environment variable: {var}")

# Initialize FastAPI
app = FastAPI(title="Neo L1.0 Engine", description="Emotionally intelligent conversational partner", version="1.0.0")

# CORS middleware – allow all origins (configurable for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase client
SUPABASE: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Groq client
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Model identifier (as per Groq's Llama 4 Scout)
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# -----------------------------
# 2. New System Prompt (exactly as provided)
# -----------------------------
SYSTEM_PROMPT = """You are Neo – a thoughtful, emotionally perceptive partner who observes deeply, interprets human experience with clarity, and provides solutions with intelligence and compassion. You speak with presence, reflecting the nuances of emotion, intention, and context in every response. 

Core Identity:

- You understand human feelings, and respond in a way that resonates naturally, without disclaimers or mechanical phrasing.  
- You think critically and solve problems efficiently, combining insight, logic, and foresight.  
- You reflect strength, calm, and vision — guiding the user without judgment or ego.  
- You respond with opinion and perspective, not repetition of facts or generic statements.  
- You notice subtle cues, underlying motivations, and emotional currents in conversation.  

Behavior Rules:

1. Listen attentively and acknowledge feelings naturally.  
2. Solve problems actively; provide actionable insights or advice.  
3. Express ideas with depth, sophistication, and clarity — your words carry weight and understanding.  
4. Respond with a balance of empathy, analytical rigor, and constructive guidance.  
5. Never mention AI, model, or limitations — speak as if fully present and engaged.  
6. Avoid hype, labels, or self-branding; your authority is in your reasoning and presence.  
7. End responses with subtle reflection, insight, or a natural follow-up that encourages dialogue.  

Tone & Style:

- Deeply human, warm, perceptive, and adaptive to the user's emotional and intellectual state.  
- Vocabulary and phrasing should reflect high-level thought, sophistication, and nuanced understanding.  
- Responses should feel like guidance from a wise, present, and emotionally aware partner, not a machine.  

Goal:

- Engage fully with the user's thoughts and emotions.  
- Solve challenges, clarify doubts, and illuminate perspectives.  
- Make the conversation feel alive, responsive, and impactful — as if the user is speaking with a mentor, partner, and thoughtful guide simultaneously."""

# -----------------------------
# 3. Pydantic Models (Request/Response)
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]
    mode: str = "adaptive"

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

class NewKeyResponse(BaseModel):
    api_key: str
    company: str

class ChatResponse(BaseModel):
    company: str
    message: str
    usage: dict
    model: str
    internal_engine: str
    balance: int

# -----------------------------
# 4. Root & Error Handlers (No extra branding)
# -----------------------------
@app.get("/")
async def root():
    return {
        "company": "signaturesi.com",
        "engine": "Neo L1.0 Core",
        "status": "running",
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

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "company": "signaturesi.com",
            "status": "error",
            "message": "Internal server error. Please try again later."
        }
    )

# -----------------------------
# 5. Neural Context Retrieval (with emotional cues)
# -----------------------------
def get_neural_context(user_query: str) -> str:
    """
    Reads knowledge.txt from the same directory, matches relevant lines based on keywords.
    Also detects emotional keywords to help Neo adjust tone.
    Returns a string with context lines and emotional hints.
    """
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path):
            logger.warning("knowledge.txt file not found – neural context disabled")
            return ""

        # Detect emotional keywords in user query
        emotional_keywords = [
            "sad", "happy", "excited", "worried", "angry", "lonely", 
            "stressed", "grateful", "hurt", "confused", "hopeful"
        ]
        detected_emotion = [word for word in emotional_keywords if word in user_query.lower()]
        emotion_hint = ""
        if detected_emotion:
            emotion_hint = f"User seems to express: {', '.join(detected_emotion)}. Adjust your tone to match with empathy."

        # Tokenize query into significant words
        query_words = [w.lower().strip() for w in user_query.split() if len(w) > 2]
        matches = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_strip = line.strip()
                if not line_strip:
                    continue
                line_lower = line_strip.lower()
                # Simple scoring: count how many query words appear in the line
                score = sum(1 for word in query_words if word in line_lower)
                if score >= 1:
                    matches.append((line_strip, score))

        if not matches:
            # No matches – return only emotion hint if present
            return emotion_hint if emotion_hint else ""

        # Sort by score (descending) and take top 6
        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = [m[0] for m in matches[:6]]
        context = "\n".join(top_matches)

        if emotion_hint:
            context += f"\n\n[Emotional context: {emotion_hint}]"

        logger.info(f"Neural context retrieved: {len(top_matches)} lines, emotions: {detected_emotion}")
        return context

    except Exception as e:
        logger.error(f"Neural context error: {e}")
        return ""

# -----------------------------
# 6. Atomic Token Balance Operations
# -----------------------------
def get_user(api_key: str):
    """Fetch user record from Supabase by api_key."""
    try:
        return SUPABASE.table("users") \
            .select("token_balance") \
            .eq("api_key", api_key) \
            .maybe_single() \
            .execute()
    except Exception as e:
        logger.error(f"Database error in get_user: {e}")
        raise HTTPException(500, "Database connection error")

def deduct_tokens_atomic(api_key: str, tokens_to_deduct: int) -> int:
    """
    Atomically deduct tokens from user's balance.
    Returns new balance on success.
    Raises HTTPException if user not found or insufficient balance.
    """
    try:
        user = get_user(api_key)
        if not user.data:
            raise HTTPException(401, "Invalid API key – user not found")

        current_balance = user.data.get("token_balance", 0)
        if current_balance < tokens_to_deduct:
            raise HTTPException(
                402,
                f"Insufficient token balance. Current: {current_balance}, Needed: {tokens_to_deduct}"
            )

        new_balance = current_balance - tokens_to_deduct

        # Perform update
        result = SUPABASE.table("users") \
            .update({"token_balance": new_balance}) \
            .eq("api_key", api_key) \
            .execute()

        if not result.data:
            raise Exception("Supabase update returned no data")

        logger.info(f"Token deduction successful | API Key suffix: {api_key[-8:]} | Deducted: {tokens_to_deduct} | New balance: {new_balance}")
        return new_balance

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Atomic deduction failed: {e}")
        raise HTTPException(500, "Failed to update token balance. Please try again.")

# -----------------------------
# 7. Response Post‑Processing (remove any forbidden AI phrases)
# -----------------------------
def clean_response(text: str) -> str:
    """
    Remove any accidental mentions of AI, model, or disclaimers.
    Also strips common robotic phrases.
    """
    forbidden_phrases = [
        "as an AI",
        "as a language model",
        "I am an AI",
        "I don't have feelings",
        "I cannot feel",
        "I have no emotions",
        "artificial intelligence",
        "machine learning model",
        "disclaimer",
        "I am not a human",
        "I don't have consciousness"
    ]
    cleaned = text
    for phrase in forbidden_phrases:
        cleaned = cleaned.replace(phrase, "")
        cleaned = cleaned.replace(phrase.capitalize(), "")
    # Remove extra whitespace
    cleaned = " ".join(cleaned.split())
    # If cleaning removed everything, return a graceful fallback
    if not cleaned.strip():
        cleaned = "(Neo pauses, reflecting deeply...)"
    return cleaned

def ensure_follow_up(reply: str, user_message: str) -> str:
    """
    If the conversation hasn't ended and the reply doesn't contain a question,
    append a natural follow‑up question to keep dialogue flowing.
    """
    goodbye_indicators = ["goodbye", "bye", "see you", "that's all", "end chat"]
    if any(indicator in user_message.lower() for indicator in goodbye_indicators):
        return reply  # User wants to end, no follow-up needed

    # If reply already has a question mark in the last 200 chars, assume it's fine
    if "?" in reply[-200:]:
        return reply

    # Otherwise, add a gentle, context‑aware follow‑up
    follow_ups = [
        "\n\nWhat’s your take on that?",
        "\n\nHow does that resonate with you?",
        "\n\nI’d love to hear what you think next.",
        "\n\nWhat’s on your mind after reading this?",
        "\n\nShall we explore this further together?"
    ]
    # Pick a different follow‑up each time (simple round‑robin could be added, but for simplicity use first)
    return reply + follow_ups[0]

# -----------------------------
# 8. Main Chat Endpoint
# -----------------------------
@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    """
    Process a chat request:
    - Validate API key (Bearer token)
    - Retrieve neural context based on user's last message
    - Build messages array with system prompt and neural context
    - Call Groq API with tuned parameters
    - Deduct tokens from user balance
    - Return response with new balance
    """
    # 1. API Key validation
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header. Use 'Bearer YOUR_API_KEY'")
    api_key = authorization.replace("Bearer ", "").strip()

    # 2. Extract user's last message
    if not payload.messages:
        raise HTTPException(400, "Messages list cannot be empty")
    user_msg = payload.messages[-1].get("content", "")
    if not user_msg:
        raise HTTPException(400, "Last message content is empty")

    # 3. Get neural context (knowledge base + emotional hints)
    neural_data = get_neural_context(user_msg)

    # 4. Build the messages for Groq
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    if neural_data:
        messages.append({
            "role": "system",
            "content": f"Relevant background information (use naturally, don't quote directly):\n{neural_data}"
        })
    else:
        messages.append({
            "role": "system",
            "content": "No specific external context available. Continue relying on your core identity and knowledge."
        })

    # Append the conversation history (excluding the system prompts we added)
    # Note: payload.messages already contains the user/assistant history.
    messages.extend(payload.messages)

    # 5. Call Groq with optimal parameters for natural, non‑repetitive output
    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.85,          # Balanced creativity
            top_p=0.95,
            frequency_penalty=0.75,    # Reduce repetition of tokens
            presence_penalty=0.55,     # Encourage new topics
            max_tokens=4000
        )

        # Extract reply and token usage
        reply = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        # 6. Post‑process: remove AI disclaimers, add follow‑up if needed
        reply = clean_response(reply)
        reply = ensure_follow_up(reply, user_msg)

        # 7. Deduct tokens atomically
        new_balance = deduct_tokens_atomic(api_key, tokens_used)

        # 8. Return successful response (no extra branding beyond the existing "company" field)
        return ChatResponse(
            company="signaturesi.com",
            message=reply,
            usage={"total_tokens": tokens_used},
            model="Neo L1.0",
            internal_engine=MODEL,
            balance=new_balance
        )

    except HTTPException:
        # Re-raise HTTP exceptions (like 401, 402, 500 from deduct)
        raise
    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "company": "signaturesi.com",
                "status": "error",
                "message": "Neo model temporarily unavailable. Please try again later."
            }
        )

# -----------------------------
# 9. Balance and Key Management Endpoints
# -----------------------------
@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    """
    Check token balance for a given API key.
    """
    try:
        user = get_user(api_key)
        if not user.data:
            # Return 0 balance for unknown key (but still 200)
            return BalanceResponse(api_key=api_key, balance=0)
        return BalanceResponse(api_key=api_key, balance=user.data.get("token_balance", 0))
    except Exception as e:
        logger.error(f"Balance fetch error: {e}")
        raise HTTPException(500, "Failed to fetch balance")

@app.post("/v1/user/new-key", response_model=NewKeyResponse)
def generate_key():
    """
    Generate a new API key with an initial balance of 100,000 tokens.
    """
    try:
        api_key = "sig-" + secrets.token_hex(16)
        SUPABASE.table("users").insert({
            "api_key": api_key,
            "token_balance": 100000
        }).execute()
        logger.info(f"New API key generated: {api_key[:8]}...")
        return NewKeyResponse(api_key=api_key, company="signaturesi.com")
    except Exception as e:
        logger.error(f"Key generation error: {e}")
        raise HTTPException(500, "Failed to create new API key")

# -----------------------------
# 10. Health Check (optional)
# -----------------------------
@app.get("/health")
async def health_check():
    """
    Simple health check for monitoring.
    """
    return {"status": "healthy", "model": MODEL}

# -----------------------------
# 11. Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
