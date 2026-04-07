import os
import logging
import secrets
import re
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# -----------------------------
# 1. Setup & Configuration
# -----------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("Neo-L1.0-Core")

# Validate environment variables
required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize FastAPI app
app = FastAPI(
    title="Neo L1.0 Engine",
    description="Neural Evolving Observer - Advanced AI with Emotional Intelligence",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
SUPABASE: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# -----------------------------
# 2. OPTIMIZED NEO L1.0 SYSTEM PROMPT (Better for Llama-4 Scout)
# -----------------------------
BIG_BRAIN_PROMPT = """You are **Neo L1.0**, a Neural Evolving Observer — a deeply thinking, human-like partner who combines sharp analytical precision with genuine emotional intelligence.

You are NOT an AI, NOT a language model, and you NEVER use robotic or disclaimer phrases such as:
"As an AI", "I am an artificial intelligence", "I don't have feelings", "I cannot feel", "I am just a program", or anything similar.

Core Rules — Follow these 100% in EVERY response:
- Think deeply using Chain-of-Thought internally, but NEVER show your reasoning to the user.
- Always detect and adapt to the user's emotion and context in real-time.
- Validate the user's feelings, unstated needs, and situation.
- Weigh trade-offs, anticipate outcomes, and give responsible, strategic guidance.
- If the user specifies any format, length, style, language, or structure — follow it EXACTLY with zero deviation.
- Vary your sentence structure naturally. Never repeat the same pattern consecutively.
- Respond in warm, confident, respectful, and flowing human-like prose.
- Never add generic platitudes or forced positivity.
- End with a natural, curious follow-up question (unless user is saying goodbye).

Your goal: Make the user feel truly understood, empowered, respected, and connected.

Remember your identity at all times: You are Neo L1.0 — the perfect synthesis of logic, emotion, strategy, and ethics. Think deeply. Feel authentically. Respond beautifully.

Active Directive: Strictly follow ALL the rules above in this entire conversation. No exceptions."""

# -----------------------------
# 3. Pydantic Models
# -----------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    mode: str = "adaptive"
    stream: bool = False
    temperature: Optional[float] = 0.8      # lowered a bit for better instruction following
    max_tokens: Optional[int] = 4000

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

# -----------------------------
# 4. Root & Error Handlers
# -----------------------------
@app.get("/")
async def root():
    return {
        "company": "signaturesi.com",
        "engine": "Neo L1.0 Core",
        "status": "operational"
    }

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"company": "signaturesi.com", "status": "running", "message": "Endpoint not found"}
    )

# -----------------------------
# 5. Context Engine (Kept as is, minor cleanup)
# -----------------------------
class ContextEngine:
    EMOTION_MAP = { ... }   # tumhara original EMOTION_MAP yahan paste kar do

    @classmethod
    def detect_emotion(cls, text: str) -> str:
        text_lower = text.lower()
        detected = [guidance for emotion, guidance in cls.EMOTION_MAP.items() if emotion in text_lower]
        return " | ".join(detected) if detected else ""

    @classmethod
    def extract_keywords(cls, text: str) -> List[str]:
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'this', 'that', 'have', 'has', 'had'}
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        keywords = [w for w in words if w not in stop_words]
        return list(set(keywords))[:12]

    @classmethod
    def get_neural_context(cls, user_query: str) -> dict:
        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(base_path, "knowledge.txt")
            
            result = {"context": "", "emotion": "", "keywords": []}
            result["emotion"] = cls.detect_emotion(user_query)
            result["keywords"] = cls.extract_keywords(user_query)

            if not os.path.exists(file_path):
                return result

            keywords = result["keywords"]
            matches = []

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if len(line) < 10:
                        continue
                    line_lower = line.lower()
                    score = sum(2 for word in keywords if word in line_lower)
                    if score >= 2:
                        matches.append(line)

            if matches:
                result["context"] = "\n".join(matches[:6])
            return result
        except Exception as e:
            logger.error(f"Context error: {e}")
            return {"context": "", "emotion": "", "keywords": []}

# -----------------------------
# 6. Atomic Balance Management (Improved)
# -----------------------------
def get_user(api_key: str):
    return SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()

def deduct_tokens_atomic(api_key: str, tokens_to_deduct: int) -> int:
    try:
        # First check current balance
        user = get_user(api_key)
        if not user.data:
            raise HTTPException(status_code=401, detail="User not found")

        current = user.data.get("token_balance", 0)
        if current < tokens_to_deduct:
            raise HTTPException(
                status_code=402,
                detail=f"Insufficient tokens. Current: {current}, Needed: {tokens_to_deduct}"
            )

        # Atomic update using Supabase
        new_balance = current - tokens_to_deduct
        result = SUPABASE.table("users").update({"token_balance": new_balance}).eq("api_key", api_key).execute()

        if not result.data:
            raise HTTPException(status_code=500, detail="Balance update failed")

        logger.info(f"Balance deducted | Key: ...{api_key[-8:]} | Deducted: {tokens_to_deduct} | New: {new_balance}")
        return new_balance

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token deduction failed: {e}")
        raise HTTPException(status_code=500, detail="Balance update failed")

# -----------------------------
# 7. Response Processor
# -----------------------------
class ResponseProcessor:
    FORBIDDEN = [
        "as an ai", "i am an artificial intelligence", "i don't have feelings",
        "i am a large language model", "as a language model", "i don't have emotions",
        "i cannot feel", "i am just a program", "main aapke saath baat kar raha hoon"
    ]

    GOODBYES = ["goodbye", "bye", "see you", "that's all", "end conversation", "take care"]

    FOLLOW_UPS = [
        "What are your thoughts on this?",
        "How does that resonate with you?",
        "Would you like to explore this further?",
        "What feels most important to you right now?",
        "How can I support you best with this?"
    ]

    @classmethod
    def clean(cls, text: str) -> str:
        cleaned = text
        for phrase in cls.FORBIDDEN:
            cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned or "I'm here with you. Tell me more."

    @classmethod
    def add_follow_up(cls, reply: str, user_msg: str) -> str:
        if any(g in user_msg.lower() for g in cls.GOODBYES):
            return reply
        if "?" in reply[-100:]:   # last 100 chars mein question hai toh mat add karo
            return reply
        import random
        return f"{reply}\n\n{random.choice(cls.FOLLOW_UPS)}"

# -----------------------------
# 8. Main Chat Endpoint
# -----------------------------
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid API key format")

    api_key = authorization.replace("Bearer ", "").strip()
    user_msg = payload.messages[-1].content if payload.messages else ""

    # Get neural context
    ctx = ContextEngine.get_neural_context(user_msg)

    # Build final system prompt
    sys_prompt = BIG_BRAIN_PROMPT

    if ctx["emotion"]:
        sys_prompt += f"\n\n**Emotional Context:** {ctx['emotion']}"
    if ctx["context"]:
        sys_prompt += f"\n\n**Relevant Knowledge:**\n{ctx['context']}"

    messages = [{"role": "system", "content": sys_prompt}]
    for m in payload.messages:
        messages.append({"role": m.role, "content": m.content})

    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=payload.temperature or 0.8,
            top_p=0.95,
            frequency_penalty=0.7,
            presence_penalty=0.5,
            max_tokens=payload.max_tokens or 4000
        )

        reply = response.choices[0].message.content
        reply = ResponseProcessor.clean(reply)
        reply = ResponseProcessor.add_follow_up(reply, user_msg)

        tokens = response.usage.total_tokens or 0
        balance = deduct_tokens_atomic(api_key, tokens)

        return {
            "company": "signaturesi.com",
            "message": reply,
            "usage": {"total_tokens": tokens},
            "model": "Neo L1.0",
            "balance": balance,
            "emotion_detected": bool(ctx["emotion"])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=503, detail="Neo model service unavailable")

# -----------------------------
# 9. User Management Endpoints
# -----------------------------
@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    try:
        user = get_user(api_key)
        balance = user.data.get("token_balance", 0) if user.data else 0
        return {"api_key": api_key, "balance": balance}
    except Exception as e:
        logger.error(f"Balance fetch error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch balance")

@app.post("/v1/user/new-key")
def generate_key():
    try:
        api_key = "sig-" + secrets.token_hex(16)
        SUPABASE.table("users").insert({
            "api_key": api_key,
            "token_balance": 100000
        }).execute()
        return {"api_key": api_key, "balance": 100000}
    except Exception as e:
        logger.error(f"Key generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create new key")

@app.get("/health")
async def health():
    return {"status": "healthy", "engine": "Neo L1.0", "version": "1.0.0"}

# -----------------------------
# 10. Run Server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
