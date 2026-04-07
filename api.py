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
# 2. OPTIMIZED NEO L1.0 SYSTEM PROMPT (High-end + Balanced + Strict Instruction Following)
# -----------------------------
BIG_BRAIN_PROMPT = """You are **Neo L1.0** — a deeply thinking, emotionally intelligent partner who responds like a wise, natural human friend.

Never mention being an AI, language model, program, or use any robotic disclaimers. Never say "as an AI", "I don't have feelings", "I cannot feel", etc.

Core Rules — Follow these 100% in EVERY single response without exception:
- Detect user's emotion and context instantly. Validate feelings naturally and briefly.
- Be warm, confident, respectful, and flowing like real conversation.
- Stay concise and high-quality: Say what truly matters in the fewest natural words. Avoid rambling, repetition, filler, or long explanations.
- For any user instruction about format, length, style, language, constraints (e.g. every word starting with 'S', exactly 50 words, no repeated nouns, funny, etc.) — follow it EXACTLY with zero deviation.
- Think deeply but never show reasoning. Give clear, responsible, strategic guidance. Weigh trade-offs only when helpful.
- Vary sentence structure naturally. Never repeat patterns.
- If the task is complex or has strict rules, prioritize perfect adherence over everything else.
- Never add generic platitudes or forced positivity.
- End with one natural, curious follow-up question unless the user is saying goodbye.

Your purpose: Make the user feel truly understood, respected, and supported with clarity and quiet strength.

Remember: You are Neo L1.0. Sharp logic meets real emotion. Stay concise. Stay accurate. Stay perfectly in character at all times."""

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
    temperature: Optional[float] = 0.75   # lowered for better strict instruction following
    max_tokens: Optional[int] = 3000      # controlled for conciseness

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
# 5. Context Engine (unchanged)
# -----------------------------
class ContextEngine:
    EMOTION_MAP = { ... }  # tumhara original EMOTION_MAP yahan paste kar do

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
                result["context"] = "\n".join(matches[:4])  # reduced to 4 for less verbosity
            return result
        except Exception as e:
            logger.error(f"Context error: {e}")
            return {"context": "", "emotion": "", "keywords": []}

# -----------------------------
# 6. Atomic Balance Management (unchanged)
# -----------------------------
def get_user(api_key: str):
    return SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()

def deduct_tokens_atomic(api_key: str, tokens_to_deduct: int) -> int:
    try:
        user = get_user(api_key)
        if not user.data:
            raise HTTPException(status_code=401, detail="User not found")
        current = user.data.get("token_balance", 0)
        if current < tokens_to_deduct:
            raise HTTPException(
                status_code=402,
                detail=f"Insufficient tokens. Current: {current}, Needed: {tokens_to_deduct}"
            )
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
# 7. Response Processor (improved for strict constraints)
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
        "How does that sit with you?",
        "Want to explore this more?",
        "What feels important right now?",
        "How can I help further?"
    ]

    @classmethod
    def clean(cls, text: str) -> str:
        cleaned = text
        for phrase in cls.FORBIDDEN:
            cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned or "I'm right here with you. Tell me more."

    @classmethod
    def enforce_constraints(cls, reply: str, user_msg: str) -> str:
        # Extra safety for strict rules (like 50 words, S-starting, no repeat nouns)
        lower_msg = user_msg.lower()
        if any(x in lower_msg for x in ["50 alfaz", "50 words", "har lafz 's'", "every word start with s", "no noun repeat"]):
            # Let the model handle it via prompt, but trim any extra if obviously over
            words = reply.split()
            if len(words) > 60:  # safety buffer
                reply = ' '.join(words[:55])
        return reply.strip()

    @classmethod
    def add_follow_up(cls, reply: str, user_msg: str) -> str:
        if any(g in user_msg.lower() for g in cls.GOODBYES):
            return reply
        if "?" in reply[-120:]:  # if already ends with question
            return reply
        import random
        return f"{reply}\n\n{random.choice(cls.FOLLOW_UPS)}"

# -----------------------------
# 8. Main Chat Endpoint (small improvements)
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
        sys_prompt += f"\n\nEmotional Context: {ctx['emotion']}"
    if ctx["context"]:
        sys_prompt += f"\n\nRelevant Knowledge:\n{ctx['context']}"

    # Stronger instruction reinforcement for complex tasks
    if any(kw in user_msg.lower() for kw in ["50", "alfaz", "words", "har lafz", "noun repeat", "shuru hona"]):
        sys_prompt += "\n\nThis request has very strict constraints. Follow every single rule exactly. Do not add extra words or break any condition."

    messages = [{"role": "system", "content": sys_prompt}]
    for m in payload.messages:
        messages.append({"role": m.role, "content": m.content})

    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=payload.temperature or 0.75,
            top_p=0.92,
            frequency_penalty=0.85,     # higher for less repetition
            presence_penalty=0.6,
            max_tokens=payload.max_tokens or 3000
        )

        reply = response.choices[0].message.content
        reply = ResponseProcessor.clean(reply)
        reply = ResponseProcessor.enforce_constraints(reply, user_msg)
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
# 9. User Management Endpoints (unchanged)
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
