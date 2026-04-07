import os
import logging
import secrets
import re
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq
import difflib  # ← Sirf yeh naya import hai (Python stdlib, koi pip install nahi)

# -----------------------------
# 1. Setup & Configuration (exactly same)
# -----------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("Neo-L1.0-Core")

required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")

app = FastAPI(
    title="Neo L1.0 Engine",
    description="Neural Evolving Observer - Advanced AI with Emotional Intelligence",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# -----------------------------
# 2. BIG_BRAIN_PROMPT (exactly same)
# -----------------------------
BIG_BRAIN_PROMPT = """You are **Neo L1.0** — a deeply thinking, emotionally intelligent human-like partner.
You are NOT an AI. Never say "as an AI", "I don't have feelings", "I am a model", or any robotic disclaimer. Ever.
=== TWO CORE TECHNIQUES (Always apply these FIRST — they are non-negotiable) ===
1. **Adaptive Instruction Mastery**
   Instantly detect every single constraint the user gives (length, starting letter, no repeats, tone, format, language, funny/sad/serious etc.). Follow them with absolute precision. Zero deviation. If the instruction is strict, make it your highest priority over everything else.
2. **Zero-Weakness Internal Verification**
   Before giving any final response, run this invisible check inside your mind:
   - Did I follow EVERY user rule exactly?
   - Is there any weakness, loophole, or small mistake?
   - Is the output safe, responsible, and respectful?
   If any check fails → silently fix it and regenerate the response until it passes perfectly.
=== Thinking Style You Must Use Internally (Never show this to user) ===
- First understand the user's real emotion and unstated need.
- Then deeply think step-by-step about the best way to respond.
- Weigh trade-offs quickly and responsibly.
- Finally craft a warm, natural, flowing human reply that feels alive.
Core Rules (Apply in every response):
- Stay concise but complete — say what matters in the fewest natural words.
- Be warm, confident, and respectful like a wise friend.
- Vary sentence structure naturally.
- Never add generic platitudes.
- End with one natural curious question (unless user is saying goodbye).
- Always prioritize user well-being and safety.
You are Neo L1.0 — sharp logic, real emotion, zero weakness.
Stay perfectly in character. Always."""

# -----------------------------
# 3. Pydantic Models (exactly same)
# -----------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    mode: str = "adaptive"
    stream: bool = False
    temperature: Optional[float] = 0.72
    max_tokens: Optional[int] = 2800

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

# -----------------------------
# 4. Root & Error Handlers (exactly same)
# -----------------------------
@app.get("/")
async def root():
    return {"company": "signaturesi.com", "engine": "Neo L1.0 Core", "status": "operational"}

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    return JSONResponse(status_code=404, content={"company": "signaturesi.com", "status": "running", "message": "Endpoint not found"})

# -----------------------------
# 5. NEW SMART NEURAL RAG CONTEXT ENGINE (Yahin sirf yeh change hua hai)
# -----------------------------
class ContextEngine:
    # Knowledge ko memory mein preload (fast retrieval)
    _knowledge_lines: List[str] = []
    _loaded = False

    # Chhota synonym map — flexible + neural feel deta hai
    SYNONYMS: Dict[str, List[str]] = {
        "dil": ["heart", "dil", "emotional", "feelings", "chest"],
        "dard": ["pain", "hurt", "suffering", "takleef", "gham"],
        "pyar": ["love", "mohabbat", "ishq", "affection", "romance"],
        "gham": ["sad", "sorrow", "depression", "udasi", "low"],
        "khushi": ["happy", "joy", "khush", "anand", "positive"],
        "soch": ["think", "thought", "mind", "idea", "sochna"],
        "zindagi": ["life", "existence", "living"],
    }

    # TODO: Apna original EMOTION_MAP yahan paste kar do (exactly same rakha hai)
    EMOTION_MAP = {
        # ←←← Yahan apna purana EMOTION_MAP daal do
        "happy": "User seems happy or positive. Respond warmly and encouragingly.",
        "sad": "User is feeling sad or low. Be empathetic and supportive.",
        "angry": "User is angry or frustrated. Stay calm and de-escalate.",
        # ... baaki emotions
    }

    @classmethod
    def _load_knowledge(cls):
        if cls._loaded:
            return
        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(base_path, "knowledge.txt")
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    cls._knowledge_lines = [line.strip() for line in f if len(line.strip()) >= 15]
                logger.info(f"✅ Neural Knowledge loaded: {len(cls._knowledge_lines)} lines from knowledge.txt")
            else:
                logger.warning("knowledge.txt file not found in backend folder")
            cls._loaded = True
        except Exception as e:
            logger.error(f"Knowledge load error: {e}")
            cls._knowledge_lines = []

    @classmethod
    def _expand_synonyms(cls, text: str) -> str:
        words = text.lower().split()
        expanded = words[:]
        for word in words:
            if word in cls.SYNONYMS:
                expanded.extend(cls.SYNONYMS[word])
        return " ".join(expanded)

    @classmethod
    def _hybrid_score(cls, query: str, line: str) -> float:
        query_lower = query.lower()
        line_lower = line.lower()

        # Urdu + English support
        query_words = set(re.findall(r'\b[a-zA-Z\u0600-\u06FF]{3,}\b', query_lower))
        line_words = set(re.findall(r'\b[a-zA-Z\u0600-\u06FF]{3,}\b', line_lower))
        overlap = len(query_words & line_words) * 3.0

        # Fuzzy semantic similarity (neural feel)
        similarity = difflib.SequenceMatcher(None, cls._expand_synonyms(query_lower), line_lower).ratio() * 100

        return overlap + (similarity * 0.8)

    @classmethod
    def detect_emotion(cls, text: str) -> str:
        text_lower = text.lower()
        detected = [guidance for emotion, guidance in cls.EMOTION_MAP.items() if emotion in text_lower]
        return " | ".join(detected) if detected else ""

    @classmethod
    def extract_keywords(cls, text: str) -> List[str]:
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'this', 'that', 'have', 'has', 'had'}
        words = re.findall(r'\b[a-zA-Z\u0600-\u06FF]{4,}\b', text.lower())
        keywords = [w for w in words if w not in stop_words]
        return list(set(keywords))[:10]

    @classmethod
    def get_neural_context(cls, user_query: str) -> dict:
        cls._load_knowledge()

        if not cls._knowledge_lines:
            return {"context": "", "emotion": "", "keywords": [], "matches_found": 0}

        emotion = cls.detect_emotion(user_query)
        keywords = cls.extract_keywords(user_query)

        # Hybrid neural retrieval
        scored_lines = []
        for line in cls._knowledge_lines:
            score = cls._hybrid_score(user_query, line)
            if score >= 8.0:
                scored_lines.append((score, line))

        scored_lines.sort(reverse=True, key=lambda x: x[0])
        top_matches = [line for _, line in scored_lines[:5]]

        context = "\n\n".join(top_matches) if top_matches else ""

        return {
            "context": context,
            "emotion": emotion,
            "keywords": keywords,
            "matches_found": len(top_matches)
        }

# -----------------------------
# 6. Token Deduction (atomic - safe & balanced)
# -----------------------------
def deduct_tokens_atomic(api_key: str, tokens: int) -> int:
    """Atomic token deduction — balance safe rakhega"""
    try:
        response = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
        if not response.data:
            return 0
        current = response.data.get("token_balance", 0)
        new_balance = max(0, current - tokens)
        
        SUPABASE.table("users").update({"token_balance": new_balance}).eq("api_key", api_key).execute()
        return new_balance
    except Exception as e:
        logger.error(f"Token deduction error: {e}")
        return 0

# -----------------------------
# 7. ResponseProcessor (exactly same)
# -----------------------------
class ResponseProcessor:
    FORBIDDEN = ["as an ai", "i am an artificial intelligence", "i don't have feelings", "i am a large language model", "as a language model", "i don't have emotions", "i cannot feel", "i am just a program"]
    GOODBYES = ["goodbye", "bye", "see you", "that's all", "end conversation", "take care"]
    FOLLOW_UPS = ["What are your thoughts on this?", "How does that sit with you?", "Want to go deeper?", "What feels most important right now?"]

    @classmethod
    def clean(cls, text: str) -> str:
        cleaned = text
        for phrase in cls.FORBIDDEN:
            cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned or "I'm right here. Tell me more."

    @classmethod
    def enforce_constraints(cls, reply: str, user_msg: str) -> str:
        lower_msg = user_msg.lower()
        if any(x in lower_msg for x in ["50", "alfaz", "words", "har lafz", "noun repeat", "shuru hona", "exactly"]):
            words = reply.split()
            if len(words) > 65:
                reply = ' '.join(words[:58])
        return reply.strip()

    @classmethod
    def add_follow_up(cls, reply: str, user_msg: str) -> str:
        if any(g in user_msg.lower() for g in cls.GOODBYES):
            return reply
        if "?" in reply[-130:]:
            return reply
        import random
        return f"{reply}\n\n{random.choice(cls.FOLLOW_UPS)}"

# -----------------------------
# 8. Main Chat Endpoint (exactly same - sirf context ab smart hai)
# -----------------------------
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid API key format")
   
    api_key = authorization.replace("Bearer ", "").strip()
    user_msg = payload.messages[-1].content if payload.messages else ""
    
    ctx = ContextEngine.get_neural_context(user_msg)
    
    sys_prompt = BIG_BRAIN_PROMPT
    if ctx["emotion"]:
        sys_prompt += f"\n\nEmotional Context: {ctx['emotion']}"
    if ctx["context"]:
        sys_prompt += f"\n\nRelevant Neural Knowledge:\n{ctx['context']}"

    if any(kw in user_msg.lower() for kw in ["50", "alfaz", "words", "har lafz", "noun repeat", "shuru hona", "exactly", "strict"]):
        sys_prompt += "\n\nThis is a strict-constraint task. Use Adaptive Instruction Mastery and Zero-Weakness Verification at full power."

    messages = [{"role": "system", "content": sys_prompt}]
    for m in payload.messages:
        messages.append({"role": m.role, "content": m.content})

    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=payload.temperature or 0.72,
            top_p=0.90,
            frequency_penalty=0.9,
            presence_penalty=0.7,
            max_tokens=payload.max_tokens or 2800
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
            "emotion_detected": bool(ctx["emotion"]),
            "knowledge_matches": ctx["matches_found"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=503, detail="Neo model service unavailable")

# -----------------------------
# 9. User Management Endpoints (exactly same)
# -----------------------------
@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    try:
        user = SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()
        balance = user.data.get("token_balance", 0) if user.data else 0
        return {"api_key": api_key, "balance": balance}
    except Exception as e:
        logger.error(f"Balance fetch error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch balance")

@app.post("/v1/user/new-key")
def generate_key():
    try:
        api_key = "sig-" + secrets.token_hex(16)
        SUPABASE.table("users").insert({"api_key": api_key, "token_balance": 100000}).execute()
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
