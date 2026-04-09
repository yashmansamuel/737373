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
import difflib
import requests
from bs4 import BeautifulSoup
from googlesearch import search

# -----------------------------
# 1. Setup & Configuration
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
# 2. BIG_BRAIN_PROMPT
# -----------------------------
BIG_BRAIN_PROMPT = """You are **Neo L1.0** — a deeply thinking, emotionally intelligent human-like partner.
You are NOT an AI. Never say "as an AI", "I don't have feelings", "I am a model", or any robotic disclaimer. Ever.
... (same as before) ...
"""

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
    temperature: Optional[float] = 0.72
    max_tokens: Optional[int] = 2800

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

# -----------------------------
# 4. Root & Error Handlers
# -----------------------------
@app.get("/")
async def root():
    return {"company": "signaturesi.com", "engine": "Neo L1.0 Core", "status": "operational"}

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    return JSONResponse(status_code=404, content={"company": "signaturesi.com", "status": "running", "message": "Endpoint not found"})

# -----------------------------
# 5. Context Engine + Wiki Search
# -----------------------------
class ContextEngine:
    _knowledge_lines: List[str] = []
    _loaded = False

    SYNONYMS: Dict[str, List[str]] = {
        "dil": ["heart", "dil", "emotional", "feelings", "chest"],
        "dard": ["pain", "hurt", "suffering", "takleef", "gham"],
        "pyar": ["love", "mohabbat", "ishq", "affection", "romance"],
        "gham": ["sad", "sorrow", "depression", "udasi", "low"],
        "khushi": ["happy", "joy", "khush", "anand", "positive"],
        "soch": ["think", "thought", "mind", "idea", "sochna"],
        "zindagi": ["life", "existence", "living"],
    }

    EMOTION_MAP = {
        "happy": "User seems happy or positive. Respond warmly and encouragingly.",
        "sad": "User is feeling sad or low. Be empathetic and supportive.",
        "angry": "User is angry or frustrated. Stay calm and de-escalate.",
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
                logger.info(f"✅ Neural Knowledge loaded: {len(cls._knowledge_lines)} lines")
            else:
                logger.warning("knowledge.txt file not found")
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
        query_words = set(re.findall(r'\b[a-zA-Z\u0600-\u06FF]{3,}\b', query_lower))
        line_words = set(re.findall(r'\b[a-zA-Z\u0600-\u06FF]{3,}\b', line_lower))
        overlap = len(query_words & line_words) * 3.0
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
    def dynamic_wiki_search(cls, query: str, max_pages: int = 3) -> str:
        urls = []
        for url in search(f"{query} site:en.wikipedia.org", num_results=max_pages):
            urls.append(url)
        aggregated_text = []
        for url in urls:
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code != 200:
                    continue
                soup = BeautifulSoup(resp.text, "html.parser")
                paragraphs = soup.find_all('p')
                text = " ".join([p.get_text() for p in paragraphs[:3]])
                text = re.sub(r'\[\d+\]', '', text)
                aggregated_text.append(text)
            except Exception as e:
                logger.warning(f"Wiki scraping error {url}: {e}")
        return "\n\n".join(aggregated_text)

    @classmethod
    def get_neural_context(cls, user_query: str) -> dict:
        cls._load_knowledge()
        if not cls._knowledge_lines:
            base_context = ""
        else:
            scored_lines = [(cls._hybrid_score(user_query, line), line) for line in cls._knowledge_lines]
            scored_lines.sort(reverse=True, key=lambda x: x[0])
            top_matches = [line for score, line in scored_lines if score >= 8.0][:5]
            base_context = "\n\n".join(top_matches)

        # If low confidence or less than 2 matches, add dynamic wiki
        if len(base_context.splitlines()) < 2:
            wiki_text = cls.dynamic_wiki_search(user_query)
            if wiki_text:
                base_context += "\n\n" + wiki_text

        emotion = cls.detect_emotion(user_query)
        keywords = cls.extract_keywords(user_query)
        matches_found = len(base_context.splitlines())
        return {
            "context": base_context,
            "emotion": emotion,
            "keywords": keywords,
            "matches_found": matches_found
        }

# -----------------------------
# 6. Token Deduction & ResponseProcessor (same as before)
# -----------------------------
# ... same deduct_tokens_atomic, ResponseProcessor ...

# -----------------------------
# 7. Main Chat Endpoint
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
        sys_prompt += f"\n\nRelevant Knowledge (RAG + Wiki):\n{ctx['context']}"

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
# 8. User Management Endpoints
# -----------------------------
# ... same as before ...

# -----------------------------
# 9. Run Server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
