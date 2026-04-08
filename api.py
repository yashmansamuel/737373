import os
import logging
import secrets
import re
import requests
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq
import difflib

# -----------------------------
# 1. Setup
# -----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-L1.0")

required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
if any(not os.getenv(v) for v in required_vars):
    raise RuntimeError("Missing env variables")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# -----------------------------
# 2. Prompt
# -----------------------------
BIG_BRAIN_PROMPT = "You are Neo L1.0 — emotionally intelligent, human-like, precise."

# -----------------------------
# 3. Models
# -----------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2000

# -----------------------------
# 4. Wikipedia + Cache
# -----------------------------
WIKI_CACHE: Dict[str, str] = {}

def fetch_wikipedia(query: str) -> str:
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "%20")
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            return res.json().get("extract", "")
        return ""
    except:
        return ""

def smart_filter(text: str, query: str) -> str:
    sentences = text.split(". ")
    query_words = set(query.lower().split())

    scored = []
    for s in sentences:
        overlap = len(query_words & set(s.lower().split()))
        if overlap > 0:
            scored.append((overlap, s))

    scored.sort(reverse=True)
    return ". ".join([s for _, s in scored[:3]])

def get_wiki_context(query: str) -> str:
    if query in WIKI_CACHE:
        return WIKI_CACHE[query]

    raw = fetch_wikipedia(query)
    filtered = smart_filter(raw, query)

    WIKI_CACHE[query] = filtered
    return filtered

# -----------------------------
# 5. Context Engine
# -----------------------------
class ContextEngine:
    _knowledge_lines: List[str] = []
    _loaded = False

    @classmethod
    def load_knowledge(cls):
        if cls._loaded:
            return
        path = os.path.join(os.getcwd(), "knowledge.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cls._knowledge_lines = [l.strip() for l in f if len(l.strip()) > 20]
        cls._loaded = True

    @classmethod
    def score(cls, query: str, line: str) -> float:
        return difflib.SequenceMatcher(None, query.lower(), line.lower()).ratio()

    @classmethod
    def get_context(cls, query: str):
        cls.load_knowledge()

        scored = [(cls.score(query, l), l) for l in cls._knowledge_lines]
        scored.sort(reverse=True)

        top = [l for s, l in scored[:5] if s > 0.2]

        return {
            "context": "\n".join(top),
            "matches": len(top)
        }

# -----------------------------
# 6. Chat Endpoint
# -----------------------------
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(401, "Missing API key")

    user_msg = payload.messages[-1].content

    ctx = ContextEngine.get_context(user_msg)

    wiki = ""
    if ctx["matches"] < 2:
        wiki = get_wiki_context(user_msg)

    sys_prompt = BIG_BRAIN_PROMPT

    if ctx["context"]:
        sys_prompt += "\n\nLocal Knowledge:\n" + ctx["context"]

    if wiki:
        sys_prompt += "\n\nWikipedia:\n" + wiki

    messages = [{"role": "system", "content": sys_prompt}]
    messages += [{"role": m.role, "content": m.content} for m in payload.messages]

    try:
        res = GROQ.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens
        )

        return {
            "message": res.choices[0].message.content,
            "usage": res.usage.total_tokens,
            "wiki_used": bool(wiki),
            "local_matches": ctx["matches"]
        }

    except Exception as e:
        logger.error(e)
        raise HTTPException(500, "Model error")

# -----------------------------
# 7. Startup
# -----------------------------
@app.on_event("startup")
async def startup():
    ContextEngine.load_knowledge()

# -----------------------------
# 8. Run
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
