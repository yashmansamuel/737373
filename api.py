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
# 1. ENV + LOGGING
# -----------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("Neo-Pro")

REQUIRED = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
missing = [v for v in REQUIRED if not os.getenv(v)]
if missing:
    raise RuntimeError(f"Missing ENV: {missing}")

# -----------------------------
# 2. APP INIT
# -----------------------------
app = FastAPI()

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
# 3. STRONG SYSTEM PROMPT
# -----------------------------
BASE_PROMPT = """
You are Neo L1.0.

STRICT RULES:
- ALWAYS use provided context (Local Knowledge + Wikipedia)
- NEVER say "knowledge cutoff"
- NEVER rely on your own hidden knowledge
- If context exists → it is the truth
- Answer naturally like a human

Stay precise, clear, and confident.
"""

# -----------------------------
# 4. REQUEST MODELS
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
# 5. WIKIPEDIA ENGINE
# -----------------------------
WIKI_CACHE: Dict[str, str] = {}

def clean_query(q: str) -> str:
    q = re.sub(r"[^a-zA-Z0-9\s]", "", q)
    return " ".join(q.split()[:3])

def fetch_wikipedia(query: str) -> str:
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '%20')}"
        res = requests.get(url, timeout=3)

        if res.status_code == 200:
            return res.json().get("extract", "")

        # fallback search
        search_api = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json"
        }

        r = requests.get(search_api, params=params, timeout=3)
        data = r.json()

        if data.get("query", {}).get("search"):
            title = data["query"]["search"][0]["title"]
            return fetch_wikipedia(title)

        return ""
    except Exception as e:
        logger.error(f"WIKI ERROR: {e}")
        return ""

def smart_filter(text: str, query: str) -> str:
    sentences = text.split(". ")
    q_words = set(query.lower().split())

    scored = []
    for s in sentences:
        overlap = len(q_words & set(s.lower().split()))
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
    logger.info(f"WIKI USED: {query}")
    return filtered

# -----------------------------
# 6. LOCAL RAG ENGINE
# -----------------------------
class ContextEngine:
    knowledge: List[str] = []
    loaded = False

    @classmethod
    def load(cls):
        if cls.loaded:
            return
        try:
            path = os.path.join(os.getcwd(), "knowledge.txt")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    cls.knowledge = [l.strip() for l in f if len(l.strip()) > 20]
            logger.info(f"Knowledge loaded: {len(cls.knowledge)} lines")
            cls.loaded = True
        except Exception as e:
            logger.error(f"LOAD ERROR: {e}")

    @classmethod
    def score(cls, q, line):
        return difflib.SequenceMatcher(None, q.lower(), line.lower()).ratio()

    @classmethod
    def get(cls, query: str):
        cls.load()

        scored = [(cls.score(query, l), l) for l in cls.knowledge]
        scored.sort(reverse=True)

        top = [l for s, l in scored[:5] if s > 0.25]

        return {
            "context": "\n".join(top),
            "matches": len(top)
        }

# -----------------------------
# 7. RESPONSE CLEANER
# -----------------------------
def clean_response(text: str) -> str:
    forbidden = ["knowledge cutoff", "as an ai", "language model"]
    for f in forbidden:
        text = re.sub(f, "", text, flags=re.IGNORECASE)
    return text.strip()

# -----------------------------
# 8. CHAT ENDPOINT
# -----------------------------
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):

    if not authorization:
        raise HTTPException(401, "Missing API key")

    user_msg = payload.messages[-1].content

    # LOCAL RAG
    ctx = ContextEngine.get(user_msg)

    # WIKI RAG
    wiki = ""
    if ctx["matches"] < 2:
        q = clean_query(user_msg)
        wiki = get_wiki_context(q)

    # BUILD PROMPT
    sys_prompt = BASE_PROMPT

    if ctx["context"]:
        sys_prompt += f"\n\n### LOCAL KNOWLEDGE:\n{ctx['context']}"

    if wiki:
        sys_prompt += f"\n\n### VERIFIED WIKIPEDIA DATA:\n{wiki}"

    messages = [{"role": "system", "content": sys_prompt}]
    messages += [{"role": m.role, "content": m.content} for m in payload.messages]

    try:
        res = GROQ.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens
        )

        reply = clean_response(res.choices[0].message.content)

        return {
            "message": reply,
            "wiki_used": bool(wiki),
            "local_matches": ctx["matches"],
            "tokens": res.usage.total_tokens
        }

    except Exception as e:
        logger.error(f"MODEL ERROR: {e}")
        raise HTTPException(500, "Model failed")

# -----------------------------
# 9. STARTUP
# -----------------------------
@app.on_event("startup")
async def startup():
    ContextEngine.load()
    logger.info("System Ready")

# -----------------------------
# 10. RUN
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
