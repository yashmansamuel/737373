import os
import logging
import secrets
import re
import requests
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import difflib

# -----------------------------
# 1. ENV + LOGGING
# -----------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-L1.0")

required_vars = ["GROQ_API_KEY"]
missing = [v for v in required_vars if not os.getenv(v)]
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

GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# -----------------------------
# 3. STRONG SYSTEM PROMPT
# -----------------------------
SYSTEM_PROMPT = """
You are Neo L1.0.

STRICT RULES:
- You MUST use the provided knowledge (Local Knowledge + Wikipedia).
- DO NOT rely on your own memory.
- DO NOT say "knowledge cutoff".
- If external knowledge is provided, base your answer ONLY on that.
- Be natural, clear, and confident.
"""

# -----------------------------
# 4. REQUEST MODEL
# -----------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2000

# -----------------------------
# 5. QUERY CLEANING
# -----------------------------
def clean_query(text: str) -> str:
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return " ".join(words[:3])  # top 3 keywords

# -----------------------------
# 6. WIKIPEDIA ENGINE
# -----------------------------
WIKI_CACHE: Dict[str, str] = {}

def fetch_wikipedia(query: str) -> str:
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '%20')}"
        res = requests.get(url, timeout=3)

        if res.status_code == 200:
            return res.json().get("extract", "")

        # fallback search
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json"
        }

        r = requests.get(search_url, params=params)
        data = r.json()

        if data["query"]["search"]:
            title = data["query"]["search"][0]["title"]
            return fetch_wikipedia(title)

        return ""
    except Exception as e:
        logger.error(f"Wikipedia error: {e}")
        return ""

def filter_sentences(text: str, query: str) -> str:
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
    filtered = filter_sentences(raw, query)

    WIKI_CACHE[query] = filtered
    return filtered

# -----------------------------
# 7. LOCAL RAG ENGINE
# -----------------------------
class ContextEngine:
    knowledge: List[str] = []
    loaded = False

    @classmethod
    def load(cls):
        if cls.loaded:
            return
        path = os.path.join(os.getcwd(), "knowledge.txt")

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cls.knowledge = [l.strip() for l in f if len(l.strip()) > 20]

        cls.loaded = True
        logger.info(f"Loaded {len(cls.knowledge)} knowledge lines")

    @classmethod
    def score(cls, q: str, line: str):
        return difflib.SequenceMatcher(None, q.lower(), line.lower()).ratio()

    @classmethod
    def search(cls, query: str):
        cls.load()

        scored = [(cls.score(query, l), l) for l in cls.knowledge]
        scored.sort(reverse=True)

        top = [l for s, l in scored[:5] if s > 0.25]

        return {
            "context": "\n".join(top),
            "matches": len(top)
        }

# -----------------------------
# 8. CHAT ENDPOINT
# -----------------------------
@app.post("/chat")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(401, "Missing API key")

    user_msg = payload.messages[-1].content

    # 1. LOCAL SEARCH
    ctx = ContextEngine.search(user_msg)

    # 2. WIKI FALLBACK
    wiki = ""
    if ctx["matches"] < 2:
        clean = clean_query(user_msg)
        wiki = get_wiki_context(clean)

    # DEBUG
    print("LOCAL:", ctx["matches"])
    print("WIKI:", wiki[:200])

    # 3. BUILD PROMPT
    sys_prompt = SYSTEM_PROMPT

    if ctx["context"]:
        sys_prompt += "\n\n### LOCAL KNOWLEDGE:\n" + ctx["context"]

    if wiki:
        sys_prompt += "\n\n### VERIFIED WIKIPEDIA DATA:\n" + wiki + "\nUse this strictly."

    # 4. SEND TO MODEL
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
            "response": res.choices[0].message.content,
            "wiki_used": bool(wiki),
            "local_matches": ctx["matches"]
        }

    except Exception as e:
        logger.error(e)
        raise HTTPException(500, "Model error")

# -----------------------------
# 9. STARTUP
# -----------------------------
@app.on_event("startup")
async def startup():
    ContextEngine.load()

# -----------------------------
# 10. RUN
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
