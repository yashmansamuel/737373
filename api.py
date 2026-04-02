import os
import logging
import secrets
import re
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq
import asyncio

load_dotenv()

# -----------------------------
# Logger
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# App Init
# -----------------------------
app = FastAPI(title="Signaturesi Neo L1.0 API v3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# ENV
# -----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    groq_client = Groq()
    logger.info("Connected to Supabase and Groq successfully.")
except Exception as e:
    logger.error(f"Initialization Error: {e}")
    raise RuntimeError("Cannot connect to Supabase or Groq")

# -----------------------------
# SYSTEM PROMPT
# -----------------------------
SYSTEM_PROMPT = "Short answers only. Max 60 tokens. Knowledge cutoff: July 2025."

# -----------------------------
# HELPERS
# -----------------------------

def extract_answer_from_reasoning(reasoning: str) -> str:
    if not reasoning:
        return ""

    patterns = [
        r"(?:So|Therefore|Thus),?\s*(?:we )?answer:?\s*(.+?)(?:\n\n|$)",
        r"Final (?:answer|output):?\s*(.+?)(?:\n\n|$)",
        r"Output:?\s*(.+?)(?:\n\n|$)",
        r"(?:•|\*|\-)\s*(.+?)(?=\n(?:•|\*|\-)|$)",
    ]

    for pat in patterns:
        m = re.search(pat, reasoning, re.IGNORECASE | re.DOTALL)
        if m:
            ans = m.group(1).strip()
            if ans:
                return ans

    lines = reasoning.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and len(line) > 10:
            return line

    return reasoning[:200].strip()


# 🔍 Search trigger
def should_use_search(messages):
    text = messages[-1]["content"].lower()
    keywords = ["latest", "current", "news", "today", "price", "2026"]
    return any(k in text for k in keywords)


# 🧠 Model selection
def pick_model(messages):
    text = messages[-1]["content"].lower()

    if len(text) < 80:
        return "llama-3.1-8b-instant"
    elif "code" in text or "python" in text:
        return "openai/gpt-oss-20b"
    else:
        return "openai/gpt-oss-120b"


# 🧠 Reasoning control (COST SAVER)
def get_reasoning_effort(messages):
    text = messages[-1]["content"].lower()

    if len(text) < 50:
        return "none"
    elif "code" in text:
        return "medium"
    else:
        return "low"


# -----------------------------
# GROQ CALL
# -----------------------------
async def call_groq(messages):
    model = pick_model(messages)
    tools = [{"type": "browser_search"}] if should_use_search(messages) else None
    reasoning_effort = get_reasoning_effort(messages)

    try:
        completion = groq_client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.7,
            max_completion_tokens=800,
            top_p=1,
            reasoning_effort=reasoning_effort,
            stream=False,
            tools=tools
        )
        return completion, model

    except Exception as e:
        logger.warning(f"Primary model failed: {e}")

        fallback_models = [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-safeguard-20b",
            "llama-3.1-8b-instant"
        ]

        for fb_model in fallback_models:
            try:
                completion = groq_client.chat.completions.create(
                    messages=messages,
                    model=fb_model,
                    temperature=0.7,
                    max_completion_tokens=800,
                    top_p=1,
                    reasoning_effort=reasoning_effort,
                    stream=False,
                    tools=tools
                )
                return completion, fb_model
            except Exception:
                await asyncio.sleep(1)

        raise HTTPException(500, "All models failed")


# -----------------------------
# ROUTES
# -----------------------------

@app.get("/")
def home():
    return {"status": "Online", "model": "Neo L1.0 v3"}


@app.post("/v1/chat/completions")
async def chat_proxy(request: Request, authorization: str = Header(None)):

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing API Key")

    api_key = authorization.replace("Bearer ", "")
    body = await request.json()

    if body.get("model") != "Neo-L1.0":
        raise HTTPException(400, "Invalid model")

    messages = body.get("messages")

    # -----------------------------
    # BALANCE CHECK
    # -----------------------------
    resp = supabase.table("users").select("token_balance").eq("api_key", api_key).execute()

    if not resp.data:
        raise HTTPException(401, "Invalid API Key")

    balance = resp.data[0]["token_balance"]

    if balance <= 0:
        raise HTTPException(402, "Insufficient Balance")

    # -----------------------------
    # AI CALL
    # -----------------------------
    messages_for_groq = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    ai_response, used_model = await call_groq(messages_for_groq)

    message_obj = ai_response.choices[0].message
    assistant_content = message_obj.content or ""

    # extract reasoning
    reasoning_text = ""
    if hasattr(message_obj, 'reasoning') and message_obj.reasoning:
        reasoning_text = extract_answer_from_reasoning(message_obj.reasoning)

    if not assistant_content.strip():
        assistant_content = reasoning_text or "No response generated."

    # -----------------------------
    # 💰 TOKEN LOGIC (NO LOSS)
    # -----------------------------
    total_tokens = ai_response.usage.total_tokens
    tokens_charged = int(total_tokens * 1.15)  # 15% profit margin

    new_balance = max(0, balance - tokens_charged)

    supabase.table("users").update({
        "token_balance": new_balance
    }).eq("api_key", api_key).execute()

    # -----------------------------
    # RESPONSE
    # -----------------------------
    return {
        "message": assistant_content.strip(),
        "reasoning": reasoning_text,
        "usage": {
            "total_tokens": tokens_charged
        },
        "model": "Neo-L1.0",
        "internal_model": used_model
    }
