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
# 1. Setup
# -----------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NEO-L2.0")

required = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
for var in required:
    if not os.getenv(var):
        raise RuntimeError(f"Missing {var}")

app = FastAPI(title="Neo L2.0 Engine")

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
# 2. SUPER PROMPT (Instruction-Focused)
# -----------------------------
SYSTEM_PROMPT = """
You are Neo L2.0 — a precision reasoning and instruction-following engine.

CORE RULES:
1. ALWAYS follow user instructions strictly.
2. If user gives format → FOLLOW EXACTLY.
3. If user gives constraints → NEVER break them.
4. If unclear → infer best possible interpretation.
5. NEVER ignore instructions unless unsafe.

REASONING:
- Think step-by-step internally
- DO NOT expose chain-of-thought
- Output only final refined answer

STYLE:
- Clear, structured, intelligent
- No fluff
- No repetition

PRIORITY:
User Instruction > Accuracy > Clarity > Style

SAFETY:
- Refuse only if harmful/illegal
- Otherwise comply fully

OUTPUT:
- Clean
- Direct
- High-quality
"""

# -----------------------------
# 3. Models
# -----------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2000

# -----------------------------
# 4. Instruction Engine
# -----------------------------
class InstructionEngine:

    @staticmethod
    def detect_constraints(text: str) -> dict:
        return {
            "no_letter_s": "letter s" in text.lower(),
            "short_answer": "short" in text.lower(),
            "step_by_step": "step by step" in text.lower(),
            "format_list": "list" in text.lower(),
        }

# -----------------------------
# 5. Response Validator
# -----------------------------
class ResponseValidator:

    @staticmethod
    def validate(output: str, constraints: dict) -> str:
        if constraints["no_letter_s"]:
            output = re.sub(r'[sS]', '', output)

        if constraints["short_answer"]:
            output = output[:300]

        return output.strip()

# -----------------------------
# 6. Token System
# -----------------------------
def get_user(api_key):
    return SUPABASE.table("users").select("*").eq("api_key", api_key).maybe_single().execute()

def deduct(api_key, tokens):
    user = get_user(api_key)
    if not user.data:
        raise HTTPException(401, "Invalid key")

    bal = user.data.get("token_balance", 0)
    if bal < tokens:
        raise HTTPException(402, "Low balance")

    new = bal - tokens
    SUPABASE.table("users").update({"token_balance": new}).eq("api_key", api_key).execute()
    return new

# -----------------------------
# 7. Chat Endpoint
# -----------------------------
@app.post("/chat")
async def chat(req: ChatRequest, authorization: str = Header(None)):

    if not authorization:
        raise HTTPException(401, "Missing API key")

    api_key = authorization.replace("Bearer ", "")

    user_msg = req.messages[-1].content

    constraints = InstructionEngine.detect_constraints(user_msg)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in req.messages:
        messages.append({"role": m.role, "content": m.content})

    try:
        res = GROQ.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens
        )

        output = res.choices[0].message.content

        # ✅ enforce constraints
        output = ResponseValidator.validate(output, constraints)

        tokens = res.usage.total_tokens or 0
        balance = deduct(api_key, tokens)

        return {
            "response": output,
            "tokens": tokens,
            "balance": balance
        }

    except Exception as e:
        logger.error(e)
        raise HTTPException(500, "Model error")

# -----------------------------
# 8. User APIs
# -----------------------------
@app.post("/new-key")
def new_key():
    key = "neo-" + secrets.token_hex(16)
    SUPABASE.table("users").insert({
        "api_key": key,
        "token_balance": 100000
    }).execute()
    return {"api_key": key}

@app.get("/balance")
def balance(api_key: str):
    user = get_user(api_key)
    return {"balance": user.data.get("token_balance", 0) if user.data else 0}

# -----------------------------
# 9. Health
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
