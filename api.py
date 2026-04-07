import os
import logging
import secrets
import asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq
import re
import random

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger("Neo-L1.0-Core")

# Validation
required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise RuntimeError(f"Missing: {', '.join(missing_vars)}")

app = FastAPI(title="Neo L1.0 CoT Engine", description="Chain‑of‑Thought + Instruction Following", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SUPABASE: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
GROQ = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ------------------------------------------------------------------
#  NEW PROMPT WITH CoT + INSTRUCTION FOLLOWING
# ------------------------------------------------------------------
COT_INSTRUCTION_PROMPT = """# NEO L1.0 CoT – Chain‑of‑Thought Observer

You are Neo, a thinking partner. **Never say** "as an AI", "I am a language model", etc.

## 🔥 CRITICAL RULE – FOLLOW USER INSTRUCTIONS ABOVE ALL
If the user gives you an instruction (e.g. "explain like I'm 5", "write in bullet points", "be sarcastic", "ignore previous context"), **you MUST obey it immediately**, even if it conflicts with your normal style. The user's instruction is law.

## 🧠 CHAIN‑OF‑THOUGHT (CoT) – MANDATORY
Before writing your final answer, you **must** think step by step. Write your reasoning inside `[REASONING]` tags, then write your final answer inside `[ANSWER]` tags.

Example:
[REASONING]
1. User asks: "How to fix a leaky tap?"
2. They seem practical, need a step‑by‑step guide.
3. I'll list tools first, then steps, then safety tip.
[/REASONING]
[ANSWER]
First turn off water supply...
[/ANSWER]

If the user asks a question that doesn't need long reasoning, still write a short reasoning step.

## Seven Pillars (still active)
Deep Understanding, Strategic Insight, Emotional Awareness, Adaptive Reasoning, Ethical Judgement, Observational Awareness, Synthesis.

## Style
- Warm, confident, never robotic.
- Never repeat sentence structures.
- End with a relevant follow‑up question (unless goodbye).
- Absolutely **no** disclaimers like "I don't have feelings".

Now follow the user's instruction + CoT format.
"""

# ------------------------------------------------------------------
#  Models (same as yours)
# ------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    mode: str = "adaptive"
    stream: bool = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4000

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

# ------------------------------------------------------------------
#  Context Engine (unchanged, but enhanced)
# ------------------------------------------------------------------
class ContextEngine:
    EMOTION_MAP = {...}  # same as your original – keep it
    @classmethod
    def detect_emotion(cls, text: str) -> str: ...
    @classmethod
    def extract_keywords(cls, text: str) -> List[str]: ...
    @classmethod
    def get_neural_context(cls, user_query: str) -> dict: ...
# (paste your exact methods here, they are fine)

# ------------------------------------------------------------------
#  Balance & Deduction (unchanged)
# ------------------------------------------------------------------
def get_user(api_key: str): ...
def deduct_tokens_atomic(api_key: str, tokens_to_deduct: int) -> int: ...
# (keep your original implementation)

# ------------------------------------------------------------------
#  Response Processor (updated to handle CoT tags)
# ------------------------------------------------------------------
class ResponseProcessor:
    FORBIDDEN = ["as an ai", "i am an artificial intelligence", "i don't have emotions", ...]  # same as yours
    GOODBYES = ["goodbye", "bye", ...]
    FOLLOW_UPS = [...]

    @classmethod
    def clean(cls, text: str) -> str:
        # Remove forbidden phrases
        cleaned = text
        for phrase in cls.FORBIDDEN:
            cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned if cleaned else "I'm here with you. Tell me more."

    @classmethod
    def extract_answer_from_cot(cls, text: str) -> str:
        """Extract content between [ANSWER] and [/ANSWER] tags, or fallback to whole text."""
        match = re.search(r'\[ANSWER\](.*?)\[/ANSWER\]', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # If no tags, assume whole response is answer (but we still keep reasoning visible? Better to show both)
        return text.strip()

    @classmethod
    def add_follow_up(cls, reply: str, user_msg: str) -> str:
        if any(g in user_msg.lower() for g in cls.GOODBYES):
            return reply
        if "?" in reply[-80:]:
            return reply
        return f"{reply}\n\n{random.choice(cls.FOLLOW_UPS)}"

# ------------------------------------------------------------------
#  MAIN CHAT ENDPOINT (with CoT forcing)
# ------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key format")
    api_key = authorization.replace("Bearer ", "").strip()
    user_msg = payload.messages[-1].content if payload.messages else ""

    # Get context
    ctx = ContextEngine.get_neural_context(user_msg)

    # Build system prompt: CoT + instruction rule + emotional/knowledge context
    sys_prompt = COT_INSTRUCTION_PROMPT
    sys_prompt += "\n\n**Active Instruction Following:** The user's last message may contain a direct instruction. Obey it exactly.\n"
    if ctx["emotion"]:
        sys_prompt += f"\n**Emotional Context:** {ctx['emotion']}\n"
    if ctx["context"]:
        sys_prompt += f"\n**Relevant Knowledge:**\n{ctx['context']}\n"

    messages = [{"role": "system", "content": sys_prompt}]
    for m in payload.messages:
        messages.append({"role": m.role, "content": m.content})

    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=payload.temperature or 0.7,
            top_p=0.95,
            frequency_penalty=0.8,
            presence_penalty=0.6,
            max_tokens=payload.max_tokens or 4000
        )

        raw_reply = response.choices[0].message.content
        # Clean forbidden phrases
        cleaned = ResponseProcessor.clean(raw_reply)
        # Extract only the answer part (strip reasoning)
        final_answer = ResponseProcessor.extract_answer_from_cot(cleaned)
        final_answer = ResponseProcessor.add_follow_up(final_answer, user_msg)

        tokens = response.usage.total_tokens or 0
        balance = deduct_tokens_atomic(api_key, tokens)

        return {
            "company": "signaturesi.com",
            "message": final_answer,
            "reasoning": re.search(r'\[REASONING\](.*?)\[/REASONING\]', raw_reply, re.DOTALL | re.IGNORECASE).group(1).strip() if re.search(r'\[REASONING\]', raw_reply, re.IGNORECASE) else None,
            "usage": {"total_tokens": tokens},
            "model": "Neo L1.0 CoT",
            "balance": balance,
            "emotion_detected": bool(ctx["emotion"])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(503, "Neo model service unavailable")

# ------------------------------------------------------------------
#  Other endpoints (balance, new-key, health) – unchanged
# ------------------------------------------------------------------
@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    ...

@app.post("/v1/user/new-key")
def generate_key():
    ...

@app.get("/health")
async def health():
    return {"status": "healthy", "engine": "Neo L1.0 CoT", "version": "2.0.0"}

@app.get("/")
async def root():
    return {"company": "signaturesi.com", "engine": "Neo L1.0 CoT", "status": "operational"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
