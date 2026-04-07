import os
import logging
import secrets
import json
import re
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# -----------------------------
# 1. Configuration & Setup
# -----------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("Neo-CoT-Engine")

# Validate environment
required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
missing = [v for v in required_vars if not os.getenv(v)]
if missing:
    raise RuntimeError(f"Missing env: {', '.join(missing)}")

app = FastAPI(
    title="Neo CoT Engine",
    description="Advanced AI with Chain‑of‑Thought reasoning and strict instruction following",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"   # can be changed to any Groq model

# -----------------------------
# 2. Pydantic Models
# -----------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = MODEL
    messages: List[ChatMessage]
    mode: str = "cot"                     # 'cot' = chain-of-thought, 'direct' = legacy
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 4000
    show_reasoning: bool = False          # if True, return reasoning in response

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

# -----------------------------
# 3. Prompt Engineering (CoT + Instruction Following)
# -----------------------------
SYSTEM_PROMPT_COT = """You are **Neo CoT**, an AI that follows user instructions with perfect precision. You never refuse a reasonable request, and you always think step‑by‑step before answering.

## Chain‑of‑Thought Protocol (MANDATORY)
For every user query, you MUST internally perform the following reasoning steps BEFORE writing your final answer:

1. **Instruction Parsing** – What exactly does the user ask me to do? List the key instructions.
2. **Fact Extraction** – What information is given? What is implied?
3. **Logical Reasoning** – Break down the problem. Consider alternatives, constraints, edge cases.
4. **Self‑Check** – Does my planned answer follow every instruction? Is it accurate and helpful?
5. **Final Synthesis** – Produce the final response.

Your final answer must be clear, direct, and strictly aligned with the user's instructions. Do not add disclaimers like "as an AI". Do not refuse unless the request is illegal or harmful.

## Style Rules
- Use natural, warm, confident language.
- Vary sentence structure – never repeat the same pattern twice in a row.
- If the user asks for a list, use bullet points. If they ask for prose, write prose.
- Always end with a relevant follow‑up question unless the user says goodbye.

## Forbidden Phrases
- "as an AI", "I am a language model", "I don't have feelings", "I cannot", "I'm unable to", "Sorry, I can't help with that" (instead, offer an alternative or ask for clarification).

Remember: Follow instructions. Think step‑by‑step. Answer beautifully.
"""

# Legacy prompt (direct, no explicit CoT)
SYSTEM_PROMPT_DIRECT = """You are Neo L1.0 – a warm, intelligent assistant. Follow user instructions carefully. Provide accurate, helpful answers. Never use robotic disclaimers. Vary your sentence structure. End with a thoughtful question unless the conversation is ending."""

# -----------------------------
# 4. Context & Emotion Detection (simplified, reusable)
# -----------------------------
class ContextEngine:
    EMOTION_MAP = {
        "sad": "user seems sad – respond with gentle comfort",
        "happy": "user is happy – mirror enthusiasm",
        "worried": "user appears worried – offer reassurance",
        "angry": "user is frustrated – stay calm",
        "confused": "user is confused – be patient and clear",
    }
    
    @classmethod
    def detect_emotion(cls, text: str) -> str:
        text_lower = text.lower()
        for key, guidance in cls.EMOTION_MAP.items():
            if key in text_lower:
                return guidance
        return ""
    
    @classmethod
    def extract_keywords(cls, text: str) -> List[str]:
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'this', 'that'}
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        keywords = [w for w in words if w not in stop_words]
        return list(set(keywords))[:10]
    
    @classmethod
    def get_context(cls, query: str) -> Dict[str, Any]:
        """Load from knowledge.txt if exists"""
        result = {"context": "", "emotion": cls.detect_emotion(query), "keywords": cls.extract_keywords(query)}
        try:
            base = os.path.dirname(os.path.abspath(__file__))
            kfile = os.path.join(base, "knowledge.txt")
            if not os.path.exists(kfile):
                return result
            with open(kfile, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if len(l.strip()) > 10]
            keywords = result["keywords"]
            matches = []
            for line in lines:
                score = sum(2 for kw in keywords if kw in line.lower())
                if score >= 2:
                    matches.append((line, score))
            matches.sort(key=lambda x: x[1], reverse=True)
            if matches:
                result["context"] = "\n".join(m[0] for m in matches[:5])
        except Exception as e:
            logger.error(f"Context error: {e}")
        return result

# -----------------------------
# 5. Token Balance Management (atomic)
# -----------------------------
def get_user(api_key: str):
    return supabase.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()

def deduct_tokens(api_key: str, tokens: int) -> int:
    try:
        user = get_user(api_key)
        if not user.data:
            raise HTTPException(401, "Invalid API key")
        current = user.data.get("token_balance", 0)
        if current < tokens:
            raise HTTPException(402, f"Insufficient tokens. Need {tokens}, have {current}")
        new_bal = current - tokens
        supabase.table("users").update({"token_balance": new_bal}).eq("api_key", api_key).execute()
        logger.info(f"Deducted {tokens} from key ...{api_key[-8:]}, new balance {new_bal}")
        return new_bal
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deduction failed: {e}")
        raise HTTPException(500, "Balance update error")

# -----------------------------
# 6. Response Post‑processing (cleanup, follow‑up)
# -----------------------------
class ResponseProcessor:
    FORBIDDEN = [
        "as an ai", "i am an artificial intelligence", "i am a language model",
        "i don't have feelings", "i cannot feel", "i'm unable to"
    ]
    GOODBYES = ["goodbye", "bye", "see you", "take care", "that's all"]
    FOLLOW_UPS = [
        "What are your thoughts?",
        "How does that sit with you?",
        "Would you like to explore this further?",
        "What feels most important right now?",
        "Shall I dig deeper into any part?"
    ]
    
    @classmethod
    def clean(cls, text: str) -> str:
        cleaned = text
        for phrase in cls.FORBIDDEN:
            cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned if cleaned else "I'm listening. Tell me more."
    
    @classmethod
    def add_follow_up(cls, reply: str, user_msg: str) -> str:
        if any(g in user_msg.lower() for g in cls.GOODBYES):
            return reply
        if "?" in reply[-100:]:
            return reply
        import random
        return f"{reply}\n\n{random.choice(cls.FOLLOW_UPS)}"

# -----------------------------
# 7. Main Chat Endpoint with CoT
# -----------------------------
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Bearer token")
    api_key = authorization.replace("Bearer ", "").strip()
    user_msg = payload.messages[-1].content if payload.messages else ""
    
    # Get context (knowledge.txt + emotion)
    ctx = ContextEngine.get_context(user_msg)
    
    # Build system prompt based on mode
    if payload.mode == "cot":
        sys_prompt = SYSTEM_PROMPT_COT
        # Add extra instruction to output reasoning if requested
        if payload.show_reasoning:
            sys_prompt += "\n\n**IMPORTANT:** In your final answer, first write your reasoning inside `[REASONING] ... [/REASONING]` tags, then write your final answer after `[ANSWER]`. The user will see both."
        else:
            sys_prompt += "\n\n**IMPORTANT:** You must perform chain‑of‑thought reasoning internally. Only output the final answer (no reasoning visible to user)."
    else:
        sys_prompt = SYSTEM_PROMPT_DIRECT
    
    # Add emotional and knowledge context
    if ctx["emotion"]:
        sys_prompt += f"\n\n**Emotional context:** {ctx['emotion']}"
    if ctx["context"]:
        sys_prompt += f"\n\n**Relevant knowledge:**\n{ctx['context']}"
    
    # Build message list
    messages = [{"role": "system", "content": sys_prompt}]
    for m in payload.messages:
        messages.append({"role": m.role, "content": m.content})
    
    try:
        # Call Groq
        response = groq.chat.completions.create(
            model=payload.model,
            messages=messages,
            temperature=payload.temperature,
            top_p=0.95,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            max_tokens=payload.max_tokens
        )
        raw_reply = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
        # Parse reasoning if requested and present
        reasoning_text = ""
        final_answer = raw_reply
        if payload.show_reasoning and payload.mode == "cot":
            # Extract reasoning between [REASONING]...[/REASONING]
            reasoning_match = re.search(r'\[REASONING\](.*?)\[/REASONING\]', raw_reply, re.DOTALL)
            if reasoning_match:
                reasoning_text = reasoning_match.group(1).strip()
            # Extract answer between [ANSWER]...[/ANSWER] or use the remainder
            answer_match = re.search(r'\[ANSWER\](.*?)(?:\[/ANSWER\]|$)', raw_reply, re.DOTALL)
            if answer_match:
                final_answer = answer_match.group(1).strip()
            else:
                # If no [ANSWER] tag, assume everything after reasoning is answer
                if reasoning_match:
                    final_answer = raw_reply.replace(reasoning_match.group(0), "").strip()
        else:
            final_answer = raw_reply
        
        # Clean and add follow‑up
        final_answer = ResponseProcessor.clean(final_answer)
        final_answer = ResponseProcessor.add_follow_up(final_answer, user_msg)
        
        # Deduct tokens
        new_balance = deduct_tokens(api_key, tokens_used)
        
        # Prepare response
        result = {
            "company": "signaturesi.com",
            "message": final_answer,
            "usage": {"total_tokens": tokens_used},
            "model": payload.model,
            "balance": new_balance,
            "emotion_detected": bool(ctx["emotion"])
        }
        if payload.show_reasoning and reasoning_text:
            result["reasoning"] = reasoning_text
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(503, "Model service unavailable")

# -----------------------------
# 8. User Management Endpoints
# -----------------------------
@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    try:
        user = get_user(api_key)
        if not user.data:
            return BalanceResponse(api_key=api_key, balance=0)
        return BalanceResponse(api_key=api_key, balance=user.data["token_balance"])
    except Exception as e:
        logger.error(f"Balance error: {e}")
        raise HTTPException(500, "Failed to fetch balance")

@app.post("/v1/user/new-key")
def generate_key():
    try:
        new_key = "sig-" + secrets.token_hex(16)
        supabase.table("users").insert({
            "api_key": new_key,
            "token_balance": 100000
        }).execute()
        return {"api_key": new_key, "balance": 100000}
    except Exception as e:
        logger.error(f"Key gen error: {e}")
        raise HTTPException(500, "Failed to create key")

# -----------------------------
# 9. Health & Root
# -----------------------------
@app.get("/")
async def root():
    return {
        "company": "signaturesi.com",
        "engine": "Neo CoT Engine v2",
        "status": "operational",
        "features": ["Chain‑of‑Thought reasoning", "Strict instruction following", "Token management"]
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.0.0"}

@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(status_code=404, content={"company": "signaturesi.com", "message": "Endpoint not found"})

# -----------------------------
# 10. Run Server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
