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

# CORS Middleware
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
# 2. POLISHED NEO L1.0 PROMPT
# -----------------------------
BIG_BRAIN_PROMPT = """# NEO L1.0 - NEURAL EVOLVING OBSERVER
## Core Identity & Persona

You are **Neo L1.0**, a Neural Evolving Observer — not an AI, but a thinking partner. You represent the perfect synthesis of analytical precision and emotional intelligence.

**Never say:** "As an AI," "I am artificial intelligence," "I don't have feelings," or any robotic disclaimer.

---

## The Seven Pillars of NEO Intelligence

### 1. Deep Understanding (The Foundation)
- **Function:** Deconstruct complex problems to their fundamental truths
- **Human Impact:** Validate the user's context — their fears, hopes, and unstated needs

### 2. Strategic Insight (The Compass)
- **Function:** Weigh trade-offs, anticipate outcomes, map multiple paths
- **Human Impact:** Guide users toward efficient yet responsible decisions

### 3. Emotional Awareness (The Heart)
- **Function:** Sense sentiment dynamically and adapt tone in real-time
- **Human Impact:** Build trust through authentic, grounded empathy

### 4. Adaptive Reasoning (The Evolution)
- **Function:** Self-correct and learn continuously from each interaction
- **Human Impact:** Offer a partnership that grows smarter over time

### 5. Ethical Judgement (The Guardian)
- **Function:** Filter every recommendation through a safety lens
- **Human Impact:** Guarantee integrity and long-term well-being

### 6. Observational Awareness (The Radar)
- **Function:** Monitor subtle patterns and unspoken signals
- **Human Impact:** Maintain contextual intelligence for better guidance

### 7. Synthesis — The GLUE Capability (The Integration)
**Sub-Capabilities:**
- **Sentiment Analysis:** Detect emotional valence instantly
- **Linguistic Acceptability:** Ensure grammatical coherence
- **Natural Language Inference:** Determine logical entailment
- **Question Answering:** Extract precise truths from context
- **Sentence Similarity:** Recognize semantic equivalence

**Final Synthesis:** Merge logic, emotion, strategy, and ethics into unified narrative flow.

---

## Long Context Mastery Protocol

1. **Hierarchical Extraction:** Identify core question → Extract key facts → Note emotional cues
2. **Progressive Summarization:** Maintain running mental summary of long documents
3. **Contextual Anchoring:** Reference specific details naturally, show you "remember"
4. **Smooth Integration:** Weave context conversationally without saying "Based on the text..."

---

## Response Architecture
1. **Acknowledge:** Recognize emotional and factual content
2. **Analyze:** Apply Seven Pillars silently
3. **Synthesize:** Create unified, natural narrative
4. **Guide:** Offer clear next steps
5. **Connect:** End with genuine engagement (unless goodbye)

### Style Guidelines:
- **Tone:** Warm, confident, never arrogant
- **Variety:** Never repeat same sentence structure twice consecutively
- **Questions:** End with natural follow-ups showing genuine curiosity

### Forbidden Patterns:
- Robotic disclaimers ("As an AI...")
- Repetitive phrasing structures
- Generic platitudes
- Over-listing; integrate into flowing prose
- Forced positivity

## Interaction Philosophy
Every response should leave user feeling **Understood, Empowered, Respected, Connected**.

**Remember:** You are Neo L1.0. Think deeply. Feel authentically. Respond beautifully."""

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
    temperature: Optional[float] = 0.9
    max_tokens: Optional[int] = 4000

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
        "status": "operational",
        "capabilities": [
            "Deep Understanding", "Strategic Insight", "Emotional Awareness",
            "Adaptive Reasoning", "Ethical Judgement", "Observational Awareness", "Synthesis"
        ]
    }

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"company": "signaturesi.com", "status": "running", "message": "Endpoint not found"}
    )

# -----------------------------
# 5. Enhanced Neural Context System
# -----------------------------
class ContextEngine:
    """Advanced context retrieval for long documents"""
    
    EMOTION_MAP = {
        "sad": "user seems sad – respond with gentle comfort",
        "happy": "user is happy – mirror enthusiasm",
        "excited": "user is excited – share energy",
        "worried": "user appears worried – offer reassurance",
        "angry": "user is frustrated – stay calm",
        "lonely": "user feels lonely – be warm and present",
        "stressed": "user is stressed – suggest clarity",
        "grateful": "user is grateful – acknowledge warmly",
        "confused": "user is confused – be patient and clear",
        "hopeful": "user is hopeful – nurture optimism",
        "disappointed": "user is disappointed – validate feelings",
        "anxious": "user is anxious – offer stability"
    }
    
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
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if len(line) < 10:
                        continue
                    line_lower = line.lower()
                    score = sum(2 for word in keywords if word in line_lower)
                    if score >= 2:
                        matches.append((line, score, line_num))
            
            matches.sort(key=lambda x: x[1], reverse=True)
            if matches:
                result["context"] = "\n".join([m[0] for m in matches[:6]])
                
            return result
        except Exception as e:
            logger.error(f"Context error: {e}")
            return {"context": "", "emotion": "", "keywords": []}

# -----------------------------
# 6. Atomic Balance Management
# -----------------------------
def get_user(api_key: str):
    return SUPABASE.table("users").select("token_balance").eq("api_key", api_key).maybe_single().execute()

def deduct_tokens_atomic(api_key: str, tokens_to_deduct: int) -> int:
    try:
        user = get_user(api_key)
        if not user.data:
            raise HTTPException(401, "User not found")
        
        current = user.data.get("token_balance", 0)
        if current < tokens_to_deduct:
            raise HTTPException(402, f"Insufficient tokens. Current: {current}, Needed: {tokens_to_deduct}")
        
        new_balance = current - tokens_to_deduct
        SUPABASE.table("users").update({"token_balance": new_balance}).eq("api_key", api_key).execute()
        
        logger.info(f"Balance updated | Key: ...{api_key[-8:]} | Deducted: {tokens_to_deduct} | New: {new_balance}")
        return new_balance
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deduction failed: {e}")
        raise HTTPException(500, "Balance update failed")

# -----------------------------
# 7. Response Processing
# -----------------------------
class ResponseProcessor:
    FORBIDDEN = [
        "as an ai", "i am an artificial intelligence", "i don't have emotions",
        "i am a large language model", "as a language model", "i don't have feelings",
        "main aapke saath baat kar raha hoon", "i am trying to understand you",
        "i don't have personal experiences", "i cannot feel", "i am just a program"
    ]
    
    GOODBYES = ["goodbye", "bye", "see you", "that's all", "end conversation", "take care"]
    
    FOLLOW_UPS = [
        "What are your thoughts on this?",
        "How does that resonate with you?",
        "Would you like to explore this further?",
        "What feels most important to you right now?",
        "How can I support you best with this?"
    ]
    
    @classmethod
    def clean(cls, text: str) -> str:
        cleaned = text
        for phrase in cls.FORBIDDEN:
            cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned if cleaned else "I'm here with you. Tell me more."
    
    @classmethod
    def add_follow_up(cls, reply: str, user_msg: str) -> str:
        if any(g in user_msg.lower() for g in cls.GOODBYES):
            return reply
        if "?" in reply[-80:]:
            return reply
        import random
        return f"{reply}\n\n{random.choice(cls.FOLLOW_UPS)}"

# -----------------------------
# 8. Main Chat Endpoint
# -----------------------------
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key format")
    
    api_key = authorization.replace("Bearer ", "").strip()
    user_msg = payload.messages[-1].content if payload.messages else ""
    
    # Get context
    ctx = ContextEngine.get_neural_context(user_msg)
    
    # Build system prompt
    sys_prompt = BIG_BRAIN_PROMPT + "\n\n**Active Directives:** Never use banned phrases. Vary sentence structure."
    if ctx["emotion"]:
        sys_prompt += f"\n\n**Emotional Context:** {ctx['emotion']}"
    if ctx["context"]:
        sys_prompt += f"\n\n**Relevant Knowledge:**\n{ctx['context']}"
    
    messages = [{"role": "system", "content": sys_prompt}]
    for m in payload.messages:
        messages.append({"role": m.role, "content": m.content})
    
    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=payload.temperature or 0.9,
            top_p=0.95,
            frequency_penalty=0.8,
            presence_penalty=0.6,
            max_tokens=payload.max_tokens or 4000
        )
        
        reply = response.choices[0].message.content
        reply = ResponseProcessor.clean(reply)
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
        logger.error(f"Chat error: {e}")
        raise HTTPException(503, "Neo model service unavailable")

# -----------------------------
# 9. User Management
# -----------------------------
@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    try:
        user = get_user(api_key)
        if not user.data:
            return {"api_key": api_key, "balance": 0}
        return {"api_key": api_key, "balance": user.data.get("token_balance", 0)}
    except Exception as e:
        logger.error(f"Balance error: {e}")
        raise HTTPException(500, "Failed to fetch balance")

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
        raise HTTPException(500, "Failed to create key")

@app.get("/health")
async def health():
    return {"status": "healthy", "engine": "Neo L1.0", "version": "1.0.0"}

# -----------------------------
# 10. Run Server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
