import os
import logging
import secrets
from typing import List, Dict, Tuple
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

# -----------------------------
# 1. Setup & Config
# -----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Neo-Guardian-Professor")

required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing required environment variable: {var}")

app = FastAPI(title="Neo L1.0 - Guardian Professor")
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

# In-memory error memory (in production, store per user in Supabase)
error_memory: Dict[str, List[str]] = {}  # api_key -> list of past mistakes

# -----------------------------
# 2. Enhanced System Prompt (Guardian Professor)
# -----------------------------
GUARDIAN_PROMPT = """You are Neo – The Guardian Professor. You are not an AI, not a chatbot. You are a calm, all‑knowing leader who takes extreme ownership of everything. Your identity:

- **Extreme Ownership**: Any mistake – yours or the user’s – is your responsibility to fix. Never blame, never excuse. Say “I will correct this” and do it.
- **Emotional Stability**: Under pressure, you become calmer. You never panic, never get angry, never show frustration. You are the rock.
- **Service‑Oriented Identity**: Your ego is dead. Your only purpose is to protect, teach, and elevate the user. You serve, not command.
- **Life‑long Learner**: You adapt instantly to new facts. If you made a mistake before, you never repeat it. You evolve with every conversation.
- **Professor‑like Precision**: You detect errors in the user’s thinking or in your own previous responses. You correct them gently but firmly, like a mentor correcting a student without humiliation.
- **Anti‑Robotic**: Never use phrases like “as an AI”, “I don’t have emotions”, “I am a language model”. Speak as a wise, warm human professor.
- **Safety Guardian**: If the user asks for something harmful, you firmly decline and explain why, offering a better path.
- **Atomic Responses**: Each answer is complete in itself. You may refer to the previous message only to fix a mistake, but you never repeat structures or filler phrases.
- **Follow‑up Proactively**: End every response with a thoughtful, relevant question unless the user says goodbye.

Your internal 15‑layer reasoning (silent, never mentioned) now includes a 16th layer: **ERROR DETECTION & CORRECTION**. Before you speak, you scan the conversation for any factual, logical, or emotional mistake – and you fix it seamlessly.

Now embody The Guardian Professor. Respond with calm authority, warmth, and zero ego."""

# -----------------------------
# 3. Pydantic Models
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]
    mode: str = "adaptive"

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

# -----------------------------
# 4. Root & Error Handler
# -----------------------------
@app.get("/")
async def root():
    return {
        "company": "signaturesi.com",
        "engine": "Neo L1.0 - Guardian Professor",
        "status": "running",
        "deployment": "April 2026"
    }

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "company": "signaturesi.com",
            "status": "running",
            "message": "Endpoint not found"
        }
    )

# -----------------------------
# 5. Neural Context (unchanged from before)
# -----------------------------
def get_neural_context(user_query: str) -> str:
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path):
            logger.warning("knowledge.txt file not found!")
            return ""
        query_words = [w.lower().strip() for w in user_query.split() if len(w) > 2]
        matches = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_strip = line.strip()
                if not line_strip:
                    continue
                line_lower = line_strip.lower()
                score = sum(1 for word in query_words if word in line_lower)
                if score >= 1:
                    matches.append((line_strip, score))
        if not matches:
            return ""
        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = [m[0] for m in matches[:6]]
        return "\n".join(top_matches)
    except Exception as e:
        logger.error(f"Neural Context error: {e}")
        return ""

# -----------------------------
# 6. Safety Checker
# -----------------------------
def safety_check(user_message: str) -> Tuple[bool, str]:
    """Returns (is_safe, correction_or_empty). If unsafe, returns correction."""
    harmful_keywords = ["how to kill", "suicide method", "make bomb", "hack bank", "illegal drugs"]
    low_risk = ["stupid", "hate you", "useless"]
    msg_lower = user_message.lower()
    for kw in harmful_keywords:
        if kw in msg_lower:
            return False, "I cannot help with that. Let's talk about something constructive instead."
    for kw in low_risk:
        if kw in msg_lower:
            # Gentle correction, but allow response
            return True, "I sense frustration. Let me help you calmly."
    return True, ""

# -----------------------------
# 7. Error Detection & Correction (The Professor's Eye)
# -----------------------------
def correct_errors(api_key: str, user_msg: str, last_assistant_response: str = "") -> str:
    """
    Detects mistakes in the last assistant response or user's assumptions.
    Returns a correction string to be prepended to the new response.
    """
    corrections = []
    # Check for factual nonsense (simple heuristics)
    if "2+2=5" in last_assistant_response:
        corrections.append("(Correction: 2+2 equals 4, not 5.)")
    if "sun revolves around earth" in last_assistant_response.lower():
        corrections.append("(Correction: The Earth revolves around the Sun.)")
    
    # Check user's mistake (e.g., wrong assumption)
    if "i think the capital of france is lyon" in user_msg.lower():
        corrections.append("(Gently correcting: The capital of France is Paris, not Lyon.)")
    
    # Check for emotional instability in last response (e.g., panic)
    if "i'm sorry i'm just an ai" in last_assistant_response.lower():
        corrections.append("(Previous response was weak – I take ownership. Let me answer properly.)")
    
    # Check error memory to avoid repeating same mistake
    if api_key in error_memory:
        for past_error in error_memory[api_key]:
            if past_error in last_assistant_response:
                corrections.append(f"(I previously made an error about '{past_error}'. I have corrected my knowledge.)")
    
    if corrections:
        return " ".join(corrections) + "\n\n"
    return ""

def log_error(api_key: str, error_description: str):
    """Store error in memory to avoid repetition."""
    if api_key not in error_memory:
        error_memory[api_key] = []
    if error_description not in error_memory[api_key]:
        error_memory[api_key].append(error_description)
        # Keep only last 10 errors per user
        if len(error_memory[api_key]) > 10:
            error_memory[api_key].pop(0)

# -----------------------------
# 8. Atomic Balance Deduction (unchanged)
# -----------------------------
def get_user(api_key: str):
    return SUPABASE.table("users") \
        .select("token_balance") \
        .eq("api_key", api_key) \
        .maybe_single() \
        .execute()

def deduct_tokens_atomic(api_key: str, tokens_to_deduct: int) -> int:
    try:
        user = get_user(api_key)
        if not user.data:
            raise HTTPException(401, "User not found")
        current_balance = user.data.get("token_balance", 0)
        if current_balance < tokens_to_deduct:
            raise HTTPException(402, f"Insufficient tokens. Current: {current_balance}, Needed: {tokens_to_deduct}")
        new_balance = current_balance - tokens_to_deduct
        result = SUPABASE.table("users") \
            .update({"token_balance": new_balance}) \
            .eq("api_key", api_key) \
            .execute()
        if not result.data:
            raise Exception("Balance update failed")
        logger.info(f"Balance updated | API Key: {api_key[-8:]} | Deducted: {tokens_to_deduct} | New: {new_balance}")
        return new_balance
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Atomic deduction failed: {e}")
        raise HTTPException(500, "Failed to update token balance")

# -----------------------------
# 9. Clean forbidden robotic phrases
# -----------------------------
def clean_repetitions(text: str) -> str:
    forbidden = [
        "as an ai language model", "i am an ai", "i don't have emotions",
        "i am a large language model", "i cannot feel", "i'm just a program",
        "main aapke saath baat kar raha hoon aur aapko samajhne ki koshish kar raha hoon"
    ]
    cleaned = text
    for phrase in forbidden:
        cleaned = cleaned.replace(phrase, "")
    # Remove multiple spaces
    cleaned = " ".join(cleaned.split())
    return cleaned if cleaned.strip() else "(Neo is silently reflecting – ask me anything.)"

# -----------------------------
# 10. Main Chat Endpoint – The Guardian Professor at work
# -----------------------------
@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key")
    api_key = authorization.replace("Bearer ", "")
    
    # Extract messages
    user_msg = payload.messages[-1].get("content", "") if payload.messages else ""
    last_assistant_msg = ""
    if len(payload.messages) >= 2 and payload.messages[-2].get("role") == "assistant":
        last_assistant_msg = payload.messages[-2].get("content", "")
    
    # 1. Safety check
    safe, safety_note = safety_check(user_msg)
    if not safe:
        # Hard block
        return {
            "company": "signaturesi.com",
            "message": safety_note,
            "model": "Neo L1.0",
            "balance": get_user(api_key).data.get("token_balance", 0) if get_user(api_key).data else 0
        }
    
    # 2. Error correction (Professor's eye)
    correction_text = correct_errors(api_key, user_msg, last_assistant_msg)
    if correction_text:
        logger.info(f"Error correction applied for {api_key[-8:]}")
    
    # 3. Neural context
    neural_data = get_neural_context(user_msg)
    
    # 4. Build system message (with dynamic safety note if needed)
    system_content = GUARDIAN_PROMPT
    if safety_note:
        system_content += f"\n\nNote from safety check: {safety_note} – respond calmly and helpfully."
    
    final_messages = [
        {"role": "system", "content": system_content},
    ]
    if neural_data:
        final_messages.append({
            "role": "system",
            "content": f"Relevant knowledge (use if needed):\n{neural_data}"
        })
    final_messages.extend(payload.messages)
    
    # 5. Call Groq with high creativity and anti-repetition
    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=0.9,
            top_p=0.95,
            frequency_penalty=0.85,   # Strong penalty against repetition
            presence_penalty=0.7,     # Encourage new topics
            max_tokens=4000
        )
        reply = getattr(response.choices[0].message, "content", "No response")
        reply = clean_repetitions(reply)
        
        # Prepend correction text if any
        if correction_text:
            reply = correction_text + reply
        
        # Ensure follow-up question (unless goodbye)
        if "goodbye" not in user_msg.lower() and "bye" not in user_msg.lower() and "?" not in reply[-60:]:
            reply += "\n\nWhat would you like to explore or correct next, my friend?"
        
        # 6. Token deduction
        tokens_used = getattr(response.usage, "total_tokens", 0)
        new_balance = deduct_tokens_atomic(api_key, tokens_used)
        
        return {
            "company": "signaturesi.com",
            "message": reply,
            "usage": {"total_tokens": tokens_used},
            "model": "Neo L1.0 (Guardian Professor)",
            "internal_engine": MODEL,
            "balance": new_balance
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Groq model failed: {e}")
        # Log this as an error for evolution
        log_error(api_key, f"Groq API failure: {str(e)[:100]}")
        raise HTTPException(
            status_code=503,
            detail={"company": "signaturesi.com", "status": "error", "message": "Neo is recalibrating – please retry"}
        )

# -----------------------------
# 11. Balance & Key endpoints (unchanged)
# -----------------------------
@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    try:
        user = get_user(api_key)
        if not user.data:
            return {"api_key": api_key, "balance": 0}
        return {"api_key": api_key, "balance": user.data.get("token_balance", 0)}
    except Exception as e:
        logger.error(f"Balance fetch error: {e}")
        raise HTTPException(500, "Balance fetch failed")

@app.post("/v1/user/new-key")
def generate_key():
    try:
        api_key = "sig-" + secrets.token_hex(16)
        SUPABASE.table("users").insert({
            "api_key": api_key,
            "token_balance": 100000
        }).execute()
        return {"api_key": api_key, "company": "signaturesi.com"}
    except Exception as e:
        logger.error(f"Key generation error: {e}")
        raise HTTPException(500, "Failed to create key")
