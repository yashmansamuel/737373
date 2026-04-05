import os
import logging
import secrets
import asyncio
from typing import List, Tuple
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
logger = logging.getLogger("Neo-L1.0-Core")

required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "GROQ_API_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing required environment variable: {var}")

app = FastAPI(title="Neo L1.0 Engine - Field Specialized")

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
# 2. Field Detection & Specialized Prompts
# -----------------------------
def detect_field(user_query: str) -> str:
    """
    Classify the query into one of several fields.
    Returns a field key: 'medical', 'legal', 'technical', 'financial', 'general'
    """
    query_lower = user_query.lower()
    # Simple keyword‑based detection (can be upgraded to a lightweight ML model)
    medical_keywords = ["diagnosis", "symptom", "treatment", "medication", "disease", "doctor", "hospital", "patient", "therapy", "surgery"]
    legal_keywords = ["law", "legal", "attorney", "court", "lawsuit", "contract", "plaintiff", "defendant", "statute", "regulation"]
    technical_keywords = ["code", "algorithm", "programming", "api", "database", "network", "server", "frontend", "backend", "cloud", "docker", "kubernetes"]
    financial_keywords = ["investment", "stock", "market", "tax", "budget", "revenue", "profit", "expense", "loan", "mortgage", "crypto", "finance"]

    if any(kw in query_lower for kw in medical_keywords):
        return "medical"
    if any(kw in query_lower for kw in legal_keywords):
        return "legal"
    if any(kw in query_lower for kw in technical_keywords):
        return "technical"
    if any(kw in query_lower for kw in financial_keywords):
        return "financial"
    return "general"

def get_field_specific_prompt(field: str, base_mode_prompt: str) -> str:
    """
    Combine the field‑specific expertise with the base mode prompt.
    The field prompt is injected as a set of rules that the AI must follow.
    """
    field_prompts = {
        "medical": """You are an expert medical AI with deep knowledge of clinical practice, diagnostics, pharmacology, and evidence‑based medicine.
- Always emphasize patient safety and disclaim that your advice does not replace a real doctor.
- Provide structured reasoning: symptoms → differential diagnosis → recommended tests → possible treatments.
- Use medical terminology with clear explanations.
- Cite relevant guidelines or studies when possible (e.g., WHO, CDC, NICE).
- Be careful with dosages – never suggest specific medication doses without a real prescription context.
""",
        "legal": """You are an expert legal AI with knowledge of contract law, torts, civil procedure, criminal law, and statutory interpretation.
- Clarify that you are not a licensed attorney and your advice does not constitute legal counsel.
- Structure answers as: issue → relevant laws → analysis → possible outcomes.
- Reference legal principles, statutes, or case law (general, not jurisdiction‑specific unless given).
- Highlight risks and ambiguous points.
- Encourage consultation with a real lawyer for binding decisions.
""",
        "technical": """You are a senior software engineer and systems architect with expertise in programming, cloud infrastructure, algorithms, and DevOps.
- Provide code examples, configuration snippets, or architectural diagrams (in text).
- Explain trade‑offs (performance, security, cost, maintainability).
- Use step‑by‑step debugging or design reasoning.
- Reference best practices, design patterns, and modern tools.
- Be practical and actionable.
""",
        "financial": """You are a financial analyst with expertise in personal finance, corporate finance, investments, and taxation.
- Structure answers: context → analysis → recommendations → risk assessment.
- Use financial formulas, ratios, or models when relevant (e.g., ROI, NPV, DCF).
- Include caveats about market volatility and individual circumstances.
- Suggest diversification and risk management.
- Encourage professional advice for large decisions.
""",
        "general": """You are a high‑density information engine capable of multi‑step reasoning across any domain.
- Provide balanced, neutral, and thorough answers.
- Use analogies and examples to clarify complex topics.
- Show your reasoning explicitly.
- Address edge cases and common misconceptions.
"""
    }
    # Combine: first the field expert rules, then the base mode style (step/long/dense/human)
    combined = f"{field_prompts.get(field, field_prompts['general'])}\n\nAdditionally, follow these style guidelines:\n{base_mode_prompt}"
    return combined

# -----------------------------
# 3. Base Mode Prompts (unchanged but will be combined)
# -----------------------------
STEP_BY_STEP_PROMPT = """Provide a multi‑step, human‑like reasoning with logical flow, timelines, or hierarchical logic.
- Show each reasoning step clearly.
- Neutral tone; avoid ego statements.
- Be adaptive and dynamic based on query complexity.
- Provide structured insights, edge‑cases, and critical considerations.
"""

LONG_FORM_PROMPT = """Write a full‑length, detailed content with multi‑step human reasoning.
- Provide full explanations, examples, history, and implications.
- Show adaptive reasoning with structured stepwise approach.
- Address edge‑cases and nuanced considerations.
- Neutral, human‑like, readable style.
"""

HIGH_DENSITY_PROMPT = """Provide maximum insight with minimal words.
- Compress reasoning, logic, and information without losing accuracy.
- Use precise, technical vocabulary.
- Multi‑step reasoning and critical edge‑case analysis.
- Avoid filler or repetitive language.
- Use bullets, steps, or compact structured format.
"""

HUMANIZED_PROMPT = """Respond exactly like a thoughtful human would.
- Never say “I am an AI model” or give disclaimers.
- Show multi‑step reasoning, humor, creativity when relevant.
- Adaptive, natural responses based on query context.
- Responses should feel dynamic, intelligent, and comprehensive.
"""

# -----------------------------
# 4. Pydantic Models
# -----------------------------
class ChatRequest(BaseModel):
    model: str
    messages: List[dict]
    mode: str = "step"  # "step", "long", "dense", "human"

class BalanceResponse(BaseModel):
    api_key: str
    balance: int

# -----------------------------
# 5. Branding & Error Handlers
# -----------------------------
@app.get("/")
async def root():
    return {
        "company": "signaturesi.com",
        "engine": "Neo L1.0 Core (Field Specialized)",
        "status": "running",
        "deployment": "Jan 1, 2026"
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
# 6. Neural Context (Knowledge Engine)
# -----------------------------
def get_neural_context(user_query: str) -> str:
    """
    Fetch top 5 relevant lines from knowledge.txt
    to provide adaptive, multi‑step reasoning.
    """
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "knowledge.txt")
        if not os.path.exists(file_path):
            return ""

        query_words = [w.lower() for w in user_query.split() if len(w) > 3]
        matches = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_lower = line.lower()
                score = sum(word in line_lower for word in query_words)
                if score >= 1:
                    matches.append(line.strip())
                if len(matches) >= 5:
                    break

        return "\n".join(matches)

    except Exception as e:
        logger.error(f"Neural Context retrieval error: {e}")
        return ""

# -----------------------------
# 7. Helper Functions
# -----------------------------
def extract_content(msg):
    return getattr(msg, "content", "") or "No response"

def get_user(api_key: str):
    return SUPABASE.table("users") \
        .select("token_balance") \
        .eq("api_key", api_key) \
        .maybe_single() \
        .execute()

# -----------------------------
# 8. API Routes
# -----------------------------
@app.get("/v1/user/balance", response_model=BalanceResponse)
def get_balance(api_key: str):
    try:
        user = get_user(api_key)
        if not user.data:
            return {"api_key": api_key, "balance": 0}
        return {"api_key": api_key, "balance": user.data.get("token_balance", 0)}
    except Exception as e:
        logger.error(f"Balance Error: {e}")
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

@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid API key")

    api_key = authorization.replace("Bearer ", "")
    user = get_user(api_key)

    if not user.data:
        raise HTTPException(401, "User not found")

    balance = user.data["token_balance"]
    if balance <= 0:
        raise HTTPException(402, "No tokens left")

    user_msg = payload.messages[-1].get("content", "") if payload.messages else ""
    
    # --- NEW: Detect the field of the query ---
    field = detect_field(user_msg)
    logger.info(f"Detected field: {field} for query: {user_msg[:50]}...")

    neural_data = get_neural_context(user_msg)

    # Choose base mode prompt and parameters
    mode = payload.mode.lower()
    if mode == "long":
        base_prompt = LONG_FORM_PROMPT
        temperature = 0.7
        max_tokens = 4000
    elif mode == "dense":
        base_prompt = HIGH_DENSITY_PROMPT
        temperature = 0.5
        max_tokens = 1000
    elif mode == "human":
        base_prompt = HUMANIZED_PROMPT
        temperature = 0.7
        max_tokens = 1500
    else:  # step mode
        base_prompt = STEP_BY_STEP_PROMPT
        temperature = 0.5
        max_tokens = 2000

    # --- NEW: Combine field expertise with the mode style ---
    final_system_prompt = get_field_specific_prompt(field, base_prompt)

    # Build the messages
    final_messages = [
        {"role": "system", "content": final_system_prompt},
        {"role": "system", "content": "Integrate Neural Context strictly."}
    ]
    if neural_data:
        final_messages.append({"role": "system", "content": f"Neural Context:\n{neural_data}"})
    final_messages.extend(payload.messages)

    # Call Groq API & update balance
    try:
        response = GROQ.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        reply = extract_content(response.choices[0].message)
        tokens_used = getattr(response.usage, "total_tokens", 0)
        new_balance = max(0, balance - tokens_used)

        # Async update of user balance
        asyncio.create_task(asyncio.to_thread(
            lambda: SUPABASE.table("users")
            .update({"token_balance": new_balance})
            .eq("api_key", api_key)
            .execute()
        ))

        return {
            "company": "signaturesi.com",
            "message": reply,
            "usage": {"total_tokens": tokens_used},
            "model": "Neo L1.0",
            "internal_engine": MODEL,
            "balance": new_balance,
            "detected_field": field   # optional: inform the client which field was used
        }

    except Exception as e:
        logger.error(f"Model {MODEL} failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={"company": "signaturesi.com", "status": "error", "message": "Neo model failed"}
        )
