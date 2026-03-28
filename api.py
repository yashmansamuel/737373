import os
import logging
import secrets
import httpx
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# -----------------------------
# Logger Setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# FastAPI Init
# -----------------------------
app = FastAPI(title="Signaturesi Neo L1.0 API")

# -----------------------------
# CORS Middleware
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production me restrict karna
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# ENV VARIABLES
# -----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -----------------------------
# CLIENTS INIT
# -----------------------------
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("Connected to Supabase and Groq successfully.")
except Exception as e:
    logger.error(f"Initialization Error: {e}")
    raise RuntimeError("Cannot connect to services")

# -----------------------------
# SYSTEM PROMPT
# -----------------------------
SYSTEM_PROMPT = """
You are Neo L1.0, an advanced reasoning AI by Signaturesi.
You have access to web search. If needed, fetch latest info.
"""

# -----------------------------
# SIMPLE WEB SEARCH FUNCTION
# -----------------------------
async def web_search(query: str):
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(
                "https://duckduckgo.com/?q=" + query + "&format=json"
            )
            return res.text[:2000]  # limit
    except:
        return "Web search failed"

# -----------------------------
# HEALTH
# -----------------------------
@app.get("/")
def home():
    return {"status": "Online", "brand": "Signaturesi", "model": "Neo L1.0"}

# -----------------------------
# BALANCE
# -----------------------------
@app.get("/v1/user/balance")
def get_balance(api_key: str):
    response = supabase.table("users").select("token_balance").eq("api_key", api_key).execute()

    if not response.data:
        raise HTTPException(status_code=404, detail="API Key not found")

    return {
        "api_key": api_key,
        "balance": response.data[0]["token_balance"]
    }

# -----------------------------
# NEW KEY
# -----------------------------
@app.post("/v1/user/new-key")
async def generate_key(request: Request):
    new_key = "sig-live-" + secrets.token_urlsafe(16)

    try:
        supabase.table("users").insert({
            "api_key": new_key,
            "token_balance": 1000,
            "country": "Unknown"
        }).execute()

        return {"api_key": new_key, "balance": 1000}

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="Key creation failed")

# -----------------------------
# CHAT
# -----------------------------
@app.post("/v1/chat/completions")
async def chat_proxy(request: Request, authorization: str = Header(None)):

    # 🔐 AUTH
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API Key")

    user_api_key = authorization.replace("Bearer ", "")

    body = await request.json()
    model_name = body.get("model")

    if model_name != "Neo-L1.0":
        raise HTTPException(status_code=400, detail="Use Neo-L1.0")

    messages = body.get("messages")

    # 💰 CHECK BALANCE
    db = supabase.table("users").select("token_balance").eq("api_key", user_api_key).execute()

    if not db.data:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    balance = db.data[0]["token_balance"]

    if balance <= 0:
        raise HTTPException(status_code=402, detail="No balance")

    # 🌐 OPTIONAL WEB SEARCH
    last_user_msg = messages[-1]["content"]

    if "search:" in last_user_msg.lower():
        query = last_user_msg.replace("search:", "")
        web_data = await web_search(query)

        messages.append({
            "role": "system",
            "content": f"Web result:\n{web_data}"
        })

    # 🤖 GROQ CALL
    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            temperature=0.4
        )

        # 📉 TOKEN USAGE (approx)
        tokens_used = len(str(response)) // 4
        new_balance = max(0, balance - tokens_used)

        supabase.table("users").update({
            "token_balance": new_balance
        }).eq("api_key", user_api_key).execute()

        # 🧠 CUSTOM RESPONSE FORMAT
        return {
            "id": response.id,
            "object": "chat.completion",
            "model": "Neo-L1.0",
            "choices": response.choices
        }

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="AI failed")
