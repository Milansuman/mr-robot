from config import env
from langchain_groq import ChatGroq
from pydantic import SecretStr

# Primary model
llm = ChatGroq(
    api_key=SecretStr(env.GROQ_API_KEY),
    model="moonshotai/kimi-k2-instruct-0905",
    temperature=0.1,
    max_retries=2
)

# Fallback models for rate limit rotation
FALLBACK_MODELS = [
    ChatGroq(
        api_key=SecretStr(env.GROQ_API_KEY),
        model="openai/gpt-oss-120b",
        temperature=0.1,
        max_retries=2
    ),
    ChatGroq(
        api_key=SecretStr(env.GROQ_API_KEY),
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.1,
        max_retries=2
    ),
    ChatGroq(
        api_key=SecretStr(env.GROQ_API_KEY),
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_retries=2
    ),
]