from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):

    openai_api_key: str
    model_name: str = "gpt-4o"
    model_fast: str = "gpt-4o-mini"
    model_embedding: str = "text-embedding-3-small"
    temperature_analysis: float = 0.1
    temperature_insight: float = 0.3             # insight + RAG agents
    temperature_creative: float = 0.8 #for campaign writing
    max_tokens: int = 1000

    # ---LangSmith-----
    langchain_tracing_v2: str = "true"
    langchain_api_key: str = ""
    langchain_project: str = "ecomsense"

    database_url: str

    # ── Redis (Upstash) ────
    upstash_redis_url: str
    upstash_redis_password: str = ""

    # ---ChromaDB-----
    chroma_path: str = "./chroma_db"
    retrieval_top_k: int = 5

    # -- Agent behavious ----------
    agent_max_iterations: int = 50
    alert_cooldown_minutes: int = 15    # don't re-alert same error cluster

    # ── Slack ─────────
    slack_webhook_url: str = ""

    # ── App ───────────
    environment: str = "development"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()