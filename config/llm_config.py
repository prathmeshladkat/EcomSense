from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma 
import redis
from config.settings import settings

# Primary model - used by mist agents for reasoning and analysis 
#temperature = 0.1 -> near-deterministic outputs, good for data analysis tasks
llm = ChatOpenAI(
    model=settings.model_name,
    temperature=settings.temperature_analysis,
    max_tokens = settings.max_tokens,
    api_key=settings.openai_api_key,
)

# fast cheap model - used for simple task like scopr checks , classififcation 
#gpt-4o-mini cost ~20x less than gpt-4o

llm_fast = ChatOpenAI(
    model=settings.model_fast,
    temperature=0.0,
    max_tokens=200,
    api_key=settings.openai_api_key,
)


# Creative model — ONLY for Campaign Writer Agent
# higher temperature = more varied, human-sounding copy
llm_creative = ChatOpenAI(
    model=settings.model_name,
    temperature=settings.temperature_creative,
    max_tokens=500,
    api_key=settings.openai_api_key,
)

# Insight model — middle ground for RAG + pattern analysis
llm_insight = ChatOpenAI(
    model=settings.model_name,
    temperature=settings.temperature_insight,
    max_tokens=settings.max_tokens,
    api_key=settings.openai_api_key,
)

# Fallback chain — if gpt-4o fails (outage, rate limit), auto-falls to mini
# .with_fallbacks() is a LangChain wrapper — transparent to calling code
llm = llm.with_fallbacks([llm_fast])

# Embeddings 
# used to convert text -> vectors before storing in chromaDB
embeddings = OpenAIEmbeddings(
    model=settings.model_embedding,
    api_key=settings.openai_api_key,
)

# -----ChromaDB -- 4 seperate collections, each for a different purpose---
import chromadb
chroma_client = chromadb.PersistentClient(path=settings.chroma_path)

# Collection 1 : Past js error clusters -> used by Insight Agent for RAG
error_collection = chroma_client.get_or_create_collection(
    name="error_history",
    metadata={"hnsw:space": "cosine"}
)


# Collection 2 : User behavioural summaries -> used by Segmentation Agent 
user_events_collection = chroma_client.get_or_create_collection(
    name="user_events",
    metadata={"hnsw:space": "cosine"}
)

# Collection 3 : generated insights -> used to detect recurring patterns
insights_collection = chroma_client.get_or_create_collection(
    name="insights",
    metadata={"hnsw:space": "cosine"}
)

campaign_collection = chroma_client.get_or_create_collection(
    name="campaign_history",
    metadata={"hnsw:space": "cosine"}
)

# Redis Connection

redis_client = redis.from_url(
    settings.upstash_redis_url,
    decode_responses=True,  # ensures we get strings back, not bytes
)


def verify_connections() -> dict:
    """
    Call this at app startup to confirm all services are reachable.
    Returns a status dict — useful for a /health endpoint in api.py.
    """
    status = {}

    # check Redis
    try:
        redis_client.ping()
        status["redis"] = "ok"
    except Exception as e:
        status["redis"] = f"error: {e}"

    # check ChromaDB
    try:
        collections = chroma_client.list_collections()
        status["chromadb"] = f"ok — {len(collections)} collections"
    except Exception as e:
        status["chromadb"] = f"error: {e}"

    # check OpenAI (light test — no actual LLM call)
    try:
        _ = embeddings.embed_query("test")
        status["openai"] = "ok"
    except Exception as e:
        status["openai"] = f"error: {e}"

    return status
