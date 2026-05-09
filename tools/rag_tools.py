# ─────────────────────────────────────────────────────────────────────────────
# Tools used by the Insight Generator Agent.
# Handles storing and retrieving documents from ChromaDB.
# ─────────────────────────────────────────────────────────────────────────────

from langchain_core.tools import tool
from datetime import datetime
from config.llm_config import (
    embeddings,
    error_collection,
    user_events_collection,
    insights_collection,
    campaign_collection,
)
import json


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — text conversion functions
# Convert structured data to natural language BEFORE embedding.
# This is the most important decision in the RAG pipeline.
# ─────────────────────────────────────────────────────────────────────────────

def error_cluster_to_text(cluster: dict) -> str:
    """Convert error cluster dict to embeddable natural language."""
    browsers = ", ".join(
        f"{b}: {c}" for b, c in cluster.get("browser_breakdown", {}).items()
    )
    return (
        f"{cluster.get('error_message', 'unknown error')} "
        f"on {cluster.get('page', 'unknown')} page, "
        f"affecting {cluster.get('count', 0)} sessions, "
        f"severity {cluster.get('severity', 'unknown')}, "
        f"browsers: {browsers or 'unknown'}, "
        f"revenue impact ₹{cluster.get('revenue_impact_inr', -1)}/hour, "
        f"first seen {cluster.get('first_seen', 'unknown')}"
    )


def user_profile_to_text(profile: dict) -> str:
    """Convert user profile dict to embeddable natural language."""
    return (
        f"User {profile.get('user_id', 'unknown')} "
        f"from {profile.get('city', 'unknown city')} "
        f"on {profile.get('device', 'unknown device')}, "
        f"cart value ₹{profile.get('cart_value', 0)}, "
        f"last active {profile.get('last_seen', 'unknown')}, "
        f"purchase count: {profile.get('purchase_count', 0)}, "
        f"last page: {profile.get('last_page', 'unknown')}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 — store_error_in_vector_db
# Embeds an error cluster and stores it in ChromaDB error_history collection.
# Called by Error Detector after every new cluster.
# ─────────────────────────────────────────────────────────────────────────────

@tool
def store_error_in_vector_db(cluster: dict) -> dict:
    """
    Embeds and stores an error cluster in ChromaDB for future RAG retrieval.
    Call this after scoring a new error cluster.

    Args:
        cluster: error cluster dict with all fields

    Returns:
        confirmation dict
    """
    try:
        # convert to natural language BEFORE embedding
        text = error_cluster_to_text(cluster)

        # metadata allows pre-filtering before similarity search
        # eg: only search errors from last 7 days on checkout page
        metadata = {
            "severity":    cluster.get("severity", "unknown"),
            "page":        cluster.get("page", "unknown"),
            "date":        cluster.get("first_seen", datetime.now().isoformat())[:10],
            "resolved":    str(cluster.get("resolved", False)),
            "count":       str(cluster.get("count", 0)),
        }

        # embed and store — ChromaDB handles the vector creation internally
        error_collection.upsert(
            ids=[cluster["cluster_id"]],
            documents=[text],
            metadatas=[metadata],
        )

        return {
            "stored": True,
            "cluster_id": cluster["cluster_id"],
            "text_embedded": text[:100] + "...",  # preview for logging
        }

    except Exception as e:
        return {"stored": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 — store_insight_in_vector_db
# Stores a generated insight in ChromaDB.
# Future insights can retrieve past ones to detect recurring patterns.
# ─────────────────────────────────────────────────────────────────────────────

@tool
def store_insight_in_vector_db(insight: dict) -> dict:
    """
    Stores a generated insight in ChromaDB insights collection.

    Args:
        insight: insight dict with insight_id, summary, pattern_type etc

    Returns:
        confirmation dict
    """
    try:
        # insights are already natural language — embed directly
        text = insight.get("summary", "")

        metadata = {
            "pattern_type": insight.get("pattern_type", "new"),
            "confidence":   insight.get("confidence", "low"),
            "date":         datetime.now().isoformat()[:10],
        }

        insights_collection.upsert(
            ids=[insight["insight_id"]],
            documents=[text],
            metadatas=[metadata],
        )

        return {"stored": True, "insight_id": insight["insight_id"]}

    except Exception as e:
        return {"stored": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 — retrieve_similar_errors
# The core RAG retrieval tool.
# Takes a natural language question, finds the most relevant past errors.
# ─────────────────────────────────────────────────────────────────────────────

@tool
def retrieve_similar_errors(
    query: str,
    n_results: int = 5,
    page_filter: str | None = None,
    severity_filter: str | None = None,
) -> dict:
    """
    Retrieves similar past errors from ChromaDB using semantic search.
    Use this to answer questions like 'has this Stripe error happened before?'

    Args:
        query: natural language description of what you're looking for
        n_results: how many similar errors to return (default 5)
        page_filter: optional — only search errors on this page
        severity_filter: optional — only search errors of this severity

    Returns:
        dict with list of similar past errors and their details
    """
    try:
        # build metadata filter for hybrid retrieval
        # filter first (fast) then similarity search (slower) within filtered set
        where_filter = {}

        if page_filter and severity_filter:
            where_filter = {
                "$and": [
                    {"page": {"$eq": page_filter}},
                    {"severity": {"$eq": severity_filter}},
                ]
            }
        elif page_filter:
            where_filter = {"page": {"$eq": page_filter}}
        elif severity_filter:
            where_filter = {"severity": {"$eq": severity_filter}}

        # run the query
        query_kwargs = {
            "query_texts": [query],
            "n_results": n_results,
        }

        # only add where clause if we have filters
        # empty where clause causes ChromaDB error
        if where_filter:
            query_kwargs["where"] = where_filter

        results = error_collection.query(**query_kwargs)

        # format results for the LLM to read
        formatted = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "similarity_score": round(
                        1 - results["distances"][0][i], 3
                    ) if results.get("distances") else None,
                })

        return {
            "query": query,
            "results": formatted,
            "count": len(formatted),
        }

    except Exception as e:
        return {"query": query, "results": [], "count": 0, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 4 — retrieve_similar_insights
# Finds past insights similar to a new question.
# Helps detect recurring patterns — "we've seen this before"
# ─────────────────────────────────────────────────────────────────────────────

@tool
def retrieve_similar_insights(query: str, n_results: int = 3) -> dict:
    """
    Retrieves similar past insights from ChromaDB.
    Use this to detect if a pattern has been observed before.

    Args:
        query: what pattern you're looking for
        n_results: how many insights to return

    Returns:
        dict with list of similar past insights
    """
    try:
        results = insights_collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        formatted = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })

        return {"query": query, "results": formatted, "count": len(formatted)}

    except Exception as e:
        return {"query": query, "results": [], "count": 0, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 5 — store_user_event_in_vector_db
# Embeds and stores user behavioral summaries for semantic segment search.
# Called by Segmentation Agent when processing user events.
# ─────────────────────────────────────────────────────────────────────────────

@tool
def store_user_event_in_vector_db(profile: dict) -> dict:
    """
    Stores a user profile summary in ChromaDB for semantic segment queries.

    Args:
        profile: user profile dict from Redis

    Returns:
        confirmation dict
    """
    try:
        user_id = profile.get("user_id", "unknown")
        text = user_profile_to_text(profile)

        metadata = {
            "user_id":        user_id,
            "city":           profile.get("city", "unknown"),
            "device":         profile.get("device", "unknown"),
            "purchase_count": str(profile.get("purchase_count", 0)),
            "cart_value":     str(profile.get("cart_value", 0)),
        }

        user_events_collection.upsert(
            ids=[user_id],
            documents=[text],
            metadatas=[metadata],
        )

        return {"stored": True, "user_id": user_id}

    except Exception as e:
        return {"stored": False, "error": str(e)}