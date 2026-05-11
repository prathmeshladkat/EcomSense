# ─────────────────────────────────────────────────────────────────────────────
# Insight Generator Agent
# Job: find patterns in events using RAG and generate insights
# ─────────────────────────────────────────────────────────────────────────────

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from config.llm_config import llm_insight
from config.prompts import INSIGHT_GENERATOR_PROMPT
from tools.rag_tools import (
    retrieve_similar_errors,
    retrieve_similar_insights,
    store_insight_in_vector_db,
    store_user_event_in_vector_db,
)
from state import EcomSenseState, Insight
from datetime import datetime
import uuid
import json
import logging

logger = logging.getLogger(__name__)

# ── Build the agent ───────────────────────────────────────────────────────────
INSIGHT_TOOLS = [
    retrieve_similar_errors,
    retrieve_similar_insights,
    store_insight_in_vector_db,
    store_user_event_in_vector_db,
]

insight_agent = create_tool_calling_agent(
    llm=llm_insight,
    tools=INSIGHT_TOOLS,
    prompt=INSIGHT_GENERATOR_PROMPT,
)

insight_agent_executor = AgentExecutor(
    agent=insight_agent,
    tools=INSIGHT_TOOLS,
    max_iterations=6,
    verbose=True,
    handle_parsing_errors=True,
)


def insight_generator_node(state: EcomSenseState) -> dict:
    """
    LangGraph node for the Insight Generator Agent.
    Generates insights from current error clusters and segments.
    """
    error_clusters = state.get("error_clusters", [])
    segments = state.get("segments", {})
    existing_insights = state.get("insights", [])

    new_insights = []

    # ── Generate insight for each active error cluster ────────────────────
    for cluster in error_clusters[:3]:  # top 3 clusters only — cost control
        question = (
            f"Has the error '{cluster.error_message}' on the {cluster.page} page "
            f"happened before? What was the root cause and how was it resolved?"
        )

        # retrieve relevant past events BEFORE calling LLM
        # this is the RAG part — grounding the LLM with real data
        retrieved = retrieve_similar_errors.invoke({
            "query": question,
            "n_results": 5,
            "page_filter": cluster.page,
        })

        # build context string from retrieved documents
        context = _build_context_string(retrieved["results"])

        try:
            result = insight_agent_executor.invoke({
                "question": question,
                "context": context,       # injected into INSIGHT_GENERATOR_PROMPT
                "agent_scratchpad": [],
            })

            output_text = result.get("output", "{}")
            output_text = _clean_json(output_text)
            parsed = json.loads(output_text)

            insight = Insight(
                insight_id=f"ins_{uuid.uuid4().hex[:8]}",
                summary=parsed.get("insight", ""),
                pattern_type=parsed.get("pattern_type", "new"),
                confidence=parsed.get("confidence", "low"),
                supporting_event_ids=[
                    r.get("metadata", {}).get("cluster_id", "")
                    for r in retrieved["results"]
                ],
                created_at=datetime.now(),
            )

            if insight.summary:  # only store non-empty insights
                new_insights.append(insight)

                # store insight in ChromaDB so future runs can find it
                store_insight_in_vector_db.invoke({
                    "insight": {
                        "insight_id": insight.insight_id,
                        "summary": insight.summary,
                        "pattern_type": insight.pattern_type,
                        "confidence": insight.confidence,
                    }
                })

        except Exception as e:
            logger.warning(f"Insight generation failed for cluster {cluster.cluster_id}: {e}")
            continue

    # ── Generate segment insight if we have active segments ───────────────
    if segments:
        largest_segment = max(segments.values(), key=lambda s: s.size)

        seg_question = (
            f"What pattern explains why {largest_segment.size} users "
            f"are in the '{largest_segment.name}' segment with "
            f"avg cart value ₹{largest_segment.avg_cart_value}?"
        )

        similar = retrieve_similar_insights.invoke({
            "query": seg_question,
            "n_results": 3,
        })
        context = _build_context_string(similar["results"])

        try:
            result = insight_agent_executor.invoke({
                "question": seg_question,
                "context": context,
                "agent_scratchpad": [],
            })
            output_text = _clean_json(result.get("output", "{}"))
            parsed = json.loads(output_text)

            if parsed.get("insight"):
                seg_insight = Insight(
                    insight_id=f"ins_{uuid.uuid4().hex[:8]}",
                    summary=parsed["insight"],
                    pattern_type=parsed.get("pattern_type", "new"),
                    confidence=parsed.get("confidence", "low"),
                    supporting_event_ids=[],
                    created_at=datetime.now(),
                )
                new_insights.append(seg_insight)

        except Exception as e:
            logger.warning(f"Segment insight generation failed: {e}")

    logger.info(f"Generated {len(new_insights)} new insights")

    return {
        "insights": existing_insights + new_insights,
        "last_insight_run": datetime.now(),
        "current_task": "insight_complete",
    }


# ── Helper functions ──────────────────────────────────────────────────────────

def _build_context_string(results: list) -> str:
    """
    Converts retrieved ChromaDB results into a readable context string.
    This is what gets injected into {context} in the insight prompt.
    """
    if not results:
        return "No relevant historical data found."

    lines = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        lines.append(
            f"[{i}] {r.get('text', '')}\n"
            f"    Date: {meta.get('date', 'unknown')} | "
            f"Severity: {meta.get('severity', 'unknown')}"
        )
    return "\n\n".join(lines)


def _clean_json(text: str) -> str:
    """Strips markdown code fences from LLM output before JSON parsing."""
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        # take the part after the first fence
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()