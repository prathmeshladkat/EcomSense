# ─────────────────────────────────────────────────────────────────────────────
# Segmentation Agent
# Job: process user events and build dynamic user segments
# ─────────────────────────────────────────────────────────────────────────────

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from config.llm_config import llm
from config.prompts import SEGMENTATION_PROMPT
from tools.segment_tools import (
    update_profile,
    get_profile,
    query_segment,
    get_segment_stats,
)
from tools.rag_tools import store_user_event_in_vector_db
from state import EcomSenseState, UserSegment
from datetime import datetime
from config.llm_config import redis_client
import json 
import logging

logger = logging.getLogger(__name__)

SEGMENT_TOOLS = [
    update_profile,
    get_profile,
    query_segment,
    get_segment_stats,
    store_user_event_in_vector_db
]

segment_agent = create_tool_calling_agent(
    llm=llm,
    tools=SEGMENT_TOOLS,
    prompt=SEGMENTATION_PROMPT,
)

segment_agent_executor = AgentExecutor(
    agent = segment_agent,
    tools = SEGMENT_TOOLS,
    max_iterations = 6 ,
    verbose = True,
    handle_parsing_errors = True,
)

def segmentation_node(state: EcomSenseState) -> dict:
    """ 
    LangGraph node for the Segmentation Agent.
    Processes user events and builds/updates segments.
    """
    raw_events = state.get("raw_events", [])
    existing_segments = state.get("segments", {})

    user_events = [
        e for e in raw_events
        if e.get("user_id") and e.get("type") in (
            "cart_added", "purchase", "page_view",
            "checkout_started", "identity"
        )
    ]

    for event in user_events:
        try:
            update_profile.invoke({
                "user_id": event["user_id"],
                "event": event,
            })
        except Exception as e:
            logger.warning(f"Profile update failed for {event.get('user_id')}: {e}")

    logger.info(f"Updated profiles for {len(user_events)} events")

    standard_queries = [
        "Find users who added to cart but did not purchase in the last 6 hours",
        "Find users who have not visited in the last 7 days but have purchased before",
        "Find users who started checkout but did not complete purchase in last 2 hours",
    ]

    new_segments = dict(existing_segments)

    for query in standard_queries:
        try:
            result = segment_agent_executor.invoke({
                "input": query,
                "agent_scratchpad": []
            })

            output_text = result.get("output", "{}")

            # clean and parse JSON output
            output_text = output_text.strip()
            if "```" in output_text:
                output_text = output_text.split("```")[1]
                if output_text.startswith("json"):
                    output_text = output_text[4:]

            parsed = json.loads(output_text)
             # query_segment tool returns actual user IDs
            segment_name = parsed.get("segment_name", "unnamed_segment")
            filter_params = parsed.get("filter", {})

            # run the actual query against Redis profiles
            query_result = query_segment.invoke({"filter_params": filter_params})

            if query_result["size"] > 0:
                # build UserSegment object
                segment = UserSegment(
                    name=segment_name,
                    user_ids=query_result["user_ids"],
                    avg_cart_value=query_result["avg_cart_value"],
                    size=query_result["size"],
                    created_at=datetime.now(),
                )
                new_segments[segment_name] = segment

                # cache segment in Redis for Campaign Writer to read
                redis_client.setex(
                    f"segment:{segment_name}",
                    3600,  # expires in 1 hour
                    json.dumps({
                        "name": segment_name,
                        "size": query_result["size"],
                        "avg_cart_value": query_result["avg_cart_value"],
                        "user_ids": query_result["user_ids"][:50],  # cap for storage
                    })
                )

                logger.info(f"Segment '{segment_name}': {query_result['size']} users")

        except Exception as e:
            logger.warning(f"Segment query failed: {e}")
            continue

    return {
        "segments": new_segments,
        "last_segment_run": datetime.now(),
        "current_task": "segmentation_complete",
    }


def run_nl_segment_query(query: str, state: EcomSenseState) -> dict:
    """
    Runs a one-off natural language segment query from the chat interface.
    Called directly by the chat handler, not by LangGraph routing.

    Example: user types "show me high value cart abandoners"
    """
    try:
        result = segment_agent_executor.invoke({
            "input": query,
            "agent_scratchpad": [],
        })
        return {"success": True, "output": result.get("output", "")}
    except Exception as e:
        return {"success": False, "error": str(e)}
