# ─────────────────────────────────────────────────────────────────────────────
# Error Detector Agent
# Job: analyze JS error clusters and score their revenue impact
# ─────────────────────────────────────────────────────────────────────────────

from langchain_core.messages import HumanMessage
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from config.llm_config import llm
from config.prompts import ERROR_DETECTOR_PROMPT
from tools.error_tools import (
    cluster_errors,
    score_revenue_impact,
    get_error_baseline,
    update_error_baseline,
)
from tools.rag_tools import store_error_in_vector_db
from state import EcomSenseState, ErrorCluster
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

# ── Build the agent ───────────────────────────────────────────────────────────
# create_tool_calling_agent wires: prompt + llm + tools
# The LLM now knows about all these tools and can call them by choice
ERROR_AGENT_TOOLS = [
    cluster_errors,
    score_revenue_impact,
    get_error_baseline,
    update_error_baseline,
    store_error_in_vector_db,
]

error_agent = create_tool_calling_agent(
    llm=llm,
    tools=ERROR_AGENT_TOOLS,
    prompt=ERROR_DETECTOR_PROMPT,
)

# AgentExecutor runs the agent in a loop until it decides it's done
# max_iterations prevents infinite tool-call loops
error_agent_executor = AgentExecutor(
    agent=error_agent,
    tools=ERROR_AGENT_TOOLS,
    max_iterations=5,           # agent can call at most 5 tools per run
    verbose=True,               # logs every tool call — useful for debugging
    handle_parsing_errors=True, # if LLM output is malformed, retry instead of crash
)


# ── Node function ─────────────────────────────────────────────────────────────
# This is what LangGraph calls. It receives the full state,
# runs the agent, and returns ONLY the keys it wants to update.
# LangGraph merges the return dict into the existing state.

def error_detector_node(state: EcomSenseState) -> dict:
    """
    LangGraph node for the Error Detector Agent.
    Reads raw_events from state, returns updated error_clusters.
    """
    logger.info("Error Detector Agent starting")

    raw_events = state.get("raw_events", [])

    # filter to only JS error events before sending to agent
    error_events = [e for e in raw_events if e.get("type") == "js_error"]

    if not error_events:
        logger.info("No error events to process")
        # return empty update — state unchanged for this key
        return {"current_task": "idle"}

    # build the input string for the agent
    # we pass structured context so the LLM doesn't have to guess
    agent_input = (
        f"Analyze these {len(error_events)} JS error events and cluster them.\n"
        f"Then score the revenue impact for each cluster.\n"
        f"Store each cluster in the vector database after scoring.\n\n"
        f"Events:\n{json.dumps(error_events[:20], indent=2)}"
        # cap at 20 events to stay within context window
    )

    try:
        result = error_agent_executor.invoke({
            "input": agent_input,
            # agent_scratchpad is where tool call results accumulate
            # LangGraph / LangChain manages this automatically
            "agent_scratchpad": [],
        })

        # parse the agent's final output into ErrorCluster objects
        # the agent returns a string — we parse it into structured data
        output_text = result.get("output", "{}")

        # clean JSON if model wrapped it in markdown code blocks
        output_text = output_text.strip()
        if output_text.startswith("```"):
            output_text = output_text.split("```")[1]
            if output_text.startswith("json"):
                output_text = output_text[4:]

        parsed = json.loads(output_text)

        # handle both single dict and list responses from the model
        if isinstance(parsed, dict):
            parsed = [parsed]

        # convert to validated ErrorCluster Pydantic objects
        clusters = []
        for item in parsed:
            try:
                cluster = ErrorCluster(
                    cluster_id=item.get("cluster_id", f"c_{datetime.now().timestamp()}"),
                    error_message=item.get("error_message", "unknown"),
                    count=item.get("count", 1),
                    page=item.get("page", "unknown"),
                    severity=item.get("severity", "medium"),
                    revenue_impact_inr=item.get("revenue_impact_inr_per_hour", -1),
                    confidence=item.get("confidence", "low"),
                    first_seen=datetime.now(),
                    browser_breakdown=item.get("browser_breakdown", {}),
                )
                clusters.append(cluster)
            except Exception as e:
                logger.warning(f"Failed to parse cluster: {e}")
                continue

        logger.info(f"Error Detector found {len(clusters)} clusters")

        # return ONLY what this agent updates
        # LangGraph merges this with the rest of the state
        return {
            "error_clusters": clusters,
            "current_task": "error_detection_complete",
        }

    except Exception as e:
        logger.error(f"Error Detector Agent failed: {e}")
        # graceful degradation — return empty clusters, don't crash the graph
        return {
            "error_clusters": [],
            "current_task": "error_detection_failed",
        }