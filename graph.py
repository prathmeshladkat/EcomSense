from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import AIMessage, HumanMessage
from datetime import datetime, timedelta
from config.settings import settings
from config.llm_config import redis_client, llm_fast
from config.prompts import CHAT_PROMPT 
from state import EcomSenseState, get_initial_state
from agents.error_detector import error_detector_node
from agents.alert_dispatcher import alert_dispatcher_node
from agents.segmentation import segmentation_node
from agents.insight_generator import insight_generator_node
from agents.campaign_writer import campaign_writer_node
import psycopg
import json
import logging

logger = logging.getLogger(__name__)

# NODE 1 - ingest_events

def ingest_events_node(state: EcomSenseState) -> dict:
    """ 
    Reads a batch of new events from the Redis stream.
    Returns up to 50 events per run to stay within context limits.
    """

    stream_key = "ecomsense:events"
    last_id_key = f"stream:last_id:{state.get('thread_id', 'default')}"

    try:
        last_id = redis_client.get(last_id_key) or "0"

        # XREAD: read up to 50 new messages after last_id
        # block=100 means wait up to 100ms for new messages
        messages = redis_client.xread(
            {stream_key: last_id},
            count=50,
            block=100,
        )

        if not messages:
            return {
                "raw_events": [],
                "current_task": "waiting_for_events",
                "iteration_count": state.get("iteration_count", 0) + 1,
            }
        
        events = []
        newest_id = last_id

        for stream_name, stream_messages in messages:
            for msg_id, msg_data in stream_messages:
                try:
                    # Redis stores everything as strings — parse JSON payload
                    if "payload" in msg_data:
                        event = json.loads(msg_data["payload"])
                    else:
                        # fallback: msg_data itself is the event
                        event = dict(msg_data)

                    events.append(event)
                    newest_id = msg_id  # track latest ID we've processed

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse event {msg_id}: {e}")
                    continue

        if newest_id != last_id:
            redis_client.set(last_id_key, newest_id)

        logger.info(f"Ingested {len(events)} events from Redis stream")

        return {
            "raw_events": events,
            "current_task": "events_ingested",
            "iteration_count": state.get("iteration_count", 0) + 1,

        }

    except Exception as e:
        logger.error(f"Event ingestion failed: {e}")
        return {
            "raw_events": [],
            "current_task": "ingestion_error",
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
    
# NODE 2 - orchestrator
def orchestrator_node(state: EcomSenseState) -> dict:
    """
    Reads current state and decides which agents need to run.
    Returns updated current_task - the conditional router reads this.
    """
    raw_events =state.get("raw_events", [])
    error_clusters = state.get("error_clusters", [])
    segments = state.get("segments", {})
    last_insight_run = state.get("last_insight_run")
    last_segment_run = state.get("last_segment_run")
    iteration_count = state.get("iteration_count", 0)
    now = datetime.now()

    if iteration_count > settings.agent_max_iterations:
        logger.warning("Max iterations reached — stopping graph")
        return {"current_task": "max_iterations_reached"}

    # --Decide what to run --------
    #Error detection has high priority
    error_events = [e for e in raw_events if e.get("type") == "js_error"]
    if len(error_events) >= 3:
        logger.info(f"Orchestrator: routing to error_detector ({len(error_events)} errors)")
        return {"current_task": "run_error_detector"}
    

    # Priority 2: Critical/high clusters exist → check if alert needed
    urgent_clusters = [
        c for c in error_clusters
        if c.severity in ("critical", "high") and not c.resolved
    ]
    if urgent_clusters:
        logger.info(f"Orchestrator: routing to alert_dispatcher ({len(urgent_clusters)} urgent)")
        return {"current_task": "run_alert_dispatcher"}

    # Priority 3: User events exist → update segments
    user_events = [
        e for e in raw_events
        if e.get("user_id") and e.get("type") in (
            "cart_added", "purchase", "checkout_started"
        )
    ]
    # also run segmentation if it's been more than 5 minutes
    segment_stale = (
        last_segment_run is None
        or (now - last_segment_run).total_seconds() > 300
    )
    if user_events or segment_stale:
        logger.info("Orchestrator: routing to segmentation")
        return {"current_task": "run_segmentation"}

    # Priority 4: Run insight generation every 10 minutes
    insight_stale = (
        last_insight_run is None
        or (now - last_insight_run).total_seconds() > 600
    )
    if insight_stale and (error_clusters or segments):
        logger.info("Orchestrator: routing to insight_generator")
        return {"current_task": "run_insight_generator"}

    # Priority 5: Large enough segments exist → write campaigns
    large_segments = [
        s for s in segments.values()
        if s.size >= 10
        and (s.campaign_sent_at is None
             or (now - s.campaign_sent_at).total_seconds() > 86400)
    ]
    if large_segments:
        logger.info(f"Orchestrator: routing to campaign_writer ({len(large_segments)} segments)")
        return {"current_task": "run_campaign_writer"}

    # Nothing urgent — idle until next event batch
    logger.info("Orchestrator: nothing to do, idling")
    return {"current_task": "idle"}


# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL ROUTER FUNCTION
# This function is what LangGraph calls to decide which node comes next.
# It reads current_task from state and returns the next node name.
# ─────────────────────────────────────────────────────────────────────────────

def route_from_orchestrator(state: EcomSenseState) -> str:
    """
    LangGraph conditional edge function.
    Returns the name of the next node to run based on current_task.
    """
    task = state.get("current_task", "idle")

    routing_map = {
        "run_error_detector": "error_detector",
        "run_alert_dispatcher":  "alert_dispatcher",
        "run_segmentation":      "segmentation",
        "run_insight_generator": "insight_generator",
        "run_campaign_writer":   "campaign_writer",
        # terminal conditions — stop the graph
        "idle":                  END,
        "max_iterations_reached": END,
        "waiting_for_events":    END,
    }

    next_node = routing_map.get(task, END)
    logger.info(f"Router: '{task}' -> '{next_node}'")
    return next_node

# ─────────────────────────────────────────────────────────────────────────────
# NODE 3 — human_review
# This node is where the graph PAUSES for human input.
# LangGraph's interrupt_before mechanism stops execution here
# and waits until the human approves or rejects via the Streamlit UI.
# ─────────────────────────────────────────────────────────────────────────────

def human_review_node(state: EcomSenseState) -> dict:
    """
    Pauses the graph for human campaign approval.
    LangGraph interrupts BEFORE this node runs - so this function
    only executes AFTER the human has given input via the UI.
    """
    pending = state.get("pending_campaigns", [])

    #check if any campaigns were approved during the HITL pause
    approved = [c for c in pending if c.status == "approved"]
    rejected = [c for c in pending if c.status == "rejected"]

    logger.info(
        f"Human review complete: {len(approved)} approved, "
        f"{len(rejected)} rejected"
    )

    updated = []
    for c in pending:
        if c.status == "approved":
            c.status = "sent"
        updated.append(c)

    return {
        "pending_campaigns": updated,
        "current_task": "review_complete",
    }

# ─────────────────────────────────────────────────────────────────────────────
#NODE 4 - chat_node
# ─────────────────────────────────────────────────────────────────────────────

def chat_node(state: EcomSenseState) -> dict:
    """
    Conversational interface node.
    Answer user questions about current system rate.
    """

    messages = state.get("messages", [])

    state_summary = _build_state_summary(state)

    try:
        # format the prompt with current state and message history
        formatted = CHAT_PROMPT.format_messages(
            state_summary=state_summary,
            messages=messages[:-1],      # all messages except the latest
            input=messages[-1].content if messages else "",
        )

        response = llm_fast.invoke(formatted)

        # add assistant response to message history
        # add_messages reducer in state automatically appends
        return {
            "messages": [AIMessage(content=response.content)],
            "current_task": "chat_complete",
        }

    except Exception as e:
        logger.error(f"Chat node failed: {e}")
        error_msg = "I encountered an error processing your question. Please try again."
        return {
            "messages": [AIMessage(content=error_msg)],
            "current_task": "chat_error",
        }


def _build_state_summary(state: EcomSenseState) -> str:
    """
    Converts current state into a readable text summary for the chat agent.
    This is what the LLM reads to answer questions about the system.
    """
    lines = []

    # error clusters summary
    clusters = state.get("error_clusters", [])
    if clusters:
        critical = [c for c in clusters if c.severity == "critical"]
        lines.append(f"ACTIVE ERRORS: {len(clusters)} clusters total, {len(critical)} critical")
        for c in clusters[:3]:  # top 3 only
            lines.append(
                f"  - [{c.severity.upper()}] {c.error_message} "
                f"on {c.page} | {c.count} sessions | "
                f"₹{c.revenue_impact_inr}/hr"
            )
    else:
        lines.append("ACTIVE ERRORS: none")

    # segments summary
    segments = state.get("segments", {})
    if segments:
        lines.append(f"USER SEGMENTS: {len(segments)} active")
        for name, seg in list(segments.items())[:3]:
            lines.append(
                f"  - {name}: {seg.size} users, "
                f"avg cart ₹{seg.avg_cart_value}"
            )
    else:
        lines.append("USER SEGMENTS: none computed yet")

    # insights summary
    insights = state.get("insights", [])
    if insights:
        lines.append(f"INSIGHTS: {len(insights)} generated")
        for ins in insights[:2]:
            lines.append(f"  - [{ins.confidence}] {ins.summary}")
    else:
        lines.append("INSIGHTS: none generated yet")

    # campaigns summary
    campaigns = state.get("pending_campaigns", [])
    pending = [c for c in campaigns if c.status == "pending_review"]
    if pending:
        lines.append(f"CAMPAIGNS AWAITING APPROVAL: {len(pending)}")
        for c in pending:
            lines.append(f"  - {c.segment_name}: '{c.push_copy}'")

    # alerts summary
    alerts = state.get("alerts_sent", [])
    if alerts:
        lines.append(f"ALERTS SENT THIS SESSION: {len(alerts)}")

    return "\n".join(lines)

def build_graph():
    """
    Assembles and compiles the LangGraph StateGraph.
    Returns a compiled graph with PostgreSQL checkpointing.

    Call this once at startup:
        app = build_graph()

    Then invoke with a thread_id:
        app.invoke(state, config={"configurable": {"thread_id": "user_1"}})
    """

    # ── PostgreSQL checkpointer setup ─────────────────────────────────────
    # PostgresSaver stores the full EcomSenseState to Postgres after
    # every node completes. If the server restarts, we resume from
    # the last checkpoint by passing the same thread_id.
    #
    # connection_kwargs: psycopg settings for the Neon serverless connection
    connection_kwargs = {
        "autocommit": True,
        # Neon requires SSL — sslmode=require
        "prepare_threshold": 0,         # required for Neon serverless
    }

    # create the connection
    # PostgresSaver.from_conn_string handles table creation automatically
    db_conn = psycopg.connect(
        settings.database_url,
        **connection_kwargs
    )
    checkpointer = PostgresSaver(db_conn)

    # create the LangGraph checkpoint tables in Postgres (runs once)
    checkpointer.setup()

    # ── Build the graph ───────────────────────────────────────────────────
    graph = StateGraph(EcomSenseState)

    # ── Add all nodes ─────────────────────────────────────────────────────
    # each string name maps to a node function
    graph.add_node("ingest_events",      ingest_events_node)
    graph.add_node("orchestrator",       orchestrator_node)
    graph.add_node("error_detector",     error_detector_node)
    graph.add_node("alert_dispatcher",   alert_dispatcher_node)
    graph.add_node("segmentation",       segmentation_node)
    graph.add_node("insight_generator",  insight_generator_node)
    graph.add_node("campaign_writer",    campaign_writer_node)
    graph.add_node("human_review",       human_review_node)
    graph.add_node("chat",               chat_node)

    # ── Set entry point ───────────────────────────────────────────────────
    # every graph run starts here
    graph.set_entry_point("ingest_events")

    # ── Fixed edges ───────────────────────────────────────────────────────
    # these always go to the same next node, no conditions
    graph.add_edge("ingest_events",  "orchestrator")

    # after each agent completes, loop back to orchestrator
    # orchestrator then decides if anything else needs to run
    graph.add_edge("error_detector",    "orchestrator")
    graph.add_edge("alert_dispatcher",  "orchestrator")
    graph.add_edge("segmentation",      "orchestrator")
    graph.add_edge("insight_generator", "orchestrator")

    # campaign writer always goes to human review
    # (not back to orchestrator — must wait for human)
    graph.add_edge("campaign_writer",   "human_review")

    # after human review — back to orchestrator for next cycle
    graph.add_edge("human_review",      "orchestrator")

    # chat is standalone — goes to END after responding
    graph.add_edge("chat",              END)

    # ── Conditional edge from orchestrator ────────────────────────────────
    # this is where the routing magic happens
    # route_from_orchestrator reads current_task and returns next node name
    graph.add_conditional_edges(
        "orchestrator",             # from this node
        route_from_orchestrator,    # call this function to decide
        {
            # map return values to node names
            "error_detector":    "error_detector",
            "alert_dispatcher":  "alert_dispatcher",
            "segmentation":      "segmentation",
            "insight_generator": "insight_generator",
            "campaign_writer":   "campaign_writer",
            END:                 END,
        }
    )

    # ── Compile with checkpointer and HITL interrupt ───────────────────────
    # interrupt_before=["human_review"] means:
    # LangGraph stops execution BEFORE entering human_review node
    # and saves state to Postgres. Execution resumes when the API
    # is called again with the same thread_id.
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],  # HITL pause point
    )

    logger.info("Graph compiled successfully with PostgreSQL checkpointer")
    return compiled


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# Used by api.py and dashboard.py to interact with the graph.
# ─────────────────────────────────────────────────────────────────────────────

def run_graph_cycle(app, thread_id: str) -> dict:
    """
    Runs one full cycle of the graph for a given thread.
    Called by the FastAPI background task every N seconds.

    Args:
        app: compiled LangGraph app from build_graph()
        thread_id: unique identifier for this user/session

    Returns:
        final state after this cycle
    """
    config = {"configurable": {"thread_id": thread_id}}

    # get or create initial state for this thread
    current_state = app.get_state(config)

    if current_state.values:
        # existing thread — resume from checkpoint
        state = current_state.values
    else:
        # new thread — start fresh
        state = get_initial_state(thread_id)

    try:
        # invoke runs the graph until it hits END or a HITL interrupt
        result = app.invoke(state, config=config)
        return result
    except Exception as e:
        logger.error(f"Graph cycle failed for thread {thread_id}: {e}")
        return state


def send_chat_message(app, thread_id: str, user_message: str) -> str:
    """
    Sends a user message to the chat node and returns the response.
    Called when user types in the Streamlit chat interface.

    Args:
        app: compiled LangGraph app
        thread_id: the user's session thread
        user_message: what the user typed

    Returns:
        assistant response string
    """
    config = {"configurable": {"thread_id": thread_id}}

    # get current state for this thread
    current_state = app.get_state(config)
    state = current_state.values if current_state.values else get_initial_state(thread_id)

    # add user message to state and route directly to chat node
    state["messages"] = state.get("messages", []) + [
        HumanMessage(content=user_message)
    ]
    state["current_task"] = "chat"

    try:
        # invoke just the chat node — skip the full pipeline
        result = app.invoke(
            state,
            config=config,
        )
        # get the last message — that's the assistant response
        messages = result.get("messages", [])
        if messages:
            return messages[-1].content
        return "No response generated."

    except Exception as e:
        logger.error(f"Chat failed for thread {thread_id}: {e}")
        return "I encountered an error. Please try again."


def resume_after_hitl(app, thread_id: str) -> dict:
    """
    Resumes the graph after human review is complete.
    Called by the FastAPI endpoint when human approves/rejects a campaign.

    LangGraph knows to resume from the human_review checkpoint
    because we pass the same thread_id that was interrupted.
    """
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # invoking with None input resumes from the last checkpoint
        result = app.invoke(None, config=config)
        return result
    except Exception as e:
        logger.error(f"HITL resume failed for thread {thread_id}: {e}")
        return {}