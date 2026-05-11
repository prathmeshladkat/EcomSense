# ─────────────────────────────────────────────────────────────────────────────
# Alert Dispatcher Agent
# Job: format error clusters into Slack messages and send them
# ─────────────────────────────────────────────────────────────────────────────

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from config.llm_config import llm_fast
from config.prompts import ALERT_DISPATCHER_PROMPT
from tools.alert_tools import (
    check_alert_cooldown,
    send_slack_alert,
    set_alert_cooldown,
     log_alert_record
)
from state import EcomSenseState, AlertRecord
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

ALERT_TOOLS = [
    check_alert_cooldown,
    set_alert_cooldown,
    send_slack_alert,
    log_alert_record
]

alert_agent = create_tool_calling_agent(
    llm = llm_fast,
    tools = ALERT_TOOLS,
    prompt = ALERT_DISPATCHER_PROMPT
)

alert_agent_executor = AgentExecutor(
    agent=alert_agent,
    tools=ALERT_TOOLS,
    max_iterations=4,
    verbose=True,
    handle_parsing_errors=True,
)

def alert_dispatcher_node(state: EcomSenseState) -> dict:
    """ 
    LangGraph node for the Alert Dispatcher Agent.
    Reads error_clusters from state, sends alerts for critical/high ones.
    """
    clusters = state.get("error_clusters", [])
    existing_alerts = state.get("alerts_sent", [])

    # only alert on critical and high severity clusters
    # medium and low go into the dashboard but don't wake anyone up
    urgent_clusters = [
        c for c in clusters 
        if c.severity in ("critical", "high")
    ]

    if not urgent_clusters:
        logger.info("No urgent clusters to alert on")
        return {"current_task":"idle"}
    
    new_alerts = []

    for cluster in urgent_clusters:
        # build input for the agent — structured context
        agent_input = (
            f"Process this error cluster and send a Slack alert if needed.\n\n"
            f"Cluster details:\n"
            f"- cluster_id: {cluster.cluster_id}\n"
            f"- error: {cluster.error_message}\n"
            f"- page: {cluster.page}\n"
            f"- severity: {cluster.severity}\n"
            f"- sessions affected: {cluster.count}\n"
            f"- revenue impact: ₹{cluster.revenue_impact_inr}/hr\n"
            f"- confidence: {cluster.confidence}\n"
            f"- browser breakdown: {cluster.browser_breakdown}\n\n"
            f"Steps:\n"
            f"1. Check cooldown for cluster_id={cluster.cluster_id}\n"
            f"2. If cooldown active, skip and say so\n"
            f"3. If no cooldown, send_slack_alert with all details\n"
            f"4. Set cooldown after sending\n"
            f"5. Log the alert record"
        )

        try:
            result = alert_agent_executor.invoke({
                "input": agent_input,
                "agent_scratchpad": [],
            })

            output = result.get("output", "")

            # if agent sent an alert, record it in state
            if "sent" in output.lower() or "cooldown" not in output.lower():
                alert_record = AlertRecord(
                    cluster_id=cluster.cluster_id,
                    sent_at=datetime.now(),
                    channel="slack",
                    severity=cluster.severity,
                    message_preview=f"{cluster.severity}: {cluster.error_message}"[:100],
                )
                new_alerts.append(alert_record)

        except Exception as e:
            logger.error(f"Alert Dispatcher failed for cluster {cluster.cluster_id}: {e}")
            continue
    
    return {
        "alerts_sent": existing_alerts + new_alerts,
        "current_task": "alerting_complete",
    }


