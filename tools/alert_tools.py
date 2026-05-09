from langchain_core.tools import tool
from datetime import datetime
from config.llm_config import redis_client
from config.settings import settings
import requests
import json


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 — check_alert_cooldown
# Before sending any alert, check if we already sent one for this cluster.
# Prevents alert fatigue — no repeat alerts within 15 minutes.
# Cooldown state is stored in Redis with TTL.
# ─────────────────────────────────────────────────────────────────────────────

@tool
def check_alert_cooldown(cluster_id: str) -> dict:
    """
    Checks if an alert was recently sent for this error cluster.
    Returns True if we should skip this alert (cooldown active).

    Args:
        cluster_id: the error cluster to check

    Returns:
        dict with cooldown_active bool and time_remaining_seconds
    """
    cooldown_key = f"alert_cooldown:{cluster_id}"

    try:
        ttl = redis_client.ttl(cooldown_key)

        if ttl > 0:
            # cooldown active — key exists and has time remaining
            return {
                "cooldown_active": True,
                "time_remaining_seconds": ttl,
                "cluster_id": cluster_id,
            }
        else:
            # no cooldown — safe to send alert
            return {
                "cooldown_active": False,
                "time_remaining_seconds": 0,
                "cluster_id": cluster_id,
            }
    except Exception as e:
        # if Redis fails, allow the alert — better to over-alert than miss one
        return {"cooldown_active": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 — set_alert_cooldown
# After sending an alert, set a cooldown so we don't send again for 15 min.
# ─────────────────────────────────────────────────────────────────────────────

@tool
def set_alert_cooldown(cluster_id: str) -> dict:
    """
    Sets a 15-minute cooldown for an error cluster after alerting.
    Call this immediately after successfully sending an alert.

    Args:
        cluster_id: the error cluster that was just alerted

    Returns:
        confirmation dict
    """
    cooldown_key = f"alert_cooldown:{cluster_id}"
    cooldown_seconds = settings.alert_cooldown_minutes * 60  # 15 * 60 = 900s

    try:
        redis_client.setex(cooldown_key, cooldown_seconds, "sent")
        return {
            "cooldown_set": True,
            "cluster_id": cluster_id,
            "expires_in_seconds": cooldown_seconds,
        }
    except Exception as e:
        return {"cooldown_set": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 — send_slack_alert
# Sends a formatted message to Slack via webhook.
# Uses Block Kit format for rich formatting.
# ─────────────────────────────────────────────────────────────────────────────

@tool
def send_slack_alert(
    cluster_id: str,
    severity: str,
    error_message: str,
    page: str,
    session_count: int,
    revenue_impact_inr: int,
    reasoning: str,
    severity_emoji: str = "🟠",
) -> dict:
    """
    Sends a formatted Slack alert for an error cluster.

    Args:
        cluster_id: unique cluster identifier
        severity: critical/high/medium/low
        error_message: the JS error message
        page: affected page
        session_count: number of affected sessions
        revenue_impact_inr: estimated revenue at risk per hour
        reasoning: one sentence root cause analysis
        severity_emoji: emoji matching severity level

    Returns:
        dict with success status and Slack response
    """
    # if no webhook configured (eg in dev), just log and return
    if not settings.slack_webhook_url:
        print(f"[SLACK MOCK] {severity_emoji} {severity.upper()}: {error_message}")
        return {"sent": False, "reason": "no webhook configured — logged to console"}

    # format revenue impact for display
    if revenue_impact_inr > 0:
        revenue_str = f"₹{revenue_impact_inr:,}/hr at risk"
    else:
        revenue_str = "Revenue impact unknown"

    # Slack Block Kit message structure
    # blocks = visual components stacked vertically
    blocks = [
        {
            # header with severity emoji and level
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{severity_emoji} {severity.upper()} — EcomSense Alert",
            },
        },
        {
            # main details section
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Error:*\n`{error_message}`"},
                {"type": "mrkdwn", "text": f"*Page:*\n{page}"},
                {"type": "mrkdwn", "text": f"*Sessions Affected:*\n{session_count:,}"},
                {"type": "mrkdwn", "text": f"*Revenue Impact:*\n{revenue_str}"},
            ],
        },
        {"type": "divider"},
        {
            # reasoning / root cause
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Analysis:* {reasoning}",
            },
        },
        {
            # footer with cluster ID and timestamp
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"Cluster: `{cluster_id}` · "
                        f"Detected: {datetime.now().strftime('%H:%M:%S')} · "
                        f"EcomSense"
                    ),
                }
            ],
        },
    ]

    try:
        response = requests.post(
            settings.slack_webhook_url,
            json={"blocks": blocks},
            timeout=10,
        )

        if response.status_code == 200:
            return {
                "sent": True,
                "cluster_id": cluster_id,
                "channel": "slack",
            }
        else:
            return {
                "sent": False,
                "status_code": response.status_code,
                "error": response.text,
            }

    except Exception as e:
        return {"sent": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 4 — log_alert_record
# Saves a record of every alert sent to Redis.
# Used to track alert history and prevent duplicates across restarts.
# ─────────────────────────────────────────────────────────────────────────────

@tool
def log_alert_record(
    cluster_id: str,
    severity: str,
    channel: str,
    message_preview: str,
) -> dict:
    """
    Logs an alert record to Redis after successfully sending.

    Args:
        cluster_id: which cluster was alerted
        severity: severity level
        channel: slack/email
        message_preview: first 100 chars of what was sent

    Returns:
        confirmation dict
    """
    record = {
        "cluster_id":      cluster_id,
        "sent_at":         datetime.now().isoformat(),
        "channel":         channel,
        "severity":        severity,
        "message_preview": message_preview[:100],
    }

    try:
        # store in Redis list — keep last 100 alert records
        redis_client.lpush("alert_history", json.dumps(record))
        redis_client.ltrim("alert_history", 0, 99)

        return {"logged": True, "cluster_id": cluster_id}

    except Exception as e:
        return {"logged": False, "error": str(e)}