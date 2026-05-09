# Tools used by  Error Detector Agent.
# These are the actual computation functions - the LLM calls these
# when it need real data, not when it needs to reason about what to do.

from langchain_core.tools import tool
from datetime import datetime
from collections import defaultdict
from config.llm_config import redis_client
import hashlib
import json

# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 — cluster_errors
# Groups a batch of raw JS error events into clusters.
# Same error firing 847 times → 1 cluster with count=847.
#
# How fingerprinting works:
# We hash (error_message + page) to create a cluster ID.
# Two errors with same message on same page → same cluster.
# ─────────────────────────────────────────────────────────────────────────────
@tool
def cluster_errors(events: list[dict]) -> list[dict]:
    """
    Group raw JS error events into cluster by fingerprint
    Return a lisyt of clusters, each with count and browser breakdown

    Args:
        events: list of raw error event dicts from Redis stream

    Returns:
        list of cluster dicts ready for revenue scoring
    """
    # dictionary to accumulate events per  fingerprint
    # defaultdict means we don't need to check if key exist first
    clusters: dict[str, dict] = defaultdict(lambda: {
        "count": 0,
        "browser_breakdown": defaultdict(int),
        "pages": set(),
        "first_seen": None,
    })

    for event in events:
        # only process JS error events 
        if event.get("type") != "js_error":
            continue 

        msg = event.get("message", "unknown error")
        page = event.get("page", "unknown")

        # fingerprint = hash of message + page
        # same error on same page always maps to same clusterID
        raw = f"{msg}::{page}"
        cluster_id = "c_"+ hashlib.md5(raw.encode()).hexdigest()[:8]

        c = clusters[cluster_id]
        c["count"] += 1
        c["cluster_id"] = cluster_id
        c["error_message"] = msg
        c["page"] = page 

        # track first occurance
        ts = event.get("ts")
        if ts and (c["first_seen"] is None or ts < c["first_seen"]):
            c["first_seen"] = ts

    # convert defaultdict to regular dict for JSON serialization
    result = []
    for cid, c in clusters.items():
        result.append({
            "cluster_id": c["cluster_id"],
            "error_message": c["error_message"],
            "page": c["page"],
            "count": c["count"],
            "browser_breakdown": dict(c["browser_breakdown"]),
            "first_seen": c.get("first_seen") or datetime.now().isoformat(),
        })
    result.sort(key=lambda x: x["count"], reverse=True)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 — score_revenue_impact
# Estimates hourly revenue at risk for an error cluster.
# Uses session count × avg cart value as the base formula.
# ─────────────────────────────────────────────────────────────────────────────

@tool
def score_revenue_impact(
    cluster_id: str,
    session_count: int,
    avg_cart_inr: float | None,
    page: str,
) -> dict:
    """ 
    Estimates revenue at risk per hour for an error cluster.

    Args:
        cluster_id: the cluster to score
        session_count: number of sessions affected 
        avg_cart_inr: average cart value in INR. Pass None if unknown.
        page: which page the error is on (checkout has highest weight)

    Returns:
        dict with revenue_impact_inr_per_hour and scoring details
    """

    # page weight - checkout errors are most critical
    # error on checkout directly block purchase
    page_weights = {
        "checkout":         1.0,
        "payment":          1.0,
        "cart":             0.6,
        "product":          0.3,
        "product_listing":  0.2,
        "homepage":         0.1
    }

    #default weight for unknown page
    weight = page_weights.get(page.lower(), 0.2)

    if avg_cart_inr is None or avg_cart_inr <= 0:
        return {
            "cluster_id": cluster_id,
            "revenue_impact_inr_per_hour": -1,
            "reason": "avg_cart_inr not available — cannot estimate",
            "page_weight": weight,
        }
    
    ASSUMED_CONVERSION_RATE = 0.10
    estimated_hourly = int(
        session_count * avg_cart_inr * weight * ASSUMED_CONVERSION_RATE
    )

    return {
        "cluster_id": cluster_id,
        "revenue_impact_inr_per_hour": estimated_hourly,
        "session_count": session_count,
        "avg_cart_inr": avg_cart_inr,
        "page_weight": weight,
        "reason": f"estimated: {session_count} sessions × ₹{avg_cart_inr} × {weight} weight × {ASSUMED_CONVERSION_RATE} CVR",
    }

# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 — get_error_baseline
# Gets the normal (baseline) error rate for a page from Redis.
# Used to detect anomalies — is this error rate unusual or normal?
# ─────────────────────────────────────────────────────────────────────────────
@tool
def get_error_baseline(page: str) -> dict:
    """ 
    Gets the normal error rate baseline for a page from Redis.
    Used to determine if current error count is an anomaly,

    Args: 
        page: page name to check baseline for 

    Returns:
        dict with baseline_count, current_window_count is_anomaly
    """
    baseline_key = f"baseline:errors:{page}"
    current_key = f"current:errors:{page}"

    try:
        baseline = redis_client.get(baseline_key)
        current = redis_client.get(current_key)

        baseline_count = int(baseline) if baseline else 0
        current_count  = int(current) if current else 0

        # anomaly = current is more than 3x the baseline
        is_anomaly = (
            current_count > (baseline_count * 3)
            and current_count > 10
        )

        return {
            "page": page,
            "baseline_count": baseline_count,
            "current_count": current_count,
            "multiplier": round(current_count / max(baseline_count, 1), 2),
            "is_anomaly": is_anomaly,

        }
    except Exception as e:
        # if Redis fails, return safe defaults — don't crash the agent
        return {
            "page": page,
            "baseline_count": 0,
            "current_count": 0,
            "multiplier": 1.0,
            "is_anomaly": False,
            "error": str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 4 — update_error_baseline
# Updates the rolling baseline count in Redis.
# Called after each event batch to keep baselines current.
# ─────────────────────────────────────────────────────────────────────────────

@tool
def update_error_baseline(page: str, count: int) -> dict:
    """ 
    Updates the rolling error baseline for a page in redis.

    Args:
        page: page name
        count: current error count to record

    Returns:
        confirmation dict
    """
    try:
        baseline_key = f"baseline:errors:{page}"
        current_key = f"current:errors:{page}"

        # store current as the new baseline with 30min expiry
        redis_client.setex(baseline_key, 1800, count)
        redis_client.setex(current_key, 300, count)  # current expires in 5min

        return {"page": page, "baseline_updated": count, "status": "ok"}
    except Exception as e:
        return {"page": page, "status": "error", "error": str(e)}
