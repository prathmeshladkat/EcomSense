# ─────────────────────────────────────────────────────────────────────────────
# Tools used by the Segmentation Agent.
# Manages user profiles in Redis and queries them to build segments.
# ─────────────────────────────────────────────────────────────────────────────

from langchain_core.tools import tool
from datetime import datetime, timedelta
from config.llm_config import redis_client
import json


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 — update_profile
# Called every time a user event arrives.
# Builds and maintains the per-user profile in Redis.
# ─────────────────────────────────────────────────────────────────────────────

@tool
def update_profile(user_id: str, event: dict) -> dict:
    """
    Updates a user's profile in Redis based on a new event.
    Creates the profile if it doesn't exist yet.

    Args:
        user_id: unique user identifier
        event: event dict with type, page, value etc

    Returns:
        updated profile summary
    """
    profile_key = f"profile:{user_id}"
    events_key = f"events:{user_id}"
    event_type = event.get("type", "unknown")
    now = datetime.now().isoformat()

    try:
        # HSET updates individual fields without overwriting the whole profile
        # think of it like UPDATE SET in SQL
        redis_client.hset(profile_key, mapping={
            "user_id": user_id,
            "last_seen": now,
            "last_event": event_type,
        })

        # track event-specific fields
        if event_type == "cart_added":
            cart_value = event.get("cart_value", 0)
            redis_client.hset(profile_key, mapping={
                "cart_value": cart_value,
                "cart_added_at": now,
                "last_cart_page": event.get("page", "unknown"),
            })
            # increment cart add counter
            redis_client.hincrby(profile_key, "cart_add_count", 1)

        elif event_type == "purchase":
            redis_client.hset(profile_key, mapping={
                "last_purchase_at": now,
                "last_order_value": event.get("order_value", 0),
                "cart_value": 0,        # clear cart after purchase
            })
            redis_client.hincrby(profile_key, "purchase_count", 1)

        elif event_type == "page_view":
            redis_client.hincrby(profile_key, "page_view_count", 1)
            redis_client.hset(profile_key, "last_page", event.get("page", ""))

        elif event_type == "identify":
            # identify event enriches profile with user attributes
            attrs = event.get("attributes", {})
            if attrs:
                redis_client.hset(profile_key, mapping=attrs)

        # append event to user's event log (keep last 50 only)
        redis_client.lpush(events_key, json.dumps({
            "type": event_type,
            "ts": now,
            "page": event.get("page", ""),
        }))
        redis_client.ltrim(events_key, 0, 49)  # keep only last 50 events

        # profile expires in 30 days of inactivity
        redis_client.expire(profile_key, 86400 * 30)
        redis_client.expire(events_key, 86400 * 30)

        return {"user_id": user_id, "event_recorded": event_type, "status": "ok"}

    except Exception as e:
        return {"user_id": user_id, "status": "error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 — get_profile
# Retrieves a single user's full profile from Redis.
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_profile(user_id: str) -> dict:
    """
    Gets a user's full profile from Redis.

    Args:
        user_id: unique user identifier

    Returns:
        profile dict with all attributes and recent events
    """
    try:
        profile = redis_client.hgetall(f"profile:{user_id}")
        if not profile:
            return {"user_id": user_id, "found": False}

        # get recent events
        raw_events = redis_client.lrange(f"events:{user_id}", 0, 9)
        recent_events = [json.loads(e) for e in raw_events]

        return {
            **profile,
            "found": True,
            "recent_events": recent_events,
        }
    except Exception as e:
        return {"user_id": user_id, "found": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 — query_segment
# The most important segment tool.
# Queries all user profiles in Redis and returns IDs matching a filter.
# ─────────────────────────────────────────────────────────────────────────────

@tool
def query_segment(filter_params: dict) -> dict:
    """
    Queries user profiles in Redis to build a segment.
    Called by Segmentation Agent with a filter dict it generated.

    Args:
        filter_params: dict with filter criteria. Supported keys:
            - has_event: str (user must have this event type)
            - missing_event: str (user must NOT have this event recently)
            - time_window_hours: int (look back window)
            - min_cart_value: float
            - min_purchase_count: int
            - inactive_days: int

    Returns:
        dict with matching user_ids and segment stats
    """
    try:
        # scan all profile keys in Redis
        # in production with millions of users, you'd use a proper DB query
        # for portfolio scale, Redis SCAN is fine
        matching_users = []
        cursor = 0
        now = datetime.now()

        while True:
            # SCAN is non-blocking — safer than KEYS * in production
            cursor, keys = redis_client.scan(
                cursor=cursor,
                match="profile:*",
                count=100
            )

            for key in keys:
                profile = redis_client.hgetall(key)
                if not profile:
                    continue

                user_id = profile.get("user_id", key.replace("profile:", ""))
                passes = True

                # ── filter: must have done a specific event ────────────────
                if "has_event" in filter_params:
                    required_event = filter_params["has_event"]
                    # check event log for this event type
                    raw_events = redis_client.lrange(f"events:{user_id}", 0, 49)
                    event_types = [
                        json.loads(e).get("type") for e in raw_events
                    ]
                    if required_event not in event_types:
                        passes = False

                # ── filter: must NOT have done a specific event recently ───
                if passes and "missing_event" in filter_params:
                    missing = filter_params["missing_event"]
                    window_hours = filter_params.get("time_window_hours", 24)
                    cutoff = (now - timedelta(hours=window_hours)).isoformat()

                    raw_events = redis_client.lrange(f"events:{user_id}", 0, 49)
                    # check if missing_event happened after cutoff
                    recent_of_type = [
                        json.loads(e) for e in raw_events
                        if json.loads(e).get("type") == missing
                        and json.loads(e).get("ts", "") > cutoff
                    ]
                    if recent_of_type:
                        passes = False  # user DID do the missing event — exclude

                # ── filter: minimum cart value ─────────────────────────────
                if passes and "min_cart_value" in filter_params:
                    cart_val = float(profile.get("cart_value", 0))
                    if cart_val < filter_params["min_cart_value"]:
                        passes = False

                # ── filter: minimum purchase count ─────────────────────────
                if passes and "min_purchase_count" in filter_params:
                    purchase_count = int(profile.get("purchase_count", 0))
                    if purchase_count < filter_params["min_purchase_count"]:
                        passes = False

                # ── filter: inactive for N days ────────────────────────────
                if passes and "inactive_days" in filter_params:
                    last_seen = profile.get("last_seen", "")
                    if last_seen:
                        last_dt = datetime.fromisoformat(last_seen)
                        days_inactive = (now - last_dt).days
                        if days_inactive < filter_params["inactive_days"]:
                            passes = False

                if passes:
                    matching_users.append({
                        "user_id": user_id,
                        "cart_value": float(profile.get("cart_value", 0)),
                        "purchase_count": int(profile.get("purchase_count", 0)),
                        "last_seen": profile.get("last_seen", ""),
                    })

            # cursor=0 means scan is complete
            if cursor == 0:
                break

        # compute segment stats
        cart_values = [u["cart_value"] for u in matching_users if u["cart_value"] > 0]
        avg_cart = sum(cart_values) / len(cart_values) if cart_values else 0.0

        return {
            "user_ids": [u["user_id"] for u in matching_users],
            "size": len(matching_users),
            "avg_cart_value": round(avg_cart, 2),
            "filter_applied": filter_params,
            "status": "ok",
        }

    except Exception as e:
        return {"user_ids": [], "size": 0, "error": str(e), "status": "error"}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 4 — get_segment_stats
# Returns summary statistics for a named segment.
# Used by Campaign Writer to understand who it's writing for.
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_segment_stats(segment_name: str) -> dict:
    """
    Gets stats for a stored segment from Redis.

    Args:
        segment_name: the segment key stored in Redis

    Returns:
        segment stats dict
    """
    try:
        segment_key = f"segment:{segment_name}"
        data = redis_client.get(segment_key)
        if not data:
            return {"segment_name": segment_name, "found": False}

        segment = json.loads(data)
        return {**segment, "found": True}
    except Exception as e:
        return {"segment_name": segment_name, "found": False, "error": str(e)}