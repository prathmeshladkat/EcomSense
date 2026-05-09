# Tools used by  Error Detector Agent.
# These are the actual computation functions - the LLM calls these
# when it need real data, not when it needs to reason about what to do.

from langchain_core.tools import tool
from datetime import datetime
from collections import defaultdict
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