from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from config.llm_config import llm_creative, llm_fast
from config.prompts import CAMPAIGN_WRITER_PROMPT, CAMPAIGN_CRITIQUE_PROMPT
from tools.segment_tools import get_segment_stats
from tools.alert_tools import check_alert_cooldown
from state import EcomSenseState, Campaign
from datetime import datetime
from config.llm_config import redis_client
import uuid
import json
import logging

logger = logging.getLogger(__name__)

# Campaign writer uses LCEL chains instead of AgentExecutor
# because it's a two-step sequential process (draft → critique)
# not an open-ended tool-calling loop

# chain
draft_chain = CAMPAIGN_WRITER_PROMPT | llm_creative | JsonOutputParser()

critique_chain = CAMPAIGN_CRITIQUE_PROMPT | llm_fast | JsonOutputParser()

def campaign_writer_node(state: EcomSenseState) -> dict:
    """ 
    LangGraph node for Campaign Writer Agent.
    Generates campaign draft for eligible segments.
    Campaigns go into pending_campaigns - they wait for HITL approval.
    """
    segments = state.get("segments", {})
    insights = state.get("insights", [])
    existing_campaigns = state.get("pending_campaigns", [])

    new_campaigns = []

    for segment_name, segment in segments.items():

        #Don't write a campaign if segment is too small
        if segment.size < 10:
            logger.info(f"Segment '{segment_name}' too small ({segment.size}), skipping")
            continue

        #Don't write if we sent a campaign to this segment recently
        cooldown_key = f"campaign_cooldown:{segment_name}"
        if redis_client.exists(cooldown_key):
            logger.info(f"Campaign cooldown active for '{segment_name}', skipping")
            continue

        # ── Find the most relevant insight for this segment ───────────────
        # Give the campaign writer context about WHY users are in this segment
        relevant_insight = "No specific insight available."
        for insight in insights:
            # simple keyword match — good enough for portfolio scale
            if any(word in insight.summary.lower()
                   for word in segment_name.lower().split("_")):
                relevant_insight = insight.summary
                break

        # ── Step 1: Draft ─────────────────────────────────────────────────
        try:
            draft = draft_chain.invoke({
                "segment_name": segment_name,
                "segment_size": segment.size,
                "avg_cart_value": segment.avg_cart_value,
                "insight": relevant_insight,
                "brand_tone": "friendly, helpful, not pushy",
            })

            logger.info(f"Draft written for segment '{segment_name}'")

        except Exception as e:
            logger.error(f"Draft failed for '{segment_name}': {e}")
            continue

        # ── Step 2: Critique and revise ───────────────────────────────────
        # LLM checks its own draft — catches length violations, tone issues
        try:
            critique = critique_chain.invoke({
                "segment_name": segment_name,
                "draft": json.dumps(draft, indent=2),
            })

            # if critique found issues, use revised copy
            if not critique.get("passes", True):
                logger.info(
                    f"Critique found issues for '{segment_name}': "
                    f"{critique.get('issues', [])}"
                )
                # use revised versions from critique
                push_copy = critique.get("revised_push_copy") or draft.get("push_copy", "")
                email_subject = critique.get("revised_email_subject") or draft.get("email_subject", "")
                sms_copy = critique.get("revised_sms_copy") or draft.get("sms_copy", "")
            else:
                push_copy = draft.get("push_copy", "")
                email_subject = draft.get("email_subject", "")
                sms_copy = draft.get("sms_copy", "")

        except Exception as e:
            logger.warning(f"Critique failed, using raw draft: {e}")
            push_copy = draft.get("push_copy", "")
            email_subject = draft.get("email_subject", "")
            sms_copy = draft.get("sms_copy", "")

        # ── Build Campaign object ──────────────────────────────────────────
        try:
            campaign = Campaign(
                campaign_id=f"camp_{uuid.uuid4().hex[:8]}",
                segment_name=segment_name,
                push_copy=push_copy[:80],       # hard truncate as safety net
                email_subject=email_subject[:50],
                email_body=draft.get("email_body", ""),
                sms_copy=sms_copy[:160],
                status="pending_review",        # waits for human approval
            )

            new_campaigns.append(campaign)
            logger.info(
                f"Campaign created for '{segment_name}': "
                f"'{campaign.push_copy}'"
            )

        except Exception as e:
            logger.error(f"Campaign object creation failed for '{segment_name}': {e}")
            continue

    logger.info(f"Campaign Writer created {len(new_campaigns)} campaigns")

    return {
        # append new campaigns to existing ones
        # existing approved/sent campaigns are preserved
        "pending_campaigns": existing_campaigns + new_campaigns,
        "current_task": "campaigns_drafted",
    }

def approve_campaign(campaign_id: str, state: EcomSenseState) -> dict:
    """
    Called from the Streamlit HITL approval UI.
    Marks a campaign as approved and sets a 24h cooldown.
    """
    campaigns = state.get("pending_campaigns", [])

    updated = []
    for c in campaigns:
        if c.campaign_id == campaign_id:
            c.status = "approved"
            # set 24h cooldown so segment doesn't get messaged again tomorrow
            redis_client.setex(
                f"campaign_cooldown:{c.segment_name}",
                86400,  # 24 hours in seconds
                "sent"
            )
            logger.info(f"Campaign {campaign_id} approved for '{c.segment_name}'")
        updated.append(c)

    return {"pending_campaigns": updated}


def reject_campaign(campaign_id: str, reason: str, state: EcomSenseState) -> dict:
    """
    Called from the Streamlit HITL rejection UI.
    Marks campaign as rejected with a reason for re-drafting.
    """
    campaigns = state.get("pending_campaigns", [])

    updated = []
    for c in campaigns:
        if c.campaign_id == campaign_id:
            c.status = "rejected"
            c.rejection_reason = reason
            logger.info(f"Campaign {campaign_id} rejected: {reason}")
        updated.append(c)

    return {"pending_campaigns": updated}

