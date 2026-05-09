from datetime import datetime

# ── test error tools ───────────────────────────────────────────
from tools.error_tools import (
    cluster_errors,
    score_revenue_impact,
    get_error_baseline,
)

# test clustering
test_events = [
    {
        'type': 'js_error',
        'message': 'Stripe.js undefined',
        'page': 'checkout',
        'browser': 'Safari',
        'ts': datetime.now().isoformat(),
    },
    {
        'type': 'js_error',
        'message': 'Stripe.js undefined',
        'page': 'checkout',
        'browser': 'Safari',
        'ts': datetime.now().isoformat(),
    },
    {
        'type': 'js_error',
        'message': 'Stripe.js undefined',
        'page': 'checkout',
        'browser': 'Chrome',
        'ts': datetime.now().isoformat(),
    },
    {
        'type': 'js_error',
        'message': 'GTM 404',
        'page': 'homepage',
        'browser': 'Chrome',
        'ts': datetime.now().isoformat(),
    },
    {
        'type': 'page_view',
        'page': 'homepage',
    },  # non-error, should be ignored
]

clusters = cluster_errors.invoke({'events': test_events})

print(f'✓ cluster_errors: {len(clusters)} clusters from 5 events (expected 2)')
print(
    f"  cluster 1: {clusters[0]['error_message']} "
    f"count={clusters[0]['count']}"
)

# test revenue scoring with cart value
score = score_revenue_impact.invoke({
    'cluster_id': 'c_001',
    'session_count': 847,
    'avg_cart_inr': 1850.0,
    'page': 'checkout',
})

print(f"✓ score_revenue_impact: ₹{score['revenue_impact_inr_per_hour']}/hr")

# test hallucination guard — no cart value must return -1
score_no_cart = score_revenue_impact.invoke({
    'cluster_id': 'c_002',
    'session_count': 200,
    'avg_cart_inr': None,
    'page': 'checkout',
})

assert (
    score_no_cart['revenue_impact_inr_per_hour'] == -1
), 'Hallucination guard failed!'

print('✓ hallucination guard: revenue=-1 when no cart value')

# ── test segment tools ─────────────────────────────────────────
from tools.segment_tools import update_profile, get_profile

result = update_profile.invoke({
    'user_id': 'test_user_001',
    'event': {
        'type': 'cart_added',
        'cart_value': 2499,
        'page': 'product',
    },
})

print(f"✓ update_profile: {result['status']}")

profile = get_profile.invoke({'user_id': 'test_user_001'})

print(
    f"✓ get_profile: found={profile['found']} "
    f"cart={profile.get('cart_value')}"
)

# ── test rag tools ─────────────────────────────────────────────
from tools.rag_tools import (
    store_error_in_vector_db,
    retrieve_similar_errors,
)

store_result = store_error_in_vector_db.invoke({
    'cluster': {
        'cluster_id': 'c_test_001',
        'error_message': 'Stripe.js undefined',
        'page': 'checkout',
        'count': 847,
        'severity': 'critical',
        'revenue_impact_inr': 45000,
        'browser_breakdown': {
            'Safari': 600,
            'Chrome': 247,
        },
        'first_seen': datetime.now().isoformat(),
    }
})

print(f"✓ store_error_in_vector_db: stored={store_result['stored']}")

retrieve_result = retrieve_similar_errors.invoke({
    'query': 'payment script failing on checkout',
    'n_results': 3,
})

print(
    f"✓ retrieve_similar_errors: "
    f"found {retrieve_result['count']} results"
)

# ── test alert tools ───────────────────────────────────────────
from tools.alert_tools import (
    check_alert_cooldown,
    set_alert_cooldown,
)

cooldown = check_alert_cooldown.invoke({
    'cluster_id': 'c_new_test'
})

print(
    f"✓ check_alert_cooldown: "
    f"active={cooldown['cooldown_active']} "
    f"(should be False)"
)

set_result = set_alert_cooldown.invoke({
    'cluster_id': 'c_new_test'
})

print(f"✓ set_alert_cooldown: set={set_result['cooldown_set']}")

cooldown_after = check_alert_cooldown.invoke({
    'cluster_id': 'c_new_test'
})

print(
    f"✓ cooldown now active: "
    f"{cooldown_after['cooldown_active']} "
    f"(should be True)"
)

print()
print('All tools working. Step 3 complete.')