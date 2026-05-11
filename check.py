import logging
from datetime import datetime

logging.basicConfig(level=logging.WARNING)

print('Testing graph.py...')

# ── test graph imports ──────────────────────────────────────────

from graph import (
    build_graph,
    orchestrator_node,
    route_from_orchestrator,
    ingest_events_node,
    _build_state_summary,
)

print('✓ all graph functions imported')

# ── test orchestrator logic ─────────────────────────────────────

from state import (
    get_initial_state,
    ErrorCluster,
)

state = get_initial_state('test_thread_001')

# test: no events → idle
state['raw_events'] = []

result = orchestrator_node(state)

print(
    f"✓ orchestrator with no events: "
    f"task={result['current_task']}"
)

# test: error spike → routes to error detector
state['raw_events'] = [
    {
        'type': 'js_error',
        'message': 'Stripe.js undefined',
        'page': 'checkout',
    },
    {
        'type': 'js_error',
        'message': 'Stripe.js undefined',
        'page': 'checkout',
    },
    {
        'type': 'js_error',
        'message': 'Stripe.js undefined',
        'page': 'checkout',
    },
]

result = orchestrator_node(state)

print(
    f"✓ orchestrator with error spike: "
    f"task={result['current_task']}"
)

assert result['current_task'] == 'run_error_detector'

# ── test router ────────────────────────────────────────────────

next_node = route_from_orchestrator({
    'current_task': 'run_error_detector'
})

print(f'✓ router: run_error_detector → {next_node}')

assert next_node == 'error_detector'

next_node = route_from_orchestrator({
    'current_task': 'idle'
})

print(f'✓ router: idle → {next_node}')

# ── test max iterations cap ────────────────────────────────────

state['iteration_count'] = 100

result = orchestrator_node(state)

print(
    f"✓ iteration cap: "
    f"task={result['current_task']}"
)

assert result['current_task'] == 'max_iterations_reached'

# ── test state summary builder ─────────────────────────────────

state = get_initial_state('test_thread_002')

state['error_clusters'] = [
    ErrorCluster(
        cluster_id='c_001',
        error_message='Stripe.js undefined',
        count=847,
        page='checkout',
        severity='critical',
        revenue_impact_inr=45000,
        confidence='high',
        first_seen=datetime.now(),
        browser_breakdown={
            'Safari': 600,
        },
    )
]

summary = _build_state_summary(state)

print(f'✓ state summary built ({len(summary)} chars)')

assert 'Stripe.js' in summary

# ── test graph compilation ─────────────────────────────────────

print()
print('Testing graph compilation with PostgreSQL...')

try:
    app = build_graph()

    print('✓ graph compiled with PostgreSQL checkpointer')

    try:
        print(f'✓ nodes: {list(app.nodes.keys())}')
    except Exception:
        print('✓ graph object created')

except Exception as e:
    print(f'✗ graph compilation failed: {e}')
    print('  Check DATABASE_URL in .env')

print()
print('Step 5 complete.')