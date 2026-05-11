lru_cache means Settings() is only created once — singleton pattern
same as creating one DB connection for the whole app

*************************************************
Tools are functions that agents can call by choice. Think of them like API endpoints — the agent decides when to call them and with what arguments. We use LangChain's @tool decorator which automatically generates the schema the LLM needs to know about the tool.

## Step 1 Testing
python -c "
from state import get_initial_state, ErrorCluster, Campaign
from datetime import datetime

# test 1 — initial state creates cleanly
state = get_initial_state('user_1')
print('✓ initial state created for:', state['thread_id'])

# test 2 — ErrorCluster validates correctly
cluster = ErrorCluster(
    cluster_id='c_001',
    error_message='Stripe.js undefined',
    count=847,
    page='checkout',
    severity='critical',
    revenue_impact_inr=45000,
    confidence='high',
    first_seen=datetime.now(),
    browser_breakdown={'Safari': 600, 'Chrome': 247}
)
print('✓ ErrorCluster created:', cluster.severity, cluster.revenue_impact_inr)

# test 3 — hallucination guard catches bad revenue value
try:
    bad = ErrorCluster(
        cluster_id='c_002', error_message='test', count=10,
        page='checkout', severity='low', revenue_impact_inr=-999,
        confidence='low', first_seen=datetime.now(), browser_breakdown={}
    )
except Exception as e:
    print('✓ hallucination guard working:', str(e)[:50])

# test 4 — Campaign validator catches long push copy
try:
    bad_campaign = Campaign(
        campaign_id='camp_001', segment_name='test',
        push_copy='This is a very long push notification that exceeds the 80 character limit easily',
        email_subject='Test', email_body='Test body', sms_copy='Test SMS'
    )
except Exception as e:
    print('✓ push length validator working:', str(e)[:50])

print()
print('All checks passed. state.py is ready.')
"

## Step 2 Testing - ✅
python -c "
# test settings
from config.settings import settings
print('✓ settings loaded, env:', settings.environment)

# test llm config
from config.llm_config import llm, llm_fast, redis_client, error_collection
print('✓ LLMs created')
print('✓ Redis ping:', redis_client.ping())
print('✓ ChromaDB error_collection ready')

# test prompts
from config.prompts import ERROR_DETECTOR_PROMPT, CHAT_PROMPT
msgs = ERROR_DETECTOR_PROMPT.format_messages(input='test error on checkout')
print('✓ ERROR_DETECTOR_PROMPT formatted, message count:', len(msgs))

# test verify_connections
from config.llm_config import verify_connections
status = verify_connections()
for k, v in status.items():
    print(f'  {k}: {v}')

print()
print('Step 2 complete. config/ is ready.')
"
hhh