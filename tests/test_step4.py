import os, uuid
from dotenv import load_dotenv
from datetime import datetime
from macro_lens import build_graph, MacroState

load_dotenv()

def make_state(observation_date=None):
    return {
        'current_date': observation_date or datetime.now().strftime('%Y-%m-%d'),
        'observation_date': observation_date,
        'macro_data': None,
        'fetch_attempts': 0,
        'data_requests': None,
        'growth_direction': None,
        'inflation_direction': None,
        'regime': None,
        'regime_confidence': None,
        'regime_rationale': None,
        'previous_regime': None,
        'tilts': None,
        'weights': None,
        'allocation_rationale': None,
        'report': None,
        'messages': [],
    }

graph = build_graph()

# Test 1: live mode
print('Running live mode...')
result = graph.invoke(make_state(), {'configurable': {'thread_id': str(uuid.uuid4())}})
assert result['macro_data'] is not None
assert result['regime'] is not None
assert result['weights'] is not None
print('PASS live mode — regime:', result['regime'])

# Test 2: backtest mode
print('Running backtest mode (2020-06-30)...')
result_pit = graph.invoke(make_state('2020-06-30'), {'configurable': {'thread_id': str(uuid.uuid4())}})
assert result_pit['macro_data'] is not None
vix_date = result_pit['macro_data']['vix']['date']
assert vix_date <= '2020-06-30', f'FAIL: VIX date leaked — {vix_date}'
print('PASS backtest mode — regime:', result_pit['regime'], '| VIX date:', vix_date)