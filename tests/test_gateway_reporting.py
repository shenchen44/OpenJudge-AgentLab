from agentlab.gateway_benchmark import cost_summary
import pytest


def test_unknown_usage_never_reported_as_complete_cost_improvement():
    metrics={'plain':{'episode_tokens':100,'requests':10,'usage_missing':0},
             'guard_repair':{'episode_tokens':90,'requests':11,'usage_missing':1}}
    result=cost_summary(metrics)
    assert result['token_overhead_pct'] is None and not result['token_usage_complete']
    assert result['reported_tokens_overhead_pct']==pytest.approx(-10)
    assert result['request_overhead_pct']==pytest.approx(10)


def test_complete_token_usage_can_be_compared():
    metrics={'plain':{'episode_tokens':100,'requests':10,'usage_missing':0},
             'guard_repair':{'episode_tokens':110,'requests':11,'usage_missing':0}}
    result=cost_summary(metrics)
    assert result['token_usage_complete'] and result['token_overhead_pct']==pytest.approx(10)
