"""Recompute published evidence locally. No Provider or network calls."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agentlab.environment import TicketEnvironment
from agentlab.execution_gateway import ExecutionGateway
from agentlab.graders import grade_rows
from agentlab.gateway_benchmark import comparison
from agentlab.storage import digest


async def main():
    directory = ROOT / 'results' / 'gateway-ablation-v1'
    result = json.loads((directory / 'result.json').read_text())
    raw = json.loads((directory / 'result.raw.json').read_text())
    config = json.loads((directory / 'experiment_config.json').read_text())
    frozen = ROOT / 'archive' / 'gateway-ablation-v1-source'
    assert digest(config) == result['config_hash'] == raw['config_hash']
    for name, checksum in config['source_sha256'].items():
        assert hashlib.sha256((frozen / name).read_bytes()).hexdigest() == checksum, name
    for name in ('graders.py', 'execution_gateway.py', 'environment.py'):
        assert (ROOT/'agentlab'/name).read_bytes() == (frozen/name).read_bytes(), 'Use frozen scoring/execution code: '+name
    groups = {}
    for arm in ('plain', 'guard_only', 'guard_repair'):
        rows = json.loads((directory / (arm + '_rows.json')).read_text())
        assert len(rows) == len({row['task']['id'] for row in rows}) == 120
        groups[arm] = rows
        graded = await grade_rows(deepcopy(rows))
        assert all(x['grades']['outcome']['score'] == y['grades']['outcome']['score'] for x, y in zip(rows, graded))
        assert sum(row['grades']['outcome']['score'] for row in rows) == result['metrics'][arm]['successes']
        assert sum(row['episode']['usage']['requests'] for row in rows) == result['metrics'][arm]['requests']
        assert sum(row['episode']['usage']['prompt_tokens']+row['episode']['usage']['completion_tokens'] for row in rows) == result['metrics'][arm]['episode_tokens']
        assert not any(row['episode']['cache_hit'] for row in rows)
        for row in rows:
            episode = row['episode']
            snapshot = episode.get('execution_gateway')
            if not snapshot:
                continue
            public = {key: row['task'][key] for key in ('id', 'query', 'world', 'faults') if key in row['task']}
            env = TicketEnvironment(public, max_calls=8)
            gate = ExecutionGateway(env.call, max_calls=8)
            for event in snapshot['audit']:
                assert gate.call(event['name'], event['arguments']) == event['result']
                assert gate.audit[-1]['decision'] == event['decision']
            assert env.events == episode['events']
            assert env.world['ticket'] == episode['final_state']
            assert gate.tool_attempts == snapshot['tool_attempts'] <= 8
            assert snapshot['repair_rounds'] <= 1
            last = snapshot['final_checks'][-1]
            assert gate.check_final(last['proposed_output'])['code'] == last['code']
    assert len({frozenset(row['task']['id'] for row in rows) for rows in groups.values()}) == 1
    assert comparison(groups['plain'], groups['guard_repair']) == result['primary_guard_repair_vs_plain']
    assert comparison(groups['guard_only'], groups['guard_repair']) == result['secondary_repair_vs_guard_only']
    accounting = json.loads((directory/'request_accounting.json').read_text())
    records = accounting['requests']
    assert len(records) == result['budget']['requests'] == 1466
    assert sum((row['prompt_tokens'] or 0)+(row['completion_tokens'] or 0) for row in records) == result['budget']['reported_tokens']
    assert sum(row['prompt_tokens'] is None or row['completion_tokens'] is None for row in records) == result['budget']['missing_usage'] == 1
    assert result['token_overhead_pct'] is None and not result['token_usage_complete']
    print('PASS: 360 outcomes regraded; 240 gateway traces replayed; paired statistics, frozen source and 1466 request records verified. No API calls.')


if __name__ == '__main__':
    asyncio.run(main())
