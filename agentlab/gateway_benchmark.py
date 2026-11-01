"""Same-prompt execution architecture ablation; no paid calls for prepare/audit."""
import argparse
import asyncio
from collections import Counter
import hashlib
from itertools import permutations
import json
from pathlib import Path
import random
import time
from .expanded_benchmark import build,load,verify,semantic_signature,paired_stats
from .execution_gateway import ExecutionGateway,GATEWAY_VERSION
from .storage import write_json,digest,EpisodeCache
from .prompts import MANUAL

SEED=2026110202
ARMS={'plain':{'execution_gate':False,'repair_limit':0},
      'guard_only':{'execution_gate':True,'repair_limit':0},
      'guard_repair':{'execution_gate':True,'repair_limit':1}}
REQUEST_CAP=1800


def prepare(data):
    data=Path(data)
    if not data.exists():build(data,seed=SEED)
    manifest=json.loads((data/'manifest.json').read_text())
    if manifest['seed']!=SEED:raise ValueError('Wrong holdout seed')
    tasks=load(data,'test')
    previous=Path(__file__).resolve().parent.parent/'data_expanded_v1'
    if previous.exists():
        old=load(previous,'test')
        if {semantic_signature(t) for t in tasks}&{semantic_signature(t) for t in old}:
            raise ValueError('Semantic overlap with prior test set')
    checked=asyncio.run(verify(data))
    return {'api_calls':0,'offline_verification':checked,'experiment_plan':{
        'unique_test_tasks':120,'families':12,'tasks_per_family':10,'arms':ARMS,
        'same_prompt':'MANUAL; fixed verbatim across all three arms','task_executions':360,
        'request_cap':REQUEST_CAP,'estimated_requests':[1400,1700],
        'estimated_reserved_token_cap':12000000,'max_model_rounds':8,'max_tool_attempts':8,
        'max_terminal_repair_rounds':1,'primary':'guard_repair vs plain strict success',
        'secondary':'guard_only ablation, side effects, attempted violations, recovery rate, token/request overhead',
        'note':'Fresh generated values; same synthetic task domain. No online/real-customer or model-training claim. Cap is HTTP attempts, not currency; worst case can exceed cap and pause.'}}


def audit_legacy(directory):
    summaries={}
    for label in ('manual','short'):
        rows=json.loads((Path(directory)/f'{label}_rows.json').read_text())
        counts=Counter()
        for row in rows:
            # Deliberately no task/expected/world is passed to the gateway.
            gate=ExecutionGateway(lambda *a:None)
            for event in row['episode']['events']:
                if event['name']=='update_ticket':
                    decision=gate.check_write(event['arguments'])
                    counts['historical_write_attempts']+=1
                    counts['would_'+decision['decision']]+=1
                    if decision['decision']!='allow':counts[decision['code']]+=1
                # Preserve the factual historical observations; do not pretend to simulate a changed trajectory.
                gate._observe(event['name'],event['arguments'],event['result'])
            verdict=gate.check_final(row['episode']['output'])
            if verdict['code']!='ACCEPT':counts['historical_final_rejections']+=1
        summaries[label]=dict(counts)
    return {'api_calls':0,'kind':'observational_trace_audit','summary':summaries,
            'note':'Retrospective checks on factual observed traces. Not a guarded rerun; no model-repair success or performance gain is measured.'}


def metrics_for(rows):
    from .experiment import summarize
    m=summarize(rows);m['batch_report']=None
    m['requests']=sum(r['episode']['usage']['requests'] for r in rows)
    counts=Counter()
    for row in rows:
        ep=row['episode'];gate=ep.get('execution_gateway')
        if not gate:continue
        blocked=sum(e['decision']['decision']=='block' for e in gate['audit'])
        skipped=sum(e['decision']['decision']=='skip' for e in gate['audit'])
        counts['blocked_tool_attempts']+=blocked
        counts['skipped_write_attempts']+=skipped
        counts['terminal_rejected_proposals']+=sum(v['code']!='ACCEPT' for v in gate['final_checks'])
        counts['terminal_repair_episodes']+=bool(gate['repair_rounds'])
        counts['episodes_with_intervention']+=bool(blocked or skipped or any(v['code']!='ACCEPT' for v in gate['final_checks']))
        counts['successful_intervention_episodes']+=bool(row['grades']['outcome']['score'] and (blocked or skipped or any(v['code']!='ACCEPT' for v in gate['final_checks'])))
    m['gateway_audit']=dict(counts)
    return m



def cost_summary(metrics):
    base,full=metrics['plain'],metrics['guard_repair']
    reported=100*(full['episode_tokens']/base['episode_tokens']-1)
    complete=not(base['usage_missing'] or full['usage_missing'])
    return {'request_overhead_pct':100*(full['requests']/base['requests']-1),
            'reported_tokens_overhead_pct':reported,
            'token_overhead_pct':reported if complete else None,
            'token_usage_complete':complete,
            'cost_note':'HTTP attempts are fully counted. Missing token usage is unknown, not zero; reported-token difference is not complete token or currency cost.'}


def comparison(a,b):
    p=paired_stats({'manual':a,'short':b})
    return {'b_minus_a_delta_pp':p['short_minus_manual_delta_pp'],'stratified_task_bootstrap95_delta_pp':p['stratified_task_bootstrap95_delta_pp'],
            'b_improved_tasks':p['short_improved_tasks'],'b_regressed_tasks':p['short_regressed_tasks'],'exact_mcnemar_p':p['exact_mcnemar_p']}


async def run(data,out,max_requests):
    from .provider import Provider
    from .budget import Budget
    from .experiment import evaluate,identity
    from .locking import run_lock
    if max_requests!=REQUEST_CAP:raise ValueError('Requires explicit --max-api-calls 1800 authorization')
    tasks=load(data,'test')
    if len(tasks)!=120 or Counter(t['family'] for t in tasks)!=Counter({f:10 for f in {t['family'] for t in tasks}}):raise ValueError('Require full balanced test set')
    if json.loads((Path(data)/'manifest.json').read_text())['seed']!=SEED:raise ValueError('Wrong holdout')
    random.Random(6142).shuffle(tasks)
    orders=list(permutations(ARMS))
    out=Path(out)
    with run_lock(out):
        p=out/'budget.json'
        if p.exists() and json.loads(p.read_text())['limits']['requests']!=REQUEST_CAP:raise ValueError('Budget mismatch')
        budget=Budget(p,REQUEST_CAP,12000000)
        provider=Provider(model='deepseek-flash',budget=budget,json_output=True)
        cache=EpisodeCache(out/'episodes.sqlite');rows={label:[] for label in ARMS}
        try:
            if provider.endpoint.rstrip('/')!='https://api.deepseek.com':raise ValueError('Official endpoint required')
            cfg={'kind':'execution_gateway_architecture_ablation','data':json.loads((Path(data)/'manifest.json').read_text()),
                 'prompt':MANUAL,'provider':identity(provider),'arms':ARMS,'gateway_version':GATEWAY_VERSION,
                 'max_steps':8,'max_tool_attempts':8,'concurrency':1,'task_timeout':300,
                 'order':'seed6142 paired task shuffle; six arm-order permutations balanced across 120 tasks',
                 'request_cap':REQUEST_CAP,'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')},
                 'primary':'guard_repair vs plain strict success, paired stratified bootstrap5000 and exact McNemar',
                 'secondary':'guard_only ablation descriptive; audit all blocked/skipped actions, no silent erasure of bad proposals',
                 'note':'Runtime uses observed tool data only; labels used exclusively by external graders. Tool attempts and all paid retries share hard caps.'}
            f=out/'experiment_config.json'
            if f.exists():
                if json.loads(f.read_text())!=cfg:raise ValueError('Frozen configuration changed')
            else:write_json(f,cfg)
            if (out/'result.json').exists():
                print(json.dumps(json.loads((out/'result.json').read_text()),ensure_ascii=False));return
            start=time.perf_counter()
            for i,task in enumerate(tasks):
                for label in orders[i%len(orders)]:
                    batch=await evaluate([task],MANUAL,'live',cache,provider,max_steps=8,concurrency=1,task_timeout=300,**ARMS[label])
                    rows[label]+=batch;write_json(out/f'{label}_rows.json',rows[label])
                    print(json.dumps({'pair':i+1,'arm':label,'success':batch[0]['grades']['outcome']['score'],'requests':budget.state['requests']},ensure_ascii=False),flush=True)
            m={k:metrics_for(v) for k,v in rows.items()}
            result={'kind':'same_prompt_architecture_ablation','config_hash':digest(cfg),'metrics':m,
                    'primary_guard_repair_vs_plain':comparison(rows['plain'],rows['guard_repair']),
                    'secondary_repair_vs_guard_only':comparison(rows['guard_only'],rows['guard_repair']),
                    **cost_summary(m),
                    'budget':json.loads(p.read_text()),'elapsed_this_invocation_seconds':time.perf_counter()-start,
                    'limitation':'120 synthetic tasks; one run per arm. Local synchronous state only; no external exactly-once or arbitrary-policy understanding claim. Secondary comparison exploratory.'}
            write_json(out/'result.json',result);print('FINAL_RESULT '+json.dumps(result,ensure_ascii=False),flush=True)
        finally:
            cache.close();await provider.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','audit','run'])
    p.add_argument('--data',default='data_gateway_v1')
    p.add_argument('--out',default='runs/gateway-ablation-v1')
    p.add_argument('--source',default='runs/expanded-fixed-v1')
    p.add_argument('--max-api-calls',type=int)
    a=p.parse_args()
    if a.command=='prepare':print(json.dumps(prepare(a.data),ensure_ascii=False,indent=2))
    elif a.command=='audit':print(json.dumps(audit_legacy(a.source),ensure_ascii=False,indent=2))
    else:asyncio.run(run(a.data,a.out,a.max_api_calls))


if __name__=='__main__':main()
