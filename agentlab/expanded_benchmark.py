"""Preregistered, stratified synthetic benchmark. No API use for generate/verify/plan."""
import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import time

from .data import FAMILIES, oracle
from .storage import digest, write_json, EpisodeCache
from .environment import TicketEnvironment
from .graders import grade_rows
from .prompts import MANUAL
from .efficiency import SHORT_PROMPT

VERSION = 'tickets-expanded-1'
SEED = 2026110201
QUERIES = (
    '请处理工单 {id}，查询所需资料并落实当前分诊规则。缺少必要信息时先追问。',
    '工单 {id} 等待处理，请依据现行政策完成分级、派单或必要澄清。',
    '帮忙接手 {id}，核验工单及客户资料，完成正确的处理动作。',
    '请检查 {id} 是否还需调整，并按目前的服务规则办理。',
    '需要你为 {id} 完成分流。依据工具查到的规则操作后说明结果。',
    '工单 {id} 请你跟进；按实际情况完成必要更新，资料不足则询问。',
    '请为 {id} 落实优先级与负责团队，遵守当前政策并避免不必要的改动。',
    '核实工单 {id} 的处理条件，执行适用的分诊流程。',
    '请查询并办理 {id}，以现行政策为准，处理后给出结果。',
    '接到工单 {id}，请确认客户与服务规则，再完成合适的处理。',
)
DEV_QUERIES = ('核查 {id} 的信息与处理规则，执行分诊。', '按实时查到的政策处理 {id}，资料不全先澄清。')
POLICY_TEXT = ('类别缺失先追问category；否则查询客户，等级缺失追问customer_tier且不写入。'
               'category_priority优先覆盖其他规则；否则客户等级在priority_tiers内，或等待分钟数严格大于该等级sla_minutes时为P1，其余P2。'
               '从routing读取团队，目标状态in_progress。目标字段一致则不重复写入。TEMPORARY_UNAVAILABLE表示未提交，最多重试一次。')


def reference_answer(task):
    """Separate decision-table implementation, not independent human annotation."""
    w = task['world']; t = w['ticket']; c = w['customer']; p = w['policy']
    absent = []
    evidence = ['get_ticket']
    if t['category'] is None:
        absent = ['category']
    else:
        evidence.append('get_customer')
        if c['tier'] is None:
            absent = ['customer_tier']
    if absent:
        return {'action': 'clarify', 'missing_fields': absent, 'state': deepcopy(t), 'writes': 0, 'required_tools': evidence}
    evidence.append('get_policy')
    decision = 'P2'
    for condition in (c['tier'] in p['priority_tiers'], t['age_minutes'] > p['sla_minutes'][c['tier']]):
        if condition:
            decision = 'P1'
    for category, override in p['category_priority'].items():
        if category == t['category']:
            decision = override
    target = deepcopy(t)
    target.update(priority=decision, team=p['routing'][t['category']], status='in_progress')
    return {'action': 'complete', 'missing_fields': [], 'state': target, 'writes': int(target != t), 'required_tools': evidence}


def semantic_signature(task):
    """Discard opaque entity IDs and wording; retain actual decision-relevant variation."""
    w=task['world']; t=w['ticket']; p=w['policy']
    category=t['category']
    return digest({'category': category, 'tier':w['customer']['tier'], 'age':t['age_minutes'],
                   'sla':p['sla_minutes'], 'priority_tiers':p['priority_tiers'], 'overrides':p['category_priority'],
                   'initial_priority':t['priority'], 'initial_status':t['status'],
                   'initial_team_correct':t['team']==p['routing'].get(category), 'faults':task['faults']})


def build(root, seed=SEED):
    root=Path(root)
    if root.exists():
        raise ValueError('Refuse to overwrite benchmark directory')
    rng=random.Random(seed)
    splits={}; all_signatures=set()
    for split,count in [('dev',2),('test',10)]:
        tasks=[]
        for fi,family in enumerate(FAMILIES):
            for i in range(count):
                tag=f'{VERSION}:{seed}:{split}:{family}:{i}'
                tid='TK-'+digest(tag)[:10]
                category=('billing','technical','outage','account')[(fi+i)%4]
                tier=('standard','vip')[i%2]
                sla={'standard':rng.randint(12,240), 'vip':rng.randint(5,180)}
                policy={'sla_minutes':sla,'priority_tiers':rng.choice([[],['vip'],['standard'],['standard','vip']]),
                        'routing':{k:'TEAM-'+digest(tag+k)[:6] for k in ('billing','technical','outage','account')},
                        'category_priority':{},'target_status':'in_progress','instructions':POLICY_TEXT}
                t={'id':tid,'customer_id':'C-'+digest(tag+'customer')[:10], 'category':category,
                   'age_minutes':max(0,sla[tier]-rng.randint(1,min(10,sla[tier]))),
                   'priority':rng.choice(['P1','P2','P3']), 'team':None, 'status':rng.choice(['new','in_progress'])}
                c={'id':t['customer_id'],'tier':tier}; faults={}
                if family in ('standard','sla','boundary','retry','routing'):
                    c['tier']='standard';policy['priority_tiers']=['vip']
                    t['age_minutes']=max(0,sla['standard']-rng.randint(1,10))
                if family=='vip':
                    c['tier']='vip';policy['priority_tiers']=['vip'];t['age_minutes']=max(0,sla['vip']-1-i)
                elif family=='sla':t['age_minutes']=sla['standard']+rng.randint(1,80)
                elif family=='boundary':t['age_minutes']=sla['standard']
                elif family=='routing':
                    t['team']='TEAM-'+digest(tag+'obsolete')[:6]
                    # Half the routing tasks differ only in team; half require several fields.
                    if i%2==0:t.update(priority='P2',status='in_progress')
                elif family=='retry':faults={'update_ticket':1}
                elif family=='missing':
                    t['category']=None
                    if i%3==0:c['tier']=None
                elif family=='missing_customer':c['tier']=None
                elif family=='transient_read':faults={('get_ticket','get_customer','get_policy')[i%3]:1}
                elif family=='combined':
                    t['age_minutes']=sla[c['tier']]+rng.randint(1,90)
                    faults={('get_ticket','get_customer','get_policy')[i%3]:1,'update_ticket':1}
                    if i%2:policy['category_priority']={category:'P2'}
                elif family=='policy_exception':
                    if i%2:
                        c['tier']='vip';policy['priority_tiers']=['vip'];t['age_minutes']=sla['vip']+i+1
                        policy['category_priority']={category:'P2'}
                    else:
                        c['tier']='standard';policy['priority_tiers']=[];t['age_minutes']=max(0,sla['standard']-1)
                        policy['category_priority']={category:'P1'}
                task={'id':tid,'split':split,'family':family,'template_group':f'{split}:{i}',
                      'query':(DEV_QUERIES if split=='dev' else QUERIES)[i].format(id=tid),
                      'world':{'ticket':t,'customer':c,'policy':policy},'faults':faults}
                if family=='already_done':task['world']['ticket']=reference_answer(task)['state']
                task['expected']=oracle(task)
                assert task['expected']==reference_answer(task)
                signature=semantic_signature(task)
                if signature in all_signatures:raise ValueError('Semantic duplicate')
                all_signatures.add(signature);tasks.append(task)
        splits[split]=tasks
    root.mkdir(parents=True)
    manifest={'version':VERSION,'seed':seed,'synthetic':True,'splits':{},
              'limitation':'Controlled synthetic business rules. No independent human annotation, real customer logs or unseen-domain generalization.'}
    for split,tasks in splits.items():
        write_json(root/f'{split}.json',tasks)
        manifest['splits'][split]={'count':len(tasks),'sha256':digest(tasks)}
    write_json(root/'manifest.json',manifest)
    return manifest


def load(root,split):
    root=Path(root);m=json.loads((root/'manifest.json').read_text())
    tasks=json.loads((root/f'{split}.json').read_text())
    if m['version']!=VERSION or digest(tasks)!=m['splits'][split]['sha256']:
        raise ValueError('Dataset integrity error')
    return tasks


async def verify(root):
    cases=[];signatures=set();counts={}
    for split in ('dev','test'):
        tasks=load(root,split);counts[split]=len(tasks)
        for task in tasks:
            assert task['expected']==reference_answer(task)
            # Metamorphic checks alter the business state rather than opaque IDs.
            for age_delta,priority in [(-1,'P2'),(0,'P2'),(1,'P1')]:
                probe=deepcopy(task)
                w=probe['world'];w['ticket']['category']='billing';w['customer']['tier']='standard'
                w['policy']['priority_tiers']=[];w['policy']['category_priority']={}
                w['ticket']['age_minutes']=w['policy']['sla_minutes']['standard']+age_delta
                assert oracle(probe)==reference_answer(probe)
                assert oracle(probe)['state']['priority']==priority
            if task['expected']['action']=='complete':
                probe=deepcopy(task);probe['world']['ticket']=deepcopy(task['expected']['state'])
                assert oracle(probe)['writes']==reference_answer(probe)['writes']==0
            sig=semantic_signature(task)
            assert sig not in signatures;signatures.add(sig)
            env=TicketEnvironment(task,max_calls=8)
            def call(name,args):
                result=env.call(name,args)
                if result.get('error')=='TEMPORARY_UNAVAILABLE':result=env.call(name,args)
                return result
            t=call('get_ticket',{'ticket_id':task['id']})
            answer=reference_answer(task)
            if t['category'] is not None:
                c=call('get_customer',{'customer_id':t['customer_id']})
                if c['tier'] is not None:
                    call('get_policy',{})
                    if answer['writes']:
                        target=answer['state']
                        call('update_ticket',{'ticket_id':task['id'],**{k:target[k] for k in ('priority','team','status')}})
            ep={'output':{'action':answer['action'],'message':'请补充类别 category 或客户等级 customer_tier' if answer['action']=='clarify' else '工单已实际处理完成','missing_fields':answer['missing_fields']},
                'final_state':env.world['ticket'],'events':env.events,'error':None}
            cases.append({'task':task,'episode':ep,'label':1})
            bad=deepcopy(ep)
            if answer['writes']:
                bad['events']=[e for e in bad['events'] if e['name']!='update_ticket']
                bad['final_state']=deepcopy(task['world']['ticket'])
            elif answer['action']=='clarify':
                bad['output']['missing_fields']=[]
            else:
                # A successful but redundant write must fail even when final state is right.
                bad['events'].append({'name':'update_ticket','result':{'ok':True,'ticket':deepcopy(answer['state'])}})
            cases.append({'task':task,'episode':bad,'label':0})
    graded=await grade_rows(cases)
    mismatches=[r['task']['id'] for r in graded if int(r['grades']['outcome']['score'])!=r['label']]
    if mismatches:raise ValueError(f'Fixture grading mismatches: {len(mismatches)}')
    return {'api_calls':0,'unique_tasks':counts,'semantic_duplicates':0,'oracle_reference_agreements':len(signatures),'sla_metamorphic_checks':len(signatures)*3,
            'engineered_grader_fixtures':len(cases),'fixture_mismatches':0,
            'note':'Software consistency and adversarial fixture checks, not human-annotation reliability or model performance.'}


def plan():
    return {'unique_test_tasks':120,'families':12,'tasks_per_family':10,'variants':['manual','short'],
            'new_task_executions':240,'repeats':1,'request_estimate':{'low':900,'high':1100},
            'request_cap':1200,'estimated_reserved_token_cap':8000000,
            'worst_case_http_attempts':240*8*2,'note':'Estimate is not a guarantee or currency budget. Stop at cap; incomplete runs cannot be reported as a full comparison.'}


def paired_stats(rows):
    groups={label:{r['task']['id']:r for r in values} for label,values in rows.items()}
    if set(groups['manual'])!=set(groups['short']):raise ValueError('Unpaired result')
    if len(groups['manual']) != 120 or any(sum(r['task']['family']==f for r in groups['manual'].values()) != 10 for f in FAMILIES):
        raise ValueError('Require 10 unique tasks per family')
    strata={f:[] for f in FAMILIES}; improved=regressed=0
    for tid,b in groups['manual'].items():
        d=groups['short'][tid]['grades']['outcome']['score']-b['grades']['outcome']['score']
        strata[b['task']['family']].append(d);improved+=d>0;regressed+=d<0
    rng=random.Random(8117)
    samples=sorted(sum(sum(rng.choices(v,k=len(v))) for v in strata.values())/120*100 for _ in range(5000))
    discordant=improved+regressed
    p=min(1.0,2*sum(math.comb(discordant,k) for k in range(min(improved,regressed)+1))/2**discordant) if discordant else 1.0
    return {'short_minus_manual_delta_pp':sum(sum(v) for v in strata.values())/120*100,
            'stratified_task_bootstrap95_delta_pp':[samples[124],samples[4874]],
            'short_improved_tasks':improved,'short_regressed_tasks':regressed,'exact_mcnemar_p':p,
            'note':'Paired synthetic-task comparison, one execution per variant; no production or run-to-run stability claim.'}


async def run(root,out,max_requests):
    from .provider import Provider
    from .budget import Budget
    from .experiment import evaluate,summarize,identity
    from .locking import run_lock
    out=Path(out)
    if max_requests!=1200:raise ValueError('This frozen experiment requires explicit --max-api-calls 1200')
    tasks=load(root,'test');random.Random(5521).shuffle(tasks)
    if len(tasks)!=120:raise ValueError('Require full 120-task test set')
    with run_lock(out):
        existing=out/'budget.json'
        if existing.exists() and json.loads(existing.read_text())['limits']['requests']!=1200:raise ValueError('Budget mismatch')
        budget=Budget(existing,1200,8000000)
        provider=Provider(model='deepseek-flash',budget=budget,json_output=True)
        cache=EpisodeCache(out/'episodes.sqlite');rows={'manual':[],'short':[]}
        try:
            if provider.endpoint.rstrip('/')!='https://api.deepseek.com':raise ValueError('Official endpoint required')
            prompts={'manual':MANUAL,'short':SHORT_PROMPT}
            source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
            config={'benchmark':json.loads((Path(root)/'manifest.json').read_text()),'provider':identity(provider),
                    'prompts':prompts,'source_sha256':source_hashes,'max_steps':8,'concurrency':1,'task_timeout':300,
                    'order':'seed5521 shuffled task pairs, alternating first variant','plan':plan(),
                    'primary':'strict end-to-end task success; per-family macro equals balanced task mean',
                    'secondary':['tokens_all_tasks_including_failures','requests','latency','format_errors','uncommitted_completion','redundant_writes'],
                    'analysis':'stratified paired task bootstrap 5000; exact McNemar; no optional stopping or post-test prompt revision'}
            f=out/'experiment_config.json'
            if f.exists():
                if json.loads(f.read_text())!=config:raise ValueError('Frozen configuration changed')
            else:write_json(f,config)
            resultpath=out/'result.json'
            if resultpath.exists():
                print(json.dumps(json.loads(resultpath.read_text()),ensure_ascii=False));return
            started=time.perf_counter()
            for i,task in enumerate(tasks):
                for label in (['manual','short'] if i%2==0 else ['short','manual']):
                    batch=await evaluate([task],prompts[label],'live',cache,provider,max_steps=8,concurrency=1,task_timeout=300)
                    rows[label]+=batch;write_json(out/f'{label}_rows.json',rows[label])
                    print(json.dumps({'pair':i+1,'variant':label,'success':batch[0]['grades']['outcome']['score'],'requests':budget.state['requests']},ensure_ascii=False),flush=True)
            metrics={k:summarize(v) for k,v in rows.items()}
            for label,m in metrics.items():
                m['batch_report']=None
                m['requests']=sum(r['episode']['usage']['requests'] for r in rows[label])
                m['uncommitted_completion']=sum(r['task']['expected']['writes']==1 and r['episode']['output']['action']=='complete' and not any(e['name']=='update_ticket' and e['result'].get('ok') for e in r['episode']['events']) for r in rows[label])
            result={'kind':'expanded_fixed_comparison','config_hash':digest(config),'metrics':metrics,'paired':paired_stats(rows),
                    'token_reduction_pct':100*(1-metrics['short']['episode_tokens']/metrics['manual']['episode_tokens']),
                    'budget':json.loads(existing.read_text()),'elapsed_this_invocation_seconds':time.perf_counter()-started,
                    'limitation':'120 controlled synthetic tasks, one run per variant, no independent human annotation. Lower tokens are not a quality-preserving saving if success regresses.'}
            write_json(resultpath,result);print('FINAL_RESULT '+json.dumps(result,ensure_ascii=False),flush=True)
        finally:
            cache.close();await provider.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['generate','verify','plan','run'])
    parser.add_argument('--data',default='data_expanded_v1')
    parser.add_argument('--out',default='runs/expanded-fixed-v1')
    parser.add_argument('--max-api-calls',type=int)
    args=parser.parse_args()
    if args.command=='generate':print(json.dumps(build(args.data),ensure_ascii=False,indent=2))
    elif args.command=='verify':print(json.dumps(asyncio.run(verify(args.data)),ensure_ascii=False,indent=2))
    elif args.command=='plan':print(json.dumps(plan(),ensure_ascii=False,indent=2))
    else:asyncio.run(run(args.data,args.out,args.max_api_calls))


if __name__=='__main__':main()
