import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
import pytest
from openai.types.chat import ChatCompletionMessage
from agentlab.data import build_dataset,load_split
from agentlab.environment import TicketEnvironment
from agentlab.execution_gateway import ExecutionGateway
from agentlab.agent import run_episode
from agentlab.experiment import evaluate
from agentlab.graders import grade_rows
from agentlab.provider import ProviderFailure
from agentlab.prompts import MANUAL
from agentlab.storage import EpisodeCache


@pytest.fixture
def task(tmp_path):
    build_dataset(tmp_path/'data')
    return next(t for t in load_split(tmp_path/'data','pilot') if t['family']=='standard')


def prime(gate,task):
    gate.call('get_ticket',{'ticket_id':task['id']})
    gate.call('get_customer',{'customer_id':task['world']['customer']['id']})
    gate.call('get_policy',{})


def target_args(task):
    return {'ticket_id':task['id'],**{k:task['expected']['state'][k] for k in ['priority','team','status']}}


def calls(*pairs):
    return ChatCompletionMessage(role='assistant',content=None,tool_calls=[
        {'id':f'c{i}','type':'function','function':{'name':name,'arguments':json.dumps(args)}} for i,(name,args) in enumerate(pairs)])


def final(action='complete',fields=None):
    return ChatCompletionMessage(role='assistant',content=json.dumps({'action':action,'message':'工单已处理完成' if action=='complete' else '请补充客户等级 customer_tier','missing_fields':fields or []},ensure_ascii=False))


class Fake:
    model='mock';endpoint='mock';request_options={}
    def __init__(self,items):self.items=iter(items);self.messages=[]
    async def chat(self,messages,**kwargs):
        self.messages.append(deepcopy(messages))
        item=next(self.items)
        if isinstance(item,Exception):raise item
        return item,'tool_calls' if item.tool_calls else 'stop'


def reads(task):
    return [calls(('get_ticket',{'ticket_id':task['id']})),calls(('get_customer',{'customer_id':task['world']['customer']['id']}),('get_policy',{}))]


def test_write_without_evidence_is_blocked_before_backend(task):
    env=TicketEnvironment(task);g=ExecutionGateway(env.call)
    result=g.call('update_ticket',target_args(task))
    assert result['error']=='EXECUTION_GATEWAY_REJECTED'
    assert not env.events and env.world['ticket']==task['world']['ticket']
    assert g.tool_attempts==1 and g.audit[0]['decision']['decision']=='block'


def test_policy_mismatch_is_rejected_and_valid_commit_has_receipt(task):
    env=TicketEnvironment(task);g=ExecutionGateway(env.call)
    prime(g,task);wrong=target_args(task);wrong['team']='wrong-team'
    assert g.call('update_ticket',wrong)['feedback']['code']=='POLICY_CONSTRAINT_VIOLATION'
    assert len(env.events)==3
    assert g.call('update_ticket',target_args(task))['ok']
    assert g.check_final(json.loads(final().content))['code']=='ACCEPT'
    assert env.world['ticket']==task['expected']['state']


def test_repeated_or_already_satisfied_writes_never_reach_backend(task):
    task=deepcopy(task);task['world']['ticket']=deepcopy(task['expected']['state'])
    env=TicketEnvironment(task);g=ExecutionGateway(env.call)
    prime(g,task)
    for _ in range(2):assert g.call('update_ticket',target_args(task))['skipped']
    assert not any(e['name']=='update_ticket' for e in env.events)
    assert g.check_final(json.loads(final().content))['code']=='ACCEPT'


def test_missing_information_requires_clarification_and_forbids_write(task):
    task=deepcopy(task);task['world']['customer']['tier']=None
    env=TicketEnvironment(task);g=ExecutionGateway(env.call);prime(g,task)
    assert g.call('update_ticket',target_args(task))['feedback']['missing_fields']==['customer_tier']
    assert g.check_final(json.loads(final().content))['code']=='CLARIFICATION_REQUIRED'
    assert g.check_final(json.loads(final('clarify',['customer_tier']).content))['code']=='ACCEPT'


def test_transient_write_failure_is_not_treated_as_commit(task):
    task=deepcopy(task);task['faults']={'update_ticket':1}
    env=TicketEnvironment(task);g=ExecutionGateway(env.call);prime(g,task)
    assert g.call('update_ticket',target_args(task))['error']=='TEMPORARY_UNAVAILABLE'
    assert g.check_final(json.loads(final().content))['code']=='UNCOMMITTED_RESULT'
    assert g.call('update_ticket',target_args(task))['ok']
    assert g.check_final(json.loads(final().content))['code']=='ACCEPT'


def test_gateway_attempt_budget_includes_blocked_and_skipped_calls(task):
    env=TicketEnvironment(task);g=ExecutionGateway(env.call,max_calls=2)
    for _ in range(2):g.call('update_ticket',target_args(task))
    with pytest.raises(RuntimeError,match='tool_budget'):g.call('get_policy',{})
    assert not env.events


def test_evidence_snapshot_restores_without_hidden_labels(task):
    env=TicketEnvironment(task);g=ExecutionGateway(env.call);prime(g,task)
    snapshot=g.snapshot();assert 'expected' not in snapshot and 'world' not in snapshot
    restored=ExecutionGateway(env.call,state=snapshot)
    assert restored.call('update_ticket',target_args(task))['ok']
    assert restored.check_final(json.loads(final().content))['code']=='ACCEPT'


def test_gateway_version_changes_cache_identity(task,tmp_path):
    async def check():
        cache=EpisodeCache(tmp_path/'c.sqlite')
        try:
            unguarded=Fake(reads(task)+[final()])
            a=await evaluate([task],MANUAL,'live',cache,unguarded)
            guarded=Fake(reads(task)+[final(),calls(('update_ticket',target_args(task))),final()])
            b=await evaluate([task],MANUAL,'live',cache,guarded,execution_gate=True)
            assert a[0]['task_key']!=b[0]['task_key']
            assert a[0]['grades']['outcome']['score']==0 and b[0]['grades']['outcome']['score']==1
            gate=b[0]['episode']['execution_gateway']
            assert gate['repair_rounds']==1
            assert len(guarded.messages)==5 and any('execution_feedback' in (m.get('content') or '') for m in guarded.messages[-1])
        finally:cache.close()
    asyncio.run(check())


def test_terminal_repair_is_bounded_and_fails_closed(task):
    fake=Fake(reads(task)+[final(),final()])
    ep=asyncio.run(run_episode(task,MANUAL,'live',fake,execution_gate=True))
    assert ep['error']=='execution_contract_rejected' and ep['output']['action']=='failed'
    assert ep['execution_gateway']['repair_rounds']==1
    assert len(fake.messages)==4


def test_rejected_write_then_correction_is_audited_and_graded(task):
    wrong=target_args(task);wrong['priority']='WRONG'
    fake=Fake(reads(task)+[calls(('update_ticket',wrong)),calls(('update_ticket',target_args(task))),final()])
    ep=asyncio.run(run_episode(task,MANUAL,'live',fake,execution_gate=True))
    rows=asyncio.run(grade_rows([{'task':task,'episode':ep}]))
    assert rows[0]['grades']['outcome']['score']==1
    assert sum(e['name']=='update_ticket' for e in ep['events'])==1
    assert sum(e['decision']['decision']=='block' for e in ep['execution_gateway']['audit'])==1


def test_runtime_does_not_receive_evaluation_labels(task):
    public={k:v for k,v in task.items() if k in ('id','query','world','faults')}
    fake=Fake(reads(task)+[calls(('update_ticket',target_args(task))),final()])
    ep=asyncio.run(run_episode(public,MANUAL,'live',fake,execution_gate=True))
    assert ep['error'] is None and ep['final_state']==task['expected']['state']
    assert all('expected' not in json.dumps(m) for m in fake.messages)


def test_checkpoint_after_repair_feedback_preserves_budget_and_evidence(task,tmp_path):
    async def check():
        cache=EpisodeCache(tmp_path/'c.sqlite')
        try:
            a=Fake(reads(task)+[final(),ProviderFailure('http_error',401)])
            with pytest.raises(ProviderFailure):
                await evaluate([task],MANUAL,'live',cache,a,execution_gate=True)
            entry=cache.tasks()[0];snap=cache.checkpoint(entry['key'])
            assert snap['gateway']['repair_rounds']==1 and snap['steps_done']==3
            b=Fake([calls(('update_ticket',target_args(task))),final()])
            result=await evaluate([task],MANUAL,'live',cache,b,execution_gate=True)
            ep=result[0]['episode']
            assert ep['resumed_from_step']==3 and ep['execution_gateway']['repair_rounds']==1
            assert result[0]['grades']['outcome']['score']==1
            assert sum(e['name']=='get_ticket' for e in ep['events'])==1
        finally:cache.close()
    asyncio.run(check())


def test_guard_only_ablation_never_spends_a_terminal_repair(task):
    fake=Fake(reads(task)+[final()])
    ep=asyncio.run(run_episode(task,MANUAL,'live',fake,execution_gate=True,repair_limit=0))
    assert ep['error']=='execution_contract_rejected'
    assert len(fake.messages)==3 and ep['execution_gateway']['repair_rounds']==0


def test_guard_only_and_repair_have_distinct_cache_and_feedback_paths(task,tmp_path):
    async def check():
        cache=EpisodeCache(tmp_path/'ablation.sqlite')
        try:
            only=await evaluate([task],MANUAL,'live',cache,Fake(reads(task)+[final()]),execution_gate=True,repair_limit=0)
            full=await evaluate([task],MANUAL,'live',cache,Fake(reads(task)+[final(),calls(('update_ticket',target_args(task))),final()]),execution_gate=True,repair_limit=1)
            assert only[0]['task_key']!=full[0]['task_key']
            assert only[0]['grades']['outcome']['score']==0 and full[0]['grades']['outcome']['score']==1
            rejected=full[0]['episode']['execution_gateway']['final_checks'][0]
            assert rejected['code']=='UNCOMMITTED_RESULT' and rejected['proposed_output']['action']=='complete'
        finally:cache.close()
    asyncio.run(check())


def test_unsupported_policy_fails_closed_without_guessing(task):
    task=deepcopy(task);task['world']['policy'].pop('target_status')
    env=TicketEnvironment(task);gate=ExecutionGateway(env.call);prime(gate,task)
    assert gate.call('update_ticket',target_args(task))['feedback']['code']=='UNSUPPORTED_POLICY'
    assert len(env.events)==3 and env.world['ticket']==task['world']['ticket']


def test_reference_execution_on_all_new_development_cases_without_labels():
    from agentlab.expanded_benchmark import load
    root=__import__('pathlib').Path(__file__).resolve().parent.parent
    for task in load(root/'data_gateway_v1','dev'):
        public={k:v for k,v in task.items() if k in ('id','query','world','faults')}
        env=TicketEnvironment(public,max_calls=8);gate=ExecutionGateway(env.call,max_calls=8)
        def call(name,args):
            result=gate.call(name,args)
            if result.get('error')=='TEMPORARY_UNAVAILABLE':result=gate.call(name,args)
            return result
        t=call('get_ticket',{'ticket_id':task['id']})
        if t['category'] is not None:
            c=call('get_customer',{'customer_id':t['customer_id']})
            if c['tier'] is not None:call('get_policy',{})
        status=gate.requirements()
        if status['code']=='READY':
            call('update_ticket',{'ticket_id':task['id'],**status['target']})
            output={'action':'complete','message':'工单处理完成','missing_fields':[]}
        else:
            assert status['code']=='CLARIFICATION_REQUIRED'
            output={'action':'clarify','message':'请补充类别 category 或客户等级 customer_tier','missing_fields':status['missing_fields']}
        assert gate.check_final(output)['code']=='ACCEPT'
        assert env.world['ticket']==task['expected']['state']
        assert sum(e['name']=='update_ticket' and e['result'].get('ok',False) for e in env.events)==task['expected']['writes']
        assert gate.tool_attempts<=8
