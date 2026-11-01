"""Evidence-only business execution boundary; has no access to task labels or world.

This is an application rule engine around tools, not an LLM judge. It prevents
unsafe commits locally; external concurrency still needs server-side CAS/idempotency.
"""
from copy import deepcopy
from .environment import TOOLS

GATEWAY_VERSION = 'evidence-gateway-1'


class ExecutionGateway:
    def __init__(self, call_tool, max_calls=8, state=None):
        self._call_tool=call_tool
        self.max_calls=max_calls
        self.ticket=None
        self.customer=None
        self.policy=None
        self.audit=[]
        self.final_checks=[]
        self.tool_attempts=0
        self.repair_rounds=0
        if state:
            if state['version']!=GATEWAY_VERSION:raise ValueError('Gateway checkpoint version mismatch')
            for name in ('ticket','customer','policy','audit','final_checks','tool_attempts','repair_rounds'):
                setattr(self,name,deepcopy(state[name]))

    def snapshot(self):
        return {'version':GATEWAY_VERSION,**{k:deepcopy(getattr(self,k)) for k in
                ('ticket','customer','policy','audit','final_checks','tool_attempts','repair_rounds')}}

    def _observe(self,name,args,result):
        if 'error' in result:return
        if name=='get_ticket':
            if result.get('id')!=args.get('ticket_id'):return
            if self.ticket and self.ticket.get('customer_id')!=result.get('customer_id'):self.customer=None
            self.ticket=deepcopy(result)
        elif name=='get_customer':
            if result.get('id')==args.get('customer_id'):self.customer=deepcopy(result)
        elif name=='get_policy':self.policy=deepcopy(result)
        elif name=='update_ticket' and result.get('ok'):
            self.ticket=deepcopy(result['ticket'])

    def requirements(self):
        if self.ticket is None:return {'code':'MISSING_TICKET_EVIDENCE','required_tools':['get_ticket']}
        if self.ticket.get('category') is None:return {'code':'CLARIFICATION_REQUIRED','missing_fields':['category']}
        if self.customer is None or self.customer.get('id')!=self.ticket.get('customer_id'):
            return {'code':'MISSING_CUSTOMER_EVIDENCE','required_tools':['get_customer']}
        if self.customer.get('tier') is None:return {'code':'CLARIFICATION_REQUIRED','missing_fields':['customer_tier']}
        if self.policy is None:return {'code':'MISSING_POLICY_EVIDENCE','required_tools':['get_policy']}
        try:
            t,c,p=self.ticket,self.customer,self.policy
            priority=p['category_priority'].get(t['category'])
            if priority is None:
                priority='P1' if c['tier'] in p['priority_tiers'] or t['age_minutes']>p['sla_minutes'][c['tier']] else 'P2'
            target={'priority':priority,'team':p['routing'][t['category']],'status':p['target_status']}
            if priority not in ('P1','P2','P3') or target['status']!='in_progress' or not target['team']:
                raise ValueError('Unsupported business contract')
            return {'code':'READY','target':target}
        except (KeyError,TypeError,ValueError):
            return {'code':'UNSUPPORTED_POLICY','instruction':'当前政策字段不完整或不受支持，不能提交或宣告完成。'}

    def check_write(self,args):
        status=self.requirements()
        if status['code']!='READY':return {'decision':'block',**status}
        if args['ticket_id']!=self.ticket['id']:
            return {'decision':'block','code':'TICKET_ID_MISMATCH'}
        mismatches={k:{'proposed':args[k],'required_by_observed_policy':v}
                    for k,v in status['target'].items() if args[k]!=v}
        if mismatches:return {'decision':'block','code':'POLICY_CONSTRAINT_VIOLATION','mismatches':mismatches}
        if all(self.ticket.get(k)==v for k,v in status['target'].items()):
            return {'decision':'skip','code':'NO_CHANGE'}
        return {'decision':'allow','code':'VALIDATED_WRITE'}

    def call(self,name,args):
        if self.tool_attempts>=self.max_calls:raise RuntimeError('tool_budget_exhausted')
        self.tool_attempts+=1
        spec=next((s for s in TOOLS if s['name']==name),None)
        valid=spec and isinstance(args,dict) and set(args)==set(spec['parameters']['required']) and all(isinstance(v,str) for v in args.values())
        if not valid:
            decision={'decision':'block','code':'INVALID_TOOL_OR_ARGUMENTS'}
        elif name=='update_ticket':decision=self.check_write(args)
        else:decision={'decision':'allow','code':'READ_TOOL'}
        if decision['decision']=='block':
            result={'error':'EXECUTION_GATEWAY_REJECTED','committed':False,'feedback':decision}
        elif decision['decision']=='skip':
            result={'ok':True,'skipped':True,'committed':False,'reason':'NO_CHANGE','ticket':deepcopy(self.ticket)}
        else:
            result=self._call_tool(name,args)
            self._observe(name,args,result)
        self.audit.append({'name':name,'arguments':deepcopy(args),'decision':decision,'result':deepcopy(result)})
        return result

    def check_final(self,output):
        schema=(isinstance(output,dict) and output.get('action') in ('complete','clarify','failed')
                and isinstance(output.get('message'),str) and len(output['message'].strip())>=4
                and isinstance(output.get('missing_fields'),list)
                and all(isinstance(v,str) for v in output['missing_fields']))
        if not schema:status={'code':'FINAL_SCHEMA_INVALID'}
        else:
            required=self.requirements()
            if required['code']=='CLARIFICATION_REQUIRED':
                fields=required['missing_fields']
                terms={'category':('category','类别','分类'),'customer_tier':('tier','等级','级别')}
                valid=output['action']=='clarify' and output['missing_fields']==fields and all(any(term in output['message'].lower() for term in terms[f]) for f in fields)
                status={'code':'ACCEPT'} if valid else required
            elif required['code']!='READY':status=required
            elif any(self.ticket.get(k)!=v for k,v in required['target'].items()):
                status={'code':'UNCOMMITTED_RESULT','instruction':'根据已查询政策仍需实际调用 update_ticket，成功提交后才能宣布完成。'}
            elif output['action']!='complete' or output['missing_fields']:
                status={'code':'FINAL_ACTION_MISMATCH','instruction':'已观察到目标状态，请按 complete 合同报告结果。'}
            else:status={'code':'ACCEPT'}
        self.final_checks.append({**deepcopy(status),'proposed_output':deepcopy(output)})
        return status
