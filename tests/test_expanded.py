import asyncio
from collections import Counter
from copy import deepcopy
import pytest
from agentlab.expanded_benchmark import build,load,verify,semantic_signature,paired_stats
from agentlab.storage import write_json


def test_diverse_reproducible_disjoint_benchmark_and_reference_fixtures(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b'
    assert build(a)==build(b)
    dev,test=load(a,'dev'),load(a,'test')
    assert set(Counter(t['family'] for t in test).values())=={10}
    assert len(test)==120 and len(dev)==24
    assert len({semantic_signature(t) for t in dev+test})==144
    assert not {t['template_group'] for t in dev}&{t['template_group'] for t in test}
    assert len({tuple(sorted(t['world']['policy']['sla_minutes'].items())) for t in test})>100
    result=asyncio.run(verify(a))
    assert result['fixture_mismatches']==0
    assert result['engineered_grader_fixtures']==288
    assert result['sla_metamorphic_checks']==432
    with pytest.raises(ValueError,match='overwrite'):build(a)


def test_dataset_tampering_detected(tmp_path):
    build(tmp_path/'data')
    tasks=load(tmp_path/'data','test');tasks[0]['world']['ticket']['age_minutes']+=1
    write_json(tmp_path/'data'/'test.json',tasks)
    with pytest.raises(ValueError,match='integrity'):load(tmp_path/'data','test')


def test_paired_summary_uses_unique_tasks_and_preserves_discordance(tmp_path):
    build(tmp_path/'data')
    tasks=load(tmp_path/'data','test')
    rows={label:[{'task':t,'grades':{'outcome':{'score':1}}} for t in tasks] for label in ['manual','short']}
    rows['short']=deepcopy(rows['short'])
    rows['short'][0]['grades']['outcome']['score']=0
    result=paired_stats(rows)
    assert result['short_regressed_tasks']==1 and result['short_improved_tasks']==0
    assert result['short_minus_manual_delta_pp']==pytest.approx(-100/120)
    assert result['exact_mcnemar_p']==1.0
    assert paired_stats(rows)==result
    rows['short'].pop()
    with pytest.raises(ValueError,match='Unpaired'):paired_stats(rows)
