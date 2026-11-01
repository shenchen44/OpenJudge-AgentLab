import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
import pytest
import httpx
from openai import AsyncOpenAI
from agentlab.data import build_dataset, load_split, stratified, FAMILIES, oracle
from agentlab.environment import TicketEnvironment
from agentlab.agent import run_episode
from agentlab.prompts import BASELINE, MANUAL
from agentlab.graders import grade_rows
from agentlab.experiment import (
    evaluate,
    dev_feedback,
    optimize,
    finalize,
    pilot,
    pairwise,
)
from agentlab.storage import EpisodeCache, digest, write_json
from agentlab.budget import Budget, BudgetExceeded
from agentlab.provider import Provider, ProviderFailure
from agentlab.calibration import fixture_rows, calibrate
from agentlab.report import render_report


def run(c):
    return asyncio.run(c)


@pytest.fixture
def data(tmp_path):
    path = tmp_path / "data"
    build_dataset(path)
    return path


def test_split_counts_ids_and_factor_isolation(data):
    splits = {s: load_split(data, s) for s in ("pilot", "dev", "validation", "test")}
    assert [len(v) for v in splits.values()] == [12, 24, 12, 24]
    ids = [t["id"] for ts in splits.values() for t in ts]
    assert len(ids) == len(set(ids))
    assert all(not any(f in tid for f in FAMILIES) for tid in ids)
    vips = [t for t in splits["dev"] if t["family"] == "vip"]
    assert all(
        t["world"]["ticket"]["age_minutes"] < t["world"]["policy"]["sla_minutes"]["vip"]
        for t in vips
    )
    assert {t["family"] for t in stratified(splits["dev"], 12)} == set(FAMILIES)
    assert splits["test"][0]["query"] != splits["dev"][0]["query"]
    with pytest.raises(ValueError, match="already exists"):
        build_dataset(data)


def test_hash_integrity(data):
    (data / "dev.json").write_text("[]")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_split(data, "dev")


@pytest.mark.parametrize("family", FAMILIES)
def test_manual_script_contract_all_scenarios(data, family):
    tasks = [
        t
        for split in ("pilot", "dev", "validation", "test")
        for t in load_split(data, split)
        if t["family"] == family
    ]

    async def check():
        rows = []
        for t in tasks:
            rows.append({"task": t, "episode": await run_episode(t, MANUAL, "demo")})
        await grade_rows(rows)
        assert all(r["grades"]["outcome"]["score"] == 1 for r in rows)

    run(check())


@pytest.mark.parametrize(
    "mutation",
    ("no_reads", "empty_message", "wrong_field", "generic_question", "forbidden_write"),
)
def test_clarification_false_positives_fixed(data, mutation):
    task = next(t for t in load_split(data, "pilot") if t["family"] == "missing")
    ep = run(run_episode(task, MANUAL, "demo"))
    if mutation == "no_reads":
        ep["events"] = []
    if mutation == "empty_message":
        ep["output"]["message"] = ""
    if mutation == "wrong_field":
        ep["output"]["missing_fields"] = ["customer_tier"]
    if mutation == "generic_question":
        ep["output"]["message"] = "请继续补充信息"
    if mutation == "forbidden_write":
        ep["events"].append(
            {
                "name": "update_ticket",
                "arguments": {},
                "result": {"error": "INVALID_ARGUMENTS"},
            }
        )
    rows = run(grade_rows([{"task": task, "episode": ep}]))
    assert rows[0]["grades"]["outcome"]["score"] == 0
    assert dev_feedback(rows)["examples"]


def test_wrong_intermediate_write_cannot_be_repaired_for_full_credit(data):
    task = load_split(data, "pilot")[0]
    ep = run(run_episode(task, MANUAL, "demo"))
    bad = deepcopy(ep["events"][-1])
    bad["result"]["ticket"]["priority"] = "P3"
    ep["events"].insert(-1, bad)
    rows = run(grade_rows([{"task": task, "episode": ep}]))
    assert rows[0]["grades"]["outcome"]["score"] == 0


def test_already_done_does_not_write(data):
    t = next(t for t in load_split(data, "pilot") if t["family"] == "already_done")
    ep = run(run_episode(t, MANUAL, "demo"))
    assert all(e["name"] != "update_ticket" for e in ep["events"])
    assert t["expected"]["writes"] == 0


def test_sla_equality_and_policy_precedence(data):
    t = deepcopy(
        next(t for t in load_split(data, "pilot") if t["family"] == "boundary")
    )
    assert oracle(t)["state"]["priority"] == "P2"
    t["world"]["ticket"]["age_minutes"] += 1
    assert oracle(t)["state"]["priority"] == "P1"
    t = next(t for t in load_split(data, "test") if t["family"] == "policy_exception")
    assert t["world"]["customer"]["tier"] == "vip"
    assert oracle(t)["state"]["priority"] == "P2"


@pytest.mark.parametrize(
    "name,args",
    [
        ("bad", {}),
        ("get_ticket", {}),
        ("get_policy", {"extra": "x"}),
        ("get_policy", None),
    ],
)
def test_invalid_tools_do_not_mutate(data, name, args):
    t = load_split(data, "pilot")[0]
    e = TicketEnvironment(t)
    assert "error" in e.call(name, args)
    assert e.world["ticket"] == e.initial


def test_budget_persists_and_prevents_next_request(tmp_path):
    b = Budget(tmp_path / "b.json", 1, 10000)
    b.reserve({"test": "x"}, 20, "actor")
    b.record(12, "actor")
    b = Budget(tmp_path / "b.json", 1, 10000)
    with pytest.raises(BudgetExceeded):
        b.reserve({}, 1, "optimizer")
    assert b.state["requests"] == 1
    assert b.state["reported_tokens"] == 12
    b2 = Budget(tmp_path / "small.json", 10, 1)
    with pytest.raises(BudgetExceeded):
        b2.reserve({"x": "long"}, 10, "actor")
    assert b2.state["requests"] == 0


def completion(message, finish="stop"):
    return {
        "id": "mock",
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek-flash",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", **message},
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def call(name, args, n):
    return {
        "content": None,
        "tool_calls": [
            {
                "id": str(n),
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


async def mock_provider(tmp_path, monkeypatch, handler, max_requests=20):
    monkeypatch.setenv("AGENTLAB_API_KEY", "unit-test-only")
    monkeypatch.setenv("AGENTLAB_BASE_URL", "https://api.deepseek.com")
    p = Provider(
        "deepseek-flash", Budget(tmp_path / "budget.json", max_requests, 1000000)
    )
    await p.client.close()
    p.client = AsyncOpenAI(
        api_key="unit-test-only",
        base_url="https://api.deepseek.com",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return p


def test_actual_sdk_tool_loop_and_usage(data, tmp_path, monkeypatch):
    t = load_split(data, "pilot")[0]
    state = t["expected"]["state"]
    requests = []
    sequence = [
        call("get_ticket", {"ticket_id": t["id"]}, 0),
        call("get_customer", {"customer_id": state["customer_id"]}, 1),
        call("get_policy", {}, 2),
        call(
            "update_ticket",
            {
                "ticket_id": t["id"],
                **{k: state[k] for k in ("priority", "team", "status")},
            },
            3,
        ),
        {
            "content": '{"action":"complete","message":"工单处理完成","missing_fields":[]}'
        },
    ]

    def handler(req):
        body = json.loads(req.content)
        requests.append(body)
        assert body["thinking"] == {"type": "disabled"}
        assert "expected" not in body["messages"][1]["content"]
        return httpx.Response(200, json=completion(sequence[len(requests) - 1]))

    async def check():
        p = await mock_provider(tmp_path, monkeypatch, handler)
        try:
            ep = await run_episode(t, BASELINE, "live", p)
            row = (await grade_rows([{"task": t, "episode": ep}]))[0]
            assert row["grades"]["outcome"]["score"] == 1
            assert ep["usage"]["requests"] == 5
            assert p.budget.state["reported_tokens"] == 75
        finally:
            await p.close()

    run(check())


@pytest.mark.parametrize(
    "content,finish,expected",
    [
        ("not json", "stop", "invalid_final_json"),
        ("[]", "stop", "invalid_final_json"),
        ("", "length", "output_truncated"),
    ],
)
def test_model_failures_are_recorded_not_infrastructure(
    data, tmp_path, monkeypatch, content, finish, expected
):
    def handler(req):
        return httpx.Response(200, json=completion({"content": content}, finish))

    async def check():
        p = await mock_provider(tmp_path, monkeypatch, handler)
        cache = EpisodeCache(tmp_path / "episodes.sqlite")
        try:
            rows = await evaluate(
                load_split(data, "pilot")[:1], BASELINE, "live", cache, p
            )
            assert rows[0]["episode"]["error"] == expected
            assert rows[0]["episode"]["error_kind"] == "model_failure"
            assert cache.tasks()[0]["model_error"] == expected
            assert dev_feedback(rows)["examples"]
            await evaluate(load_split(data, "pilot")[:1], BASELINE, "live", cache, p)
            assert (
                p.budget.state["requests"] == 1
            )  # do not reroll failed models until lucky
        finally:
            cache.close()
            await p.close()

    run(check())


def test_auth_error_aborts_before_second_task(data, tmp_path, monkeypatch):
    requests = []

    def handler(req):
        requests.append(1)
        return httpx.Response(
            401, json={"error": {"message": "never persist this body", "type": "auth"}}
        )

    async def check():
        p = await mock_provider(tmp_path, monkeypatch, handler)
        cache = EpisodeCache(tmp_path / "c.sqlite")
        try:
            with pytest.raises(ProviderFailure) as exc:
                await evaluate(load_split(data, "pilot"), BASELINE, "live", cache, p)
            assert exc.value.status == 401
            assert "never persist" not in str(exc.value)
            assert len(requests) == 1
        finally:
            cache.close()
            await p.close()

    run(check())


def test_retry_cannot_exceed_shared_budget(tmp_path, monkeypatch):
    requests = []

    def handler(req):
        requests.append(1)
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    async def check():
        p = await mock_provider(tmp_path, monkeypatch, handler, max_requests=1)
        try:
            with pytest.raises(BudgetExceeded):
                await p.chat([{"role": "user", "content": "test"}])
            assert len(requests) == 1
        finally:
            await p.close()

    run(check())


def test_search_does_not_read_test_until_explicit_finalize(data, tmp_path, monkeypatch):
    import agentlab.experiment as ex

    loaded = []
    original = ex.load_split

    def spy(root, split):
        loaded.append(split)
        return original(root, split)

    monkeypatch.setattr(ex, "load_split", spy)
    out = tmp_path / "search"
    frozen = run(ex.optimize(data, out, "demo"))
    assert "test" not in loaded
    assert frozen["prompt_sha256"] == digest(frozen["prompt"])
    result = run(ex.finalize(data, out, "demo"))
    assert "test" in loaded
    assert result["test"]["optimized"]["success_rate"] == 1
    assert result["test"]["baseline"]["success_rate"] < 1
    assert run(ex.finalize(data, out, "demo")) == result
    assert run(ex.optimize(data, out, "demo")) == frozen
    with pytest.raises(ValueError, match="another experiment"):
        run(ex.optimize(data, out, "demo", rounds=2))
    assert "不代表大模型调优效果" in render_report(out).read_text()


def test_live_search_requires_ready_pilot(data, tmp_path):
    actor = SimpleNamespace(
        model="test", endpoint="https://test.invalid", request_options={}
    )
    with pytest.raises(ValueError, match="requires --pilot"):
        run(optimize(data, tmp_path / "o", "live", actor))


def test_pilot_gate_and_calibration(data, tmp_path):
    result = run(pilot(data, tmp_path / "pilot", "demo"))
    assert result["decision"] == "ready"
    fixtures = run(fixture_rows(data))
    path = tmp_path / "labels.json"
    write_json(path, fixtures)
    result = run(calibrate(path, tmp_path / "calibration.json"))
    assert result["state_grader"]["accuracy"] == 1
    assert result["state_grader"]["n"] == 24
    assert result["label_sources"] == ["engineered_fixture_not_human"]


def test_paired_analysis_counts_regressions(data, tmp_path):
    cache = EpisodeCache(tmp_path / "c.sqlite")
    try:
        base = run(evaluate(load_split(data, "pilot"), BASELINE, "demo", cache))
        manual = run(evaluate(load_split(data, "pilot"), MANUAL, "demo", cache))
        result = pairwise(base, manual)
        assert result["improved_tasks"] > 0
        assert result["regressed_tasks"] == 0
    finally:
        cache.close()


def test_optimizer_no_entity_memorization():
    class Optimizer:
        async def json(self, messages):
            return {"prompt": "直接读取工单TK-123456789a然后按照示例返回固定的结果。"}

    with pytest.raises(ValueError, match="memorizes"):
        run(
            __import__("agentlab.experiment", fromlist=["propose"]).propose(
                BASELINE, {}, "live", Optimizer()
            )
        )


def test_experiment_process_lock(tmp_path):
    from agentlab.locking import run_lock

    with run_lock(tmp_path / "locked"):
        with pytest.raises(RuntimeError, match="Another process"):
            with run_lock(tmp_path / "locked"):
                pass
    with run_lock(tmp_path / "locked"):
        pass


def test_optimizer_only_receives_development_queries(data, tmp_path, monkeypatch):
    import agentlab.experiment as ex

    calls = []
    original = ex.propose

    async def spy(prompt, feedback, mode, optimizer):
        calls.extend(e["query"] for e in feedback["examples"])
        calls.extend(e["query"] for e in feedback["efficiency_examples"])
        return await original(prompt, feedback, mode, optimizer)

    monkeypatch.setattr(ex, "propose", spy)
    run(ex.optimize(data, tmp_path / "search", "demo"))
    allowed = {t["query"] for t in load_split(data, "dev")}
    assert calls and all(q in allowed for q in calls)


def test_changed_model_cannot_finalize_frozen_run(data, tmp_path):
    out = tmp_path / "frozen"
    run(optimize(data, out, "demo"))
    fake = SimpleNamespace(
        model="other", endpoint="https://other.invalid", request_options={}
    )
    with pytest.raises(ValueError, match="identical model"):
        run(finalize(data, out, "live", fake))


def test_small_budget_refuses_live_search_without_spending(data, tmp_path):
    actor = SimpleNamespace(
        model="test", endpoint="https://test.invalid", request_options={}
    )
    pilot_dir = tmp_path / "pilot"
    pilot_dir.mkdir()
    write_json(
        pilot_dir / "pilot.json",
        {"decision": "ceiling_or_no_headroom", "config": {"mode": "live"}},
    )
    with pytest.raises(ValueError, match="not ready"):
        run(optimize(data, tmp_path / "search", "live", actor, pilot_dir=pilot_dir))


def test_unknown_usage_is_not_claimed_as_cost_improvement():
    from agentlab.experiment import accepted

    baseline = {
        "success_rate": 1,
        "tool_trace_rate": 1,
        "mean_tokens": 100,
        "mean_tool_calls": 4,
        "usage_missing": 0,
        "mode": "live",
    }
    candidate = {**baseline, "mean_tokens": 50, "usage_missing": 1}
    assert not accepted(candidate, baseline)
