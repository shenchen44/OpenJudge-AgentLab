import asyncio
import json
from types import SimpleNamespace
import pytest
from agentlab.data import build_dataset
from agentlab.agent import run_episode
from agentlab.prompts import MANUAL
from agentlab.graders import grade_rows
from agentlab.experiment import pilot
from agentlab.efficiency import prepare, efficiency_check, SHORT_PROMPT
from agentlab.budget import Budget
from agentlab.storage import write_json


def setup_fixture(tmp_path):
    data = tmp_path / "data"
    build_dataset(data)
    out = tmp_path / "pilot"
    p = asyncio.run(pilot(data, out, "demo"))
    p["config"]["mode"] = "live"
    p["config"]["provider"] = {
        "model": "deepseek-flash",
        "endpoint": "https://api.deepseek.com",
        "request_options": {"extra_body": {"thinking": {"type": "disabled"}}},
    }
    for label, tokens in [("baseline", 100), ("manual", 200)]:
        f = out / f"pilot_{label}.json"
        rows = json.loads(f.read_text())
        for r in rows:
            r["episode"]["mode"] = "live"
            r["episode"]["usage"] = {
                "prompt_tokens": tokens,
                "completion_tokens": 0,
                "requests": 1,
                "usage_missing": 0,
            }
        write_json(f, rows)
    write_json(out / "pilot.json", p)
    b = Budget(out / "budget.json", 150, 1000000)
    b.state["requests"] = 98
    b.state["reserved_tokens"] = 450265
    b.save()
    return data, out, p, b


def test_preflight_reuses_controls_and_remaining_budget(tmp_path):
    data, out, p, b = setup_fixture(tmp_path)
    before = (out / "budget.json").read_bytes()
    _, controls, _, plan = prepare(data, out, out)
    assert plan["requests_remaining"] == 52 and plan["new_episodes"] == 12
    assert plan["reused_control_episodes"] == 24
    assert len(SHORT_PROMPT) < len(MANUAL)
    assert (out / "budget.json").read_bytes() == before
    assert len(controls["manual"]) == 12


def test_no_new_budget_or_new_directory(tmp_path):
    data, out, p, b = setup_fixture(tmp_path)
    with pytest.raises(ValueError, match="cannot increase"):
        prepare(data, out, out, max_requests=151)
    with pytest.raises(ValueError, match="reuse the original"):
        prepare(data, tmp_path / "other", out)


def test_changed_data_or_incomplete_controls_rejected(tmp_path):
    data, out, p, b = setup_fixture(tmp_path)
    f = out / "pilot_manual.json"
    rows = json.loads(f.read_text())
    write_json(f, rows[:-1])
    with pytest.raises(ValueError, match="Incomplete"):
        prepare(data, out, out)


def test_only_candidate_evaluated_once_then_result_reused(tmp_path, monkeypatch):
    data, out, p, b = setup_fixture(tmp_path)
    import agentlab.efficiency as module

    calls = []

    async def evaluate_stub(tasks, prompt, mode, cache, provider, **kwargs):
        calls.append((len(tasks), prompt))
        rows = []
        for t in tasks:
            ep = await run_episode(t, MANUAL, "demo")
            ep["mode"] = "live"
            ep["usage"] = {
                "prompt_tokens": 80,
                "completion_tokens": 0,
                "requests": 1,
                "usage_missing": 0,
            }
            rows.append({"task": t, "episode": ep})
        return await grade_rows(rows)

    monkeypatch.setattr(module, "evaluate", evaluate_stub)
    provider = SimpleNamespace(**p["config"]["provider"], budget=b)
    result = asyncio.run(efficiency_check(data, out, out, provider))
    assert calls == [(12, SHORT_PROMPT)]
    assert result["decision"] == "promising_efficiency_needs_independent_validation"
    assert result["comparison"]["baseline"]["token_reduction_pct"] == pytest.approx(20)
    again = asyncio.run(efficiency_check(data, out, out, provider))
    assert again == result and len(calls) == 1
    assert json.loads((out / "pilot.json").read_text()) == p


def test_correctness_regression_is_not_accepted_for_savings(tmp_path, monkeypatch):
    data, out, p, b = setup_fixture(tmp_path)
    import agentlab.efficiency as module

    async def bad_candidate(tasks, *args, **kwargs):
        rows = []
        for t in tasks:
            ep = await run_episode(t, MANUAL, "demo")
            ep["mode"] = "live"
            ep["output"]["action"] = "failed"
            ep["usage"] = {
                "prompt_tokens": 1,
                "completion_tokens": 0,
                "requests": 1,
                "usage_missing": 0,
            }
            rows.append({"task": t, "episode": ep})
        return await grade_rows(rows)

    monkeypatch.setattr(module, "evaluate", bad_candidate)
    result = asyncio.run(
        efficiency_check(
            data, out, out, SimpleNamespace(**p["config"]["provider"], budget=b)
        )
    )
    assert result["decision"] == "correctness_regression_stop"
