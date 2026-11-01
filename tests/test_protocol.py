import asyncio
import json
from copy import deepcopy
import pytest
import httpx
from openai import AsyncOpenAI
from agentlab.protocol import extract_for_diagnosis, diagnose
from agentlab.provider import Provider
from agentlab.budget import Budget
from agentlab.data import build_dataset, load_split
from agentlab.agent import run_episode
from agentlab.prompts import MANUAL
from agentlab.graders import grade_rows
from agentlab.storage import write_json
from agentlab.experiment import identity


@pytest.mark.parametrize("raw", ['{} {}', 'text {invalid} {"action":"complete"}',
    '{"action":"complete","message":"hello","missing_fields":[]} trailing',
    '{"action":"complete","message":"hello","missing_fields":[2]}', '   '])
def test_ambiguous_or_invalid_output_rejected(raw):
    with pytest.raises(ValueError):
        extract_for_diagnosis(raw)


def test_offline_diagnosis_preserves_sources_and_business_failures(tmp_path):
    async def check():
        data = tmp_path / "data"
        build_dataset(data)
        task = load_split(data, "pilot")[0]
        ep = await run_episode(task, MANUAL, "demo")
        ep["raw_final"] = '说明\n' + json.dumps(ep["output"], ensure_ascii=False)
        ep["output"] = {"action":"failed", "message":"格式错误", "missing_fields":[]}
        ep["error"] = "invalid_final_json"
        row = {"task": task, "episode": ep}
        rows = await grade_rows([row])
        paths = [tmp_path / n for n in ("pilot_baseline.json", "pilot_manual.json", "efficiency_candidate.json")]
        for p in paths:
            write_json(p, rows)
        before = [p.read_bytes() for p in paths]
        report = await diagnose(tmp_path)
        assert all(v["strict_successes"] == 0 and v["diagnostic_successes_after_extraction"] == 1 for v in report["variants"].values())
        assert [p.read_bytes() for p in paths] == before
        bad = deepcopy(rows)
        bad[0]["episode"]["final_state"] = {}
        write_json(paths[0], bad)
        report = await diagnose(tmp_path)
        assert report["variants"]["baseline"]["diagnostic_successes_after_extraction"] == 0
    asyncio.run(check())


def test_json_output_changes_identity_and_mock_payload_without_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTLAB_API_KEY", "mock-only")
    async def check():
        budget = Budget(tmp_path / "budget.json", 3, 100000)
        plain = Provider(budget=budget)
        enabled = Provider(budget=budget, json_output=True)
        assert "response_format" not in plain.request_options
        assert identity(plain) != identity(enabled)
        await enabled.client.close()
        bodies = []
        def handler(req):
            bodies.append(json.loads(req.content))
            return httpx.Response(200, json={"id":"mock", "object":"chat.completion", "created":1, "model":"deepseek-flash", "choices":[{"index":0,"message":{"role":"assistant","content":"   "},"finish_reason":"stop"}], "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}})
        enabled.client = AsyncOpenAI(api_key="mock-only", base_url="https://api.deepseek.com", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        try:
            await enabled.chat([{"role":"user", "content":"Output JSON"}], tools=[{"name":"read", "parameters":{"type":"object", "properties":{}}}])
            assert bodies[0]["response_format"] == {"type":"json_object"}
            assert bodies[0]["tools"][0]["function"]["name"] == "read"
            assert len(bodies) == 1  # Blank JSON mode output must not trigger paid auto-retries.
        finally:
            await plain.close()
            await enabled.close()
    asyncio.run(check())
