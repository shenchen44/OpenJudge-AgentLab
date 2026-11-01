"""No real API calls: exercise coordinator + real SDK with asynchronous HTTP mocks."""

import asyncio
import json
import re
from collections import Counter
from copy import deepcopy
import pytest
import httpx
from openai import AsyncOpenAI
from agentlab.data import build_dataset, load_split
from agentlab.agent import run_episode
from agentlab.prompts import MANUAL, BASELINE
from agentlab.storage import EpisodeCache
from agentlab.budget import Budget, BudgetExceeded
from agentlab.provider import Provider, ProviderFailure
from agentlab.experiment import evaluate
from agentlab.coordinator import execute_batch, WorkerTimeout


def run(c):
    return asyncio.run(c)


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "data"
    build_dataset(root)
    return root


def message_completion(message, pt=10, ct=5):
    return {
        "id": "mock",
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek-flash",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", **message},
                "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
    }


def tool(name, args, idx):
    return {
        "id": str(idx),
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def next_message(body, task):
    n = sum(m["role"] == "tool" for m in body["messages"])
    if n == 0:
        return {
            "content": None,
            "tool_calls": [tool("get_ticket", {"ticket_id": task["id"]}, 0)],
        }
    if n == 1:
        return {
            "content": None,
            "tool_calls": [
                tool(
                    "get_customer", {"customer_id": task["world"]["customer"]["id"]}, 1
                ),
                tool("get_policy", {}, 2),
            ],
        }
    if n == 3:
        state = task["expected"]["state"]
        return {
            "content": None,
            "tool_calls": [
                tool(
                    "update_ticket",
                    {
                        "ticket_id": task["id"],
                        **{k: state[k] for k in ("priority", "team", "status")},
                    },
                    3,
                )
            ],
        }
    return {
        "content": json.dumps(
            {"action": "complete", "message": "工单处理完成", "missing_fields": []},
            ensure_ascii=False,
        )
    }


async def make_provider(tmp_path, monkeypatch, handler, limit=100):
    monkeypatch.setenv("AGENTLAB_API_KEY", "unit-test-only")
    monkeypatch.setenv("AGENTLAB_BASE_URL", "https://api.deepseek.com")
    p = Provider(budget=Budget(tmp_path / "budget.json", limit, 1000000))
    await p.client.close()
    p.client = AsyncOpenAI(
        api_key="unit-test-only",
        base_url="https://api.deepseek.com",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return p


def test_duplicate_submissions_do_not_execute_or_count_twice(data, tmp_path):
    task = load_split(data, "pilot")[0]
    cache = EpisodeCache(tmp_path / "c.sqlite")
    try:
        rows = run(
            evaluate([task, deepcopy(task)], MANUAL, "demo", cache, concurrency=2)
        )
        assert len(rows) == 1
        assert rows[0]["batch_report"]["duplicates_dropped"] == 1
        assert len(cache.task_report(rows[0]["task_key"])["attempts"]) == 1
        again = run(evaluate([task], MANUAL, "demo", cache))
        assert again[0]["episode"]["cache_hit"]
        assert len(cache.task_report(rows[0]["task_key"])["attempts"]) == 1
    finally:
        cache.close()


def test_concurrent_usage_is_owned_by_request_and_context_is_isolated(
    data, tmp_path, monkeypatch
):
    tasks = load_split(data, "dev")[:2]
    lookup = {t["id"]: t for t in tasks}
    seen = Counter()
    active = 0
    peak = 0

    async def handler(req):
        nonlocal active, peak
        body = json.loads(req.content)
        tid = re.search(r"TK-[0-9a-f]{10}", body["messages"][1]["content"]).group()
        # No other ticket's messages or tool outputs may enter this context.
        text = json.dumps(body, ensure_ascii=False)
        assert all(other == tid or other not in text for other in lookup)
        seen[tid] += 1
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.015 if tid == tasks[0]["id"] else 0.004)
        active -= 1
        pt, ct = (11, 3) if tid == tasks[0]["id"] else (97, 7)
        return httpx.Response(
            200, json=message_completion(next_message(body, lookup[tid]), pt, ct)
        )

    async def check():
        p = await make_provider(tmp_path, monkeypatch, handler)
        cache = EpisodeCache(tmp_path / "episodes.sqlite")
        try:
            rows = await evaluate(tasks, MANUAL, "live", cache, p, concurrency=2)
            assert peak == 2
            assert rows[0]["batch_report"]["peak_active_workers"] == 2
            assert [r["episode"]["usage"]["requests"] for r in rows] == [4, 4]
            assert [r["episode"]["usage"]["prompt_tokens"] for r in rows] == [44, 388]
            assert [r["episode"]["usage"]["completion_tokens"] for r in rows] == [
                12,
                28,
            ]
            assert p.usage["prompt_tokens"] == 432
            assert (
                sum(r["episode"]["usage"]["prompt_tokens"] for r in rows)
                == p.usage["prompt_tokens"]
            )
            assert all(r["grades"]["outcome"]["score"] == 1 for r in rows)
            assert all(
                cache.task_report(r["task_key"])["state"] == "completed" for r in rows
            )
        finally:
            cache.close()
            await p.close()

    run(check())


def test_checkpoint_resume_does_not_repeat_completed_reads(data, tmp_path, monkeypatch):
    task = load_split(data, "pilot")[0]
    blocked = True
    sequence = []

    async def handler(req):
        body = json.loads(req.content)
        n = sum(m["role"] == "tool" for m in body["messages"])
        sequence.append(n)
        if n == 3 and blocked:
            return httpx.Response(
                401,
                json={
                    "error": {"message": "secret response body must not be persisted"}
                },
            )
        return httpx.Response(200, json=message_completion(next_message(body, task)))

    async def check():
        nonlocal blocked
        p = await make_provider(tmp_path, monkeypatch, handler)
        cache = EpisodeCache(tmp_path / "episodes.sqlite")
        try:
            with pytest.raises(ProviderFailure):
                await evaluate([task], MANUAL, "live", cache, p)
            saved = cache.tasks()[0]
            assert saved["state"] == "blocked" and saved["has_checkpoint"]
            snap = cache.checkpoint(saved["key"])
            assert snap["steps_done"] == 2 and len(snap["env"]["events"]) == 3
            blocked = False
            rows = await evaluate([task], MANUAL, "live", cache, p)
            ep = rows[0]["episode"]
            assert ep["resumed_from_step"] == 2
            assert sequence == [0, 1, 3, 3, 4]
            assert ep["usage"]["requests"] == 5 and ep["usage"]["usage_missing"] == 1
            assert ep["usage"]["prompt_tokens"] == 40
            assert rows[0]["grades"]["outcome"]["score"] == 1
            state = cache.task_report(rows[0]["task_key"])
            assert [a["state"] for a in state["attempts"]] == ["blocked", "completed"]
            assert not state["has_checkpoint"]
            raw = "".join(
                json.dumps(dict(r)) for r in cache.db.execute("SELECT * FROM requests")
            )
            assert "secret response body" not in raw and "unit-test-only" not in raw
        finally:
            cache.close()
            await p.close()

    run(check())


def test_cancel_inflight_is_unknown_and_does_not_reset_quota(
    data, tmp_path, monkeypatch
):
    task = load_split(data, "pilot")[0]
    entered = asyncio.Event()

    async def handler(req):
        entered.set()
        await asyncio.sleep(60)

    async def check():
        p = await make_provider(tmp_path, monkeypatch, handler)
        cache = EpisodeCache(tmp_path / "episodes.sqlite")
        try:
            future = asyncio.create_task(evaluate([task], BASELINE, "live", cache, p))
            await entered.wait()
            future.cancel()
            with pytest.raises(asyncio.CancelledError):
                await future
            state = cache.tasks()[0]
            assert state["state"] == "interrupted"
            assert (
                state["usage"]["requests"] == 1 and state["usage"]["usage_missing"] == 1
            )
            assert p.budget.state["requests"] == 1
            assert (
                cache.db.execute("SELECT state FROM requests").fetchone()[0]
                == "unknown"
            )
        finally:
            cache.close()
            await p.close()

    run(check())


def test_concurrent_requests_cannot_exceed_hard_budget(data, tmp_path, monkeypatch):
    tasks = load_split(data, "dev")[:2]
    lookup = {t["id"]: t for t in tasks}
    sent = 0

    async def handler(req):
        nonlocal sent
        sent += 1
        body = json.loads(req.content)
        tid = re.search(r"TK-[0-9a-f]{10}", body["messages"][1]["content"]).group()
        await asyncio.sleep(0.005)
        return httpx.Response(
            200, json=message_completion(next_message(body, lookup[tid]))
        )

    async def check():
        p = await make_provider(tmp_path, monkeypatch, handler, limit=3)
        cache = EpisodeCache(tmp_path / "episodes.sqlite")
        try:
            with pytest.raises(BudgetExceeded):
                await evaluate(tasks, MANUAL, "live", cache, p, concurrency=2)
            assert sent == 3 and p.budget.state["requests"] == 3
            assert cache.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 3
            assert all(
                t["state"] in ("interrupted", "completed") for t in cache.tasks()
            )
        finally:
            cache.close()
            await p.close()

    run(check())


def test_worker_timeout_is_persisted(data, tmp_path):
    task = load_split(data, "pilot")[0]
    cache = EpisodeCache(tmp_path / "c.sqlite")

    async def slow(task):
        await asyncio.sleep(1)

    try:
        with pytest.raises(WorkerTimeout):
            run(execute_batch([("key", task, 0)], cache, {}, slow, task_timeout=0.01))
        state = cache.task_report("key")
        assert (
            state["state"] == "blocked"
            and state["attempts"][0]["error"] == "TimeoutError"
        )
    finally:
        cache.close()


def test_process_restart_marks_stale_attempt_and_request_unknown(data, tmp_path):
    task = load_split(data, "pilot")[0]
    path = tmp_path / "c.sqlite"
    cache = EpisodeCache(path)
    entries = [("key", task, 0)]
    cache.enqueue("batch", entries, {})
    attempt = cache.claim("key")
    cache.start_request("key", task["id"], attempt, "actor", "test")
    cache.close()
    cache = EpisodeCache(path)
    try:
        cache.enqueue("batch", entries, {})
        state = cache.task_report("key")
        assert (
            state["state"] == "interrupted"
            and state["attempts"][0]["state"] == "interrupted"
        )
        assert cache.db.execute("SELECT state FROM requests").fetchone()[0] == "unknown"
        next_attempt = cache.claim("key")
        assert next_attempt != attempt
        assert cache.task_report("key")["attempts"][-1]["ordinal"] == 2
    finally:
        cache.close()


def test_cancel_and_retry_do_not_reroll_completed_model_failures(data, tmp_path):
    task = load_split(data, "pilot")[0]
    cache = EpisodeCache(tmp_path / "c.sqlite")
    try:
        cache.enqueue("batch", [("key", task, 0)], {})
        cache.cancel("key")
        with pytest.raises(RuntimeError, match="cancelled"):
            cache.claim("key")
        cache.retry("key")
        attempt = cache.claim("key")
        episode = run(run_episode(task, BASELINE, "demo"))
        cache.complete("key", attempt, episode)
        with pytest.raises(ValueError):
            cache.retry("key")
        with pytest.raises(ValueError):
            cache.cancel("key")
    finally:
        cache.close()


def test_checkpoint_restores_environment_and_remaining_steps(data, tmp_path):
    # Controlled interruption after a local write, before the model's final response.
    task = load_split(data, "pilot")[0]
    cache = EpisodeCache(tmp_path / "c.sqlite")
    from agentlab.accounting import RequestScope, request_scope
    from agentlab.environment import TicketEnvironment

    env = TicketEnvironment(task)
    for name, args in [
        ("get_ticket", {"ticket_id": task["id"]}),
        ("get_customer", {"customer_id": task["world"]["customer"]["id"]}),
        ("get_policy", {}),
    ]:
        env.call(name, args)
    target = task["expected"]["state"]
    env.call(
        "update_ticket",
        {
            "ticket_id": task["id"],
            **{k: target[k] for k in ("priority", "team", "status")},
        },
    )
    cache.enqueue("batch", [("key", task, 0)], {})
    attempt = cache.claim("key")
    cache.save_checkpoint(
        "key",
        {
            "steps_done": 3,
            "messages": [{"role": "user", "content": task["query"]}],
            "finishes": [],
            "env": {
                "world": env.world,
                "initial": env.initial,
                "faults": env.faults,
                "events": env.events,
            },
        },
    )

    class Message:
        content = '{"action":"complete","message":"工单处理完成","missing_fields":[]}'
        tool_calls = None

    class Fake:
        async def chat(self, *args, **kwargs):
            return Message(), "stop"

    async def check():
        with request_scope(RequestScope("key", task["id"], attempt, cache)):
            return await run_episode(task, MANUAL, "live", Fake())

    try:
        ep = run(check())
        assert ep["resumed_from_step"] == 3
        assert len([e for e in ep["events"] if e["name"] == "update_ticket"]) == 1
        assert ep["final_state"] == target
    finally:
        cache.close()
