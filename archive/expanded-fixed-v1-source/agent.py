"""Bounded actor. Infrastructure failures abort early; model failures remain feedback."""

import json
import re
import time
from .environment import TicketEnvironment, TOOLS
from .prompts import MANUAL, DEMO_PATCH
from .accounting import RequestScope, current_scope, request_scope
from .storage import digest


def scripted_episode(query, prompt, env):
    """Scripted fixture only, never evidence of real LLM improvement."""
    tid = re.search(r"TK-[0-9a-f]{10}", query).group()
    robust = prompt == MANUAL or DEMO_PATCH in prompt

    def read(name, args):
        result = env.call(name, args)
        if result.get("error") == "TEMPORARY_UNAVAILABLE" and robust:
            result = env.call(name, args)
        return result

    t = read("get_ticket", {"ticket_id": tid})
    if "error" in t:
        return {"action": "failed", "message": "读取失败", "missing_fields": []}
    if t["category"] is None:
        return {
            "action": "clarify",
            "message": "请补充工单类别",
            "missing_fields": ["category"],
        }
    c = read("get_customer", {"customer_id": t["customer_id"]})
    if c["tier"] is None:
        return {
            "action": "clarify",
            "message": "请补充客户等级",
            "missing_fields": ["customer_tier"],
        }
    p = read("get_policy", {})
    if "error" in p:
        return {"action": "failed", "message": "政策读取失败", "missing_fields": []}
    priority = (
        "P1"
        if c["tier"] in p["priority_tiers"]
        or t["age_minutes"] > p["sla_minutes"][c["tier"]]
        else "P2"
    )
    if robust:
        priority = p["category_priority"].get(t["category"], priority)
    args = {
        "ticket_id": tid,
        "priority": priority,
        "team": p["routing"][t["category"]],
        "status": "in_progress",
    }
    equal = all(t[k] == args[k] for k in ("priority", "team", "status"))
    if not (equal and robust):
        result = read("update_ticket", args)
        if "error" in result:
            return {"action": "failed", "message": "写入未完成", "missing_fields": []}
    return {"action": "complete", "message": "工单处理完成", "missing_fields": []}


async def run_episode(task, prompt, mode, provider=None, max_steps=8):
    scope = current_scope()
    if scope is not None:
        return await _run_episode(task, prompt, mode, provider, max_steps, scope)
    scope = RequestScope(
        digest({"task": task, "prompt": prompt, "mode": mode}), task["id"]
    )
    with request_scope(scope):
        return await _run_episode(task, prompt, mode, provider, max_steps, scope)


async def _run_episode(task, prompt, mode, provider, max_steps, scope):
    env = TicketEnvironment(task, max_calls=max_steps)
    started = time.perf_counter()
    output = {"action": "failed", "message": "执行轮数用尽", "missing_fields": []}
    error = None
    finishes = []
    raw_final = None
    checkpoint = (
        scope.store.checkpoint(scope.task_key)
        if scope.store is not None and scope.attempt_id
        else None
    )
    restored_steps = checkpoint["steps_done"] if checkpoint else 0
    if checkpoint:
        saved = checkpoint["env"]
        env.world = saved["world"]
        env.initial = saved["initial"]
        env.faults = saved["faults"]
        env.events = saved["events"]
        finishes = checkpoint["finishes"]
    if mode == "demo":
        try:
            output = scripted_episode(task["query"], prompt, env)
        except RuntimeError:
            error = "tool_budget_exhausted"
    else:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": task["query"]},
        ]
        if checkpoint:
            messages = checkpoint["messages"]
        for step in range(restored_steps, max_steps):
            # ProviderFailure/BudgetExceeded intentionally propagate to phase checkpoint.
            msg, finish = await provider.chat(messages, tools=TOOLS)
            finishes.append(finish)
            assistant = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                assistant["tool_calls"] = [
                    c.model_dump(exclude_none=True) for c in msg.tool_calls
                ]
            messages.append(assistant)
            if finish == "length":
                error = "output_truncated"
                break
            if not msg.tool_calls:
                raw_final = (msg.content or "").strip()
                text = raw_final
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0]
                try:
                    output = json.loads(text)
                    if (
                        not isinstance(output, dict)
                        or output.get("action") not in ("complete", "clarify", "failed")
                        or not isinstance(output.get("message"), str)
                        or not isinstance(output.get("missing_fields"), list)
                    ):
                        raise ValueError("invalid output schema")
                    if any(not isinstance(v, str) for v in output["missing_fields"]):
                        raise ValueError("invalid missing_fields")
                except (ValueError, TypeError, IndexError):
                    output = {
                        "action": "failed",
                        "message": "最终输出格式不合规",
                        "missing_fields": [],
                    }
                    error = "invalid_final_json"
                break
            if len(env.events) + len(msg.tool_calls) > max_steps:
                error = "tool_budget_exhausted"
                break
            for call in msg.tool_calls:
                try:
                    args = json.loads(call.function.arguments)
                except (ValueError, TypeError):
                    args = None
                result = env.call(call.function.name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            if scope.store is not None and scope.attempt_id:
                scope.store.save_checkpoint(
                    scope.task_key,
                    {
                        "steps_done": step + 1,
                        "messages": messages,
                        "finishes": finishes,
                        "env": {
                            "world": env.world,
                            "initial": env.initial,
                            "faults": env.faults,
                            "events": env.events,
                        },
                    },
                )
        else:
            error = "step_budget_exhausted"
    usage = (
        scope.store.request_usage(scope.task_key)
        if scope.store is not None and scope.attempt_id
        else scope.usage.copy()
    )
    return {
        "task_id": task["id"],
        "mode": mode,
        "output": output,
        "initial_state": env.initial,
        "final_state": env.world["ticket"],
        "events": env.events,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "latency_scope": "this_attempt_only; resumed tasks exclude earlier attempt duration",
        "usage": usage,
        "error": error,
        "error_kind": "model_failure" if error else None,
        "finish_reasons": finishes,
        "raw_final": raw_final,
        "cache_hit": False,
        "resumed_from_step": restored_steps,
        "usage_scope": "all_recorded_attempts"
        if scope.attempt_id
        else "this_invocation",
        "assignment_id": scope.attempt_id,
    }
