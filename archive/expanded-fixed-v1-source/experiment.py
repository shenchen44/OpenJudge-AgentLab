"""Staged experiment: pilot -> search/freeze -> explicit final holdout evaluation."""

import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from . import __version__
from .agent import run_episode
from .data import load_split
from .environment import TOOLS
from .graders import grade_rows
from .prompts import BASELINE, MANUAL, DEMO_PATCH
from .storage import digest, write_json, EpisodeCache
from .coordinator import execute_batch
from .accounting import RequestScope, request_scope


def summarize(rows):
    if not rows:
        raise ValueError("Cannot summarize empty evaluation")
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"]["id"], []).append(r["grades"]["outcome"]["score"])
    task_means = [statistics.mean(v) for v in by_task.values()]
    n = len(rows)
    successes = sum(r["grades"]["outcome"]["score"] for r in rows)
    p = statistics.mean(task_means)
    m = len(task_means)
    z = 1.96
    center = (p + z * z / (2 * m)) / (1 + z * z / m)
    margin = z * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / (1 + z * z / m)
    tokens = [
        r["episode"]["usage"]["prompt_tokens"]
        + r["episode"]["usage"]["completion_tokens"]
        for r in rows
    ]
    return {
        "mode": rows[0]["episode"]["mode"],
        "usage_missing": sum(
            r["episode"]["usage"].get("usage_missing", 0) for r in rows
        ),
        "batch_report": rows[0].get("batch_report"),
        "n": n,
        "unique_tasks": m,
        "successes": int(successes),
        "success_rate": p,
        "success_wilson95": [center - margin, center + margin] if n == m else None,
        "tool_trace_rate": statistics.mean(
            r["grades"]["tool_trace"]["score"] for r in rows
        ),
        "mean_tool_calls": statistics.mean(len(r["episode"]["events"]) for r in rows),
        "mean_latency_ms": statistics.mean(r["episode"]["latency_ms"] for r in rows),
        "model_failures": dict(
            Counter(r["episode"]["error"] for r in rows if r["episode"]["error"])
        ),
        "execution_errors": sum(bool(r["episode"]["error"]) for r in rows),
        "cache_hits": sum(r["episode"]["cache_hit"] for r in rows),
        "episode_tokens": sum(tokens),
        "mean_tokens": statistics.mean(tokens),
        "uncached_episode_tokens": sum(
            t for t, r in zip(tokens, rows) if not r["episode"]["cache_hit"]
        ),
        "invalid_calls": sum(
            r["grades"]["tool_trace"]["metadata"]["invalid_calls"] for r in rows
        ),
        "redundant_writes": sum(
            r["grades"]["tool_trace"]["metadata"]["redundant_writes"] for r in rows
        ),
        "family_success": {
            f: statistics.mean(
                r["grades"]["outcome"]["score"]
                for r in rows
                if r["task"]["family"] == f
            )
            for f in sorted({r["task"]["family"] for r in rows})
        },
    }


def identity(provider):
    return (
        {
            "model": provider.model,
            "endpoint": provider.endpoint,
            "request_options": provider.request_options,
        }
        if provider
        else {"model": "scripted-fixture-v2"}
    )


async def evaluate(
    tasks,
    prompt,
    mode,
    cache,
    provider=None,
    repeats=1,
    max_steps=8,
    progress=None,
    concurrency=1,
    task_timeout=300,
):
    entries = []
    metadata = {
        "engine": __version__,
        "prompt": digest(prompt),
        "mode": mode,
        "provider": identity(provider),
        "steps": max_steps,
        "concurrency": concurrency,
        "task_timeout": task_timeout,
    }
    for task in tasks:
        for repeat in range(repeats):
            key = digest(
                {
                    "engine": __version__,
                    "tools": TOOLS,
                    "task": task,
                    "prompt": prompt,
                    "mode": mode,
                    "provider": identity(provider),
                    "repeat": repeat,
                    "steps": max_steps,
                }
            )
            entries.append((key, task, repeat))

    async def runner(task):
        return await run_episode(task, prompt, mode, provider, max_steps)

    rows = await execute_batch(
        entries, cache, metadata, runner, concurrency, task_timeout, progress
    )
    return await grade_rows(rows)


def dev_feedback(rows):
    failures = [r for r in rows if not r["grades"]["outcome"]["score"]]
    counts = Counter(r["task"]["family"] for r in failures)
    examples = []
    for family in sorted(counts):
        r = next(r for r in failures if r["task"]["family"] == family)
        examples.append(
            {
                "query": r["task"]["query"],
                "events": r["episode"]["events"],
                "output": r["episode"]["output"],
                "expected": r["task"]["expected"],
                "final_state": r["episode"]["final_state"],
                "error": r["episode"]["error"],
                "feedback": r["grades"],
            }
        )
    # A correct but verbose dev trajectory can support efficiency optimization.
    efficient = sorted(
        [r for r in rows if r["grades"]["outcome"]["score"]],
        key=lambda r: len(r["episode"]["events"]),
        reverse=True,
    )[:2]
    return {
        "failure_counts": dict(counts),
        "examples": examples,
        "efficiency_examples": [
            {
                "query": r["task"]["query"],
                "events": r["episode"]["events"],
                "tool_calls": len(r["episode"]["events"]),
            }
            for r in efficient
        ],
    }


async def propose(prompt, feedback, mode, optimizer=None):
    if mode == "demo":
        return prompt + DEMO_PATCH if DEMO_PATCH not in prompt else None
    result = await optimizer.json(
        [
            {
                "role": "system",
                "content": "你是Agent系统Prompt优化器。基于开发集轨迹总结可泛化的改进。保留合法JSON输出协议。"
                "不记忆具体工单、客户、团队ID。政策是动态的，不要硬编码VIP或类别优先级。"
                "成功率优先，不降低正确性前提下减少重复读取、无效写入，独立只读工具可一轮并行调用。"
                '不得增加预算或修改工具。返回严格JSON {"prompt":"完整Prompt","rationale":"原因"}。',
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"current_prompt": prompt, "development_feedback": feedback},
                    ensure_ascii=False,
                ),
            },
        ]
    )
    text = result.get("prompt")
    if not isinstance(text, str) or not 20 <= len(text) <= 6000:
        raise ValueError("Invalid candidate prompt")
    import re

    if re.search(r"(?:TK-|TEAM-|C-)[0-9a-f]{6,}", text):
        raise ValueError("Candidate memorizes entity IDs")
    return text


def accepted(candidate, baseline):
    if candidate["success_rate"] != baseline["success_rate"]:
        return candidate["success_rate"] > baseline["success_rate"]
    if candidate["tool_trace_rate"] < baseline["tool_trace_rate"]:
        return False
    if baseline.get("usage_missing", 0) or candidate.get("usage_missing", 0):
        return False
    if baseline["mean_tokens"] > 0:
        return candidate["mean_tokens"] <= 0.95 * baseline["mean_tokens"]
    return (
        baseline.get("mode") == "demo"
        and candidate["mean_tool_calls"] < baseline["mean_tool_calls"]
    )


def pairwise(base, other):
    b = {}
    o = {}
    for r in base:
        b.setdefault(r["task"]["id"], []).append(r["grades"]["outcome"]["score"])
    for r in other:
        o.setdefault(r["task"]["id"], []).append(r["grades"]["outcome"]["score"])
    if set(b) != set(o):
        raise ValueError("Paired task sets differ")
    deltas = [statistics.mean(o[k]) - statistics.mean(b[k]) for k in sorted(b)]
    rng = random.Random(42)
    samples = sorted(
        statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(2000)
    )
    return {
        "task_paired_success_delta_pp": statistics.mean(deltas) * 100,
        "bootstrap95_delta_pp": [samples[49] * 100, samples[1949] * 100],
        "improved_tasks": sum(d > 0 for d in deltas),
        "regressed_tasks": sum(d < 0 for d in deltas),
        "note": "Task-level paired bootstrap on small synthetic data; exploratory, not a production claim",
    }


def bound_config(out, config, name="config.json"):
    path = Path(out) / name
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError(
            "Output directory belongs to another experiment; choose a new --out"
        )
    write_json(path, config)


def progress(done, total, tid, episode):
    print(
        f"[{done}/{total}] {tid} cache={episode['cache_hit']} error={episode['error']}",
        flush=True,
    )


async def pilot(
    data_dir, out, mode, provider=None, max_steps=8, concurrency=1, task_timeout=300
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_split(data_dir, "pilot")
    config = {
        "engine": __version__,
        "mode": mode,
        "data_manifest": digest(
            json.loads((Path(data_dir) / "manifest.json").read_text())
        ),
        "pilot_hash": digest(tasks),
        "provider": identity(provider),
        "max_steps": max_steps,
        "concurrency": concurrency,
        "task_timeout": task_timeout,
        "baseline": digest(BASELINE),
        "manual": digest(MANUAL),
    }
    bound_config(out, config)
    cache = EpisodeCache(out / "episodes.sqlite")
    summaries = {}
    try:
        for label, prompt in (("baseline", BASELINE), ("manual", MANUAL)):
            rows = await evaluate(
                tasks,
                prompt,
                mode,
                cache,
                provider,
                max_steps=max_steps,
                progress=progress,
                concurrency=concurrency,
                task_timeout=task_timeout,
            )
            write_json(out / f"pilot_{label}.json", rows)
            summaries[label] = summarize(rows)
    finally:
        cache.close()
    a, b = summaries["baseline"], summaries["manual"]
    if b["execution_errors"]:
        decision = "inspect_model_failures"
    elif a["success_rate"] >= 11 / 12:
        decision = (
            "ready_efficiency"
            if b["success_rate"] >= a["success_rate"]
            and not a.get("usage_missing", 0)
            and not b.get("usage_missing", 0)
            and a["mean_tokens"] > 0
            and b["mean_tokens"] <= 0.9 * a["mean_tokens"]
            else "ceiling_or_no_headroom"
        )
    elif accepted(b, a):
        decision = "ready"
    else:
        decision = "no_observed_gain"
    result = {
        "kind": "pilot",
        "config": config,
        "metrics": summaries,
        "decision": decision,
        "note": "Pilot gate is a small-sample heuristic, not statistical evidence of improvement",
    }
    write_json(out / "pilot.json", result)
    return result


async def optimize(
    data_dir,
    out,
    mode,
    provider=None,
    optimizer=None,
    rounds=1,
    repeats=1,
    max_steps=8,
    pilot_dir=None,
    concurrency=1,
    task_timeout=300,
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = digest(json.loads((Path(data_dir) / "manifest.json").read_text()))
    config = {
        "engine": __version__,
        "mode": mode,
        "manifest": manifest,
        "provider": identity(provider),
        "optimizer": identity(optimizer),
        "rounds": rounds,
        "repeats": repeats,
        "max_steps": max_steps,
        "concurrency": concurrency,
        "task_timeout": task_timeout,
        "baseline": digest(BASELINE),
        "manual": digest(MANUAL),
    }
    if mode == "live":
        if pilot_dir is None:
            raise ValueError("Live search requires --pilot from a completed pilot")
        p = json.loads((Path(pilot_dir) / "pilot.json").read_text())
        if (
            p["decision"] not in ("ready", "ready_efficiency")
            or p["config"]["mode"] != "live"
            or p["config"]["provider"] != identity(provider)
            or p["config"]["data_manifest"] != manifest
            or p["config"]["max_steps"] != max_steps
            or p["config"].get("engine") != __version__
            or p["config"].get("concurrency") != concurrency
            or p["config"].get("task_timeout") != task_timeout
        ):
            raise ValueError(
                "Pilot is not ready or model/data/budget differs; inspect pilot before spending more"
            )
        config["pilot_sha256"] = digest(p)
    bound_config(out, config)
    freeze = out / "frozen.json"
    if freeze.exists():
        frozen = json.loads(freeze.read_text())
        if frozen["config_sha256"] != digest(config) or frozen[
            "prompt_sha256"
        ] != digest(frozen["prompt"]):
            raise ValueError("Frozen integrity mismatch")
        return frozen
    dev = load_split(data_dir, "dev")
    validation = load_split(data_dir, "validation")
    cache = EpisodeCache(out / "episodes.sqlite")
    history_path = out / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    try:
        rows = await evaluate(
            validation,
            BASELINE,
            mode,
            cache,
            provider,
            repeats,
            max_steps,
            progress,
            concurrency,
            task_timeout,
        )
        write_json(out / "validation_baseline.json", rows)
        best_prompt = BASELINE
        best = summarize(rows)
        for entry in history:
            if entry["accepted"]:
                best_prompt, best = entry["prompt"], entry["validation"]
        for iteration in range(len(history), rounds):
            dev_rows = await evaluate(
                dev,
                best_prompt,
                mode,
                cache,
                provider,
                repeats,
                max_steps,
                progress,
                concurrency,
                task_timeout,
            )
            write_json(out / f"dev_round_{iteration}.json", dev_rows)
            feedback = dev_feedback(dev_rows)
            candidate_path = out / f"candidate_{iteration}.json"
            if candidate_path.exists():
                candidate = json.loads(candidate_path.read_text())["prompt"]
            else:
                try:
                    scope = RequestScope(
                        f"optimizer:{digest(config)}:{iteration}",
                        f"candidate-{iteration}",
                        store=cache,
                    )
                    with request_scope(scope):
                        candidate = await propose(
                            best_prompt, feedback, mode, optimizer
                        )
                except (ValueError, TypeError, IndexError) as exc:
                    write_json(
                        out / f"rejected_candidate_{iteration}.json",
                        {"error": type(exc).__name__},
                    )
                    break
                write_json(candidate_path, {"prompt": candidate})
            if candidate is None or digest(candidate) in {
                digest(BASELINE),
                *(e["prompt_sha256"] for e in history),
            }:
                break
            rows = await evaluate(
                validation,
                candidate,
                mode,
                cache,
                provider,
                repeats,
                max_steps,
                progress,
                concurrency,
                task_timeout,
            )
            summary = summarize(rows)
            keep = accepted(summary, best)
            entry = {
                "iteration": iteration,
                "prompt": candidate,
                "prompt_sha256": digest(candidate),
                "development_failures": feedback["failure_counts"],
                "validation": summary,
                "accepted": keep,
            }
            history.append(entry)
            write_json(history_path, history)
            write_json(out / f"validation_candidate_{iteration}.json", rows)
            if keep:
                best_prompt, best = candidate, summary
        frozen = {
            "prompt": best_prompt,
            "prompt_sha256": digest(best_prompt),
            "validation": best,
            "config_sha256": digest(config),
            "note": "Frozen without reading test. Run finalize explicitly after reviewing search.",
        }
        write_json(freeze, frozen)
        return frozen
    finally:
        cache.close()


async def finalize(
    data_dir,
    out,
    mode,
    provider=None,
    max_steps=8,
    repeats=1,
    concurrency=1,
    task_timeout=300,
):
    out = Path(out)
    config = json.loads((out / "config.json").read_text())
    frozen = json.loads((out / "frozen.json").read_text())
    manifest = digest(json.loads((Path(data_dir) / "manifest.json").read_text()))
    if (
        config.get("engine") != __version__
        or config["mode"] != mode
        or config["provider"] != identity(provider)
        or config["manifest"] != manifest
        or config["max_steps"] != max_steps
        or config["repeats"] != repeats
        or config.get("concurrency") != concurrency
        or config.get("task_timeout") != task_timeout
        or frozen["config_sha256"] != digest(config)
        or frozen["prompt_sha256"] != digest(frozen["prompt"])
    ):
        raise ValueError(
            "Finalization must use identical model, data, repeats and task budget"
        )
    if (out / "summary.json").exists():
        return json.loads((out / "summary.json").read_text())
    test = load_split(data_dir, "test")
    cache = EpisodeCache(out / "episodes.sqlite")
    all_rows = {}
    summaries = {}
    try:
        for label, prompt in (
            ("baseline", BASELINE),
            ("manual", MANUAL),
            ("optimized", frozen["prompt"]),
        ):
            rows = await evaluate(
                test,
                prompt,
                mode,
                cache,
                provider,
                repeats,
                max_steps,
                progress,
                concurrency,
                task_timeout,
            )
            write_json(out / f"test_{label}.json", rows)
            all_rows[label] = rows
            summaries[label] = summarize(rows)
        execution_store = cache.execution_summary()
    finally:
        cache.close()
    result = {
        "config": config,
        "frozen": frozen,
        "test": summaries,
        "dataset_test_sha256": digest(test),
        "history": json.loads((out / "history.json").read_text())
        if (out / "history.json").exists()
        else [],
        "paired": {
            k: pairwise(all_rows["baseline"], all_rows[k])
            for k in ("manual", "optimized")
        },
        "execution_store": execution_store,
        "budget": provider.budget.state if provider else None,
        "provider_usage_this_invocation": provider.usage if provider else None,
        "optimizer_usage_this_invocation": None,
        "disclaimer": "SCRIPTED FIXTURE, NOT LLM EVIDENCE"
        if mode == "demo"
        else "Small synthetic benchmark; no guaranteed positive result",
    }
    write_json(out / "summary.json", result)
    return result
