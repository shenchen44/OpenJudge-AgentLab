"""Static, self-contained experiment report. Escapes all model-authored content."""

import html
import json
from pathlib import Path


def render_report(directory):
    directory = Path(directory)
    result = json.loads((directory / "summary.json").read_text())
    mode = result["config"]["mode"]
    title = (
        "离线脚本演示 · 不代表大模型调优效果"
        if mode == "demo"
        else "真实模型实验 · 合成工单数据"
    )
    rows = ""
    for name, data in result["test"].items():
        ci = data["success_wilson95"]
        rows += (
            f"<tr><td>{name}</td><td>{data['n']}</td><td>{data['success_rate']:.1%}</td>"
            f"<td>{(f'{ci[0]:.1%}–{ci[1]:.1%}' if ci else '任务重复：见配对分析')}</td><td>{data['tool_trace_rate']:.1%}</td>"
            f"<td>{data['mean_tool_calls']:.2f}</td><td>{data['episode_tokens']}</td><td>{data['execution_errors']}</td></tr>"
        )
    families = ""
    for family in result["test"]["baseline"]["family_success"]:
        cells = "".join(
            f"<td>{result['test'][label]['family_success'][family]:.1%}</td>"
            for label in ("baseline", "manual", "optimized")
        )
        families += f"<tr><td>{html.escape(family)}</td>{cells}</tr>"
    cases = []
    for label in ("baseline", "optimized"):
        for row in json.loads((directory / f"test_{label}.json").read_text()):
            if row["grades"]["outcome"]["score"] == 0:
                cases.append(
                    {
                        "variant": label,
                        "query": row["task"]["query"],
                        "episode": row["episode"],
                        "expected": row["task"]["expected"],
                    }
                )
    details = html.escape(json.dumps(cases[:12], ensure_ascii=False, indent=2))
    prompt = html.escape(result["frozen"]["prompt"])
    paired = html.escape(
        json.dumps(result.get("paired", {}), ensure_ascii=False, indent=2)
    )
    history = html.escape(json.dumps(result["history"], ensure_ascii=False, indent=2))
    usage = html.escape(
        json.dumps(
            {
                "actor": result["provider_usage_this_invocation"],
                "optimizer": result["optimizer_usage_this_invocation"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    runtime = html.escape(
        json.dumps(result.get("execution_store", {}), ensure_ascii=False, indent=2)
    )
    batches = html.escape(
        json.dumps(
            {k: v.get("batch_report") for k, v in result["test"].items()},
            ensure_ascii=False,
            indent=2,
        )
    )
    output = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenJudge-AgentLab 实验报告</title><style>
body{{font:16px/1.7 system-ui,sans-serif;background:#f4f6fa;color:#19243b;margin:0}}main{{max-width:1100px;margin:40px auto;padding:30px;background:white;border-radius:18px}}
h1{{font-size:30px}}.badge{{padding:14px;background:#fff1cb;border-left:5px solid #d29000}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #dde3ee}}th{{background:#edf3ff}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#101b30;color:#deebff;padding:20px;border-radius:10px}}.scroll{{overflow-x:auto}}small{{color:#58677f}}</style>
<main><small>OPENJUDGE / AGENTLAB</small><h1>企业 Agent 评测与自动调优</h1><p class="badge">{title}</p>
<p>主指标同时检查最终状态、必要查询证据、实际澄清及禁止的中间副作用。优化只读取开发集；验证集选版本；冻结后再读取测试集。</p>
<div class="scroll"><table><tr><th>版本</th><th>任务次数</th><th>成功率</th><th>Wilson 95%区间*</th><th>轨迹合规率</th><th>平均工具调用</th><th>原始执行Token</th><th>执行异常</th></tr>{rows}</table></div>
<small>*重复试验并非独立任务；区间仅供描述。缓存复用不等于再次请求模型。Demo模式Token为0，不是成本优势。</small>
<h2>按失败场景分组</h2><table><tr><th>场景</th><th>Baseline</th><th>Manual</th><th>Optimized</th></tr>{families}</table>
<h2>按任务配对的成功率差异</h2><pre>{paired}</pre><h2>冻结的Prompt</h2><pre>{prompt}</pre><details><summary>候选版本与验证筛选记录</summary><pre>{history}</pre></details>
<details><summary>本次模型请求用量（执行与优化分开）</summary><pre>{usage}</pre></details>
<h2>持久化任务与请求账本</h2><pre>{runtime}</pre>
<details><summary>各批次调度统计</summary><pre>{batches}</pre></details>
<p>批次耗时包含本地调度和存储。缓存复用不算模型加速；并发不自动降低调用次数或费用。未知用量请求可能已计费。</p>
<h2>失败轨迹（最多12条）</h2><pre>{details}</pre>
<p>数据为程序生成的受控场景，不证明企业生产效果。真实招聘展示需报告模型、预算、独立标注及数据局限，不能把脚本演示指标写为模型优化收益。</p>
</main></html>"""
    (directory / "report.html").write_text(output, encoding="utf-8")
    return directory / "report.html"
