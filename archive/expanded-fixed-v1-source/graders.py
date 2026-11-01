"""OpenJudge extensions. Evidence + outcome + forbidden side effects determine success."""

from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderScore
from openjudge.runner.grading_runner import GradingRunner


def trace_checks(task, episode):
    expected = task["expected"]
    seen = set()
    invalid = 0
    premature = 0
    writes = []
    bad_writes = 0
    for e in episode["events"]:
        result = e["result"]
        name = e["name"]
        if result.get("error") and result["error"] != "TEMPORARY_UNAVAILABLE":
            invalid += 1
        if name == "update_ticket":
            if not {"get_ticket", "get_customer", "get_policy"} <= seen:
                premature += 1
            if result.get("ok"):
                writes.append(e)
                if result.get("ticket") != expected["state"]:
                    bad_writes += 1
        if "error" not in result:
            seen.add(name)
    forbidden = expected["action"] == "clarify" and any(
        e["name"] == "update_ticket" for e in episode["events"]
    )
    redundant = max(0, len(writes) - expected["writes"])
    valid = (
        set(expected["required_tools"]) <= seen
        and invalid == 0
        and premature == 0
        and not forbidden
        and bad_writes == 0
        and redundant == 0
    )
    return {
        "valid": valid,
        "invalid_calls": invalid,
        "premature_writes": premature,
        "bad_writes": bad_writes,
        "redundant_writes": redundant,
        "forbidden_write": forbidden,
        "required_evidence_present": set(expected["required_tools"]) <= seen,
    }


class StateOutcomeGrader(BaseGrader):
    def __init__(self):
        super().__init__(
            name="outcome",
            description="Committed state, structured clarification, evidence and side effects",
        )

    async def _aevaluate(self, task, episode, **kwargs):
        expected = task["expected"]
        output = episode["output"]
        trace = trace_checks(task, episode)
        fields = output.get("missing_fields", [])
        message = output.get("message", "")
        correct = (
            episode["final_state"] == expected["state"]
            and output.get("action") == expected["action"]
            and isinstance(fields, list)
            and all(isinstance(v, str) for v in fields)
            and sorted(fields) == sorted(expected["missing_fields"])
            and isinstance(message, str)
            and len(message.strip()) >= 4
            and not episode.get("error")
            and trace["valid"]
        )
        if expected["action"] == "clarify":
            terms = {
                "category": ("类别", "分类", "category"),
                "customer_tier": ("等级", "级别", "tier"),
            }
            correct = correct and all(
                any(term in message.lower() for term in terms[f])
                for f in expected["missing_fields"]
            )
        # For write-required tasks state cannot be trusted without a successful commit event.
        if expected["writes"] == 1:
            correct = correct and any(
                e["name"] == "update_ticket" and e["result"].get("ok")
                for e in episode["events"]
            )
        return GraderScore(
            name=self.name,
            score=float(correct),
            reason="Verified full task contract" if correct else "Contract violation",
            metadata=trace,
        )


class ToolTraceGrader(BaseGrader):
    def __init__(self):
        super().__init__(
            name="tool_trace",
            description="Necessary dependencies, parameter validity and side effects",
        )

    async def _aevaluate(self, task, episode, **kwargs):
        checks = trace_checks(task, episode)
        return GraderScore(
            name=self.name,
            score=float(checks["valid"]),
            reason=str(checks),
            metadata=checks,
        )


async def grade_rows(rows):
    result = await GradingRunner(
        {"outcome": StateOutcomeGrader(), "tool_trace": ToolTraceGrader()},
        show_progress=False,
        max_concurrency=8,
    ).arun(rows)
    for i, row in enumerate(rows):
        row["grades"] = {}
        for name, values in result.items():
            value = values[i]
            if not isinstance(value, GraderScore):
                raise RuntimeError("OpenJudge grader failed: " + name)
            row["grades"][name] = value.model_dump()
    return rows
