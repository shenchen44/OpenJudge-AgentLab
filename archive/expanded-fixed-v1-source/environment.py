"""Executable synthetic service, isolated per task. Policies are discoverable data."""

from copy import deepcopy

TOOLS = [
    {
        "name": "get_ticket",
        "description": "读取工单详情与当前字段。",
        "parameters": {
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}},
            "required": ["ticket_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_customer",
        "description": "查询客户等级，可能为空；不能猜测缺失等级。",
        "parameters": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_policy",
        "description": "读取现行分诊政策、规则优先级、SLA、路由与故障处理约定。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "update_ticket",
        "description": "写入工单的优先级、团队与状态。TEMPORARY_UNAVAILABLE表示未提交，可重试。",
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"},
                "priority": {"type": "string", "enum": ["P1", "P2", "P3"]},
                "team": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["new", "in_progress", "resolved"],
                },
            },
            "required": ["ticket_id", "priority", "team", "status"],
            "additionalProperties": False,
        },
    },
]


class TicketEnvironment:
    def __init__(self, task, max_calls=8):
        self.world = deepcopy(task["world"])
        self.initial = deepcopy(self.world["ticket"])
        self.faults = deepcopy(task.get("faults", {}))
        self.events = []
        self.max_calls = max_calls

    def call(self, name, arguments):
        if len(self.events) >= self.max_calls:
            raise RuntimeError("tool_budget_exhausted")
        event = {"name": name, "arguments": deepcopy(arguments)}
        spec = next((t for t in TOOLS if t["name"] == name), None)
        if spec is None:
            result = {"error": "UNKNOWN_TOOL"}
        elif (
            not isinstance(arguments, dict)
            or set(arguments) != set(spec["parameters"]["required"])
            or any(not isinstance(v, str) for v in arguments.values())
        ):
            result = {"error": "INVALID_ARGUMENTS"}
        elif name == "get_ticket" and arguments["ticket_id"] != self.initial["id"]:
            result = {"error": "NOT_FOUND"}
        elif (
            name == "get_customer"
            and arguments["customer_id"] != self.world["customer"]["id"]
        ):
            result = {"error": "NOT_FOUND"}
        elif name == "update_ticket" and (
            arguments["ticket_id"] != self.initial["id"]
            or arguments["priority"] not in ("P1", "P2", "P3")
            or arguments["status"] not in ("new", "in_progress", "resolved")
            or not arguments["team"]
        ):
            result = {"error": "INVALID_ARGUMENTS"}
        elif self.faults.get(name, 0) > 0:
            self.faults[name] -= 1
            result = {"error": "TEMPORARY_UNAVAILABLE", "committed": False}
        elif name == "get_ticket":
            result = deepcopy(self.world["ticket"])
        elif name == "get_customer":
            result = deepcopy(self.world["customer"])
        elif name == "get_policy":
            result = deepcopy(self.world["policy"])
        else:
            self.world["ticket"].update(
                {k: arguments[k] for k in ("priority", "team", "status")}
            )
            result = {"ok": True, "ticket": deepcopy(self.world["ticket"])}
        event["result"] = result
        self.events.append(event)
        return result
