# OpenJudge-AgentLab

面向工具调用 Agent 的**执行校验、反馈修复与可复核评测**。基于 OpenJudge 扩展业务评分器，在可执行的合成工单环境中检查 Agent 是否真正完成了操作。

本项目的核心对比保持模型与系统 Prompt 一致，改变执行机制：普通执行、执行网关、执行网关加一次终态反馈修复。它是个人工程与实验项目，未接入企业生产系统，也未训练模型权重。

## 实验结果

在 120 条新生成的合成工单上进行三组配对实验，共 360 次真实任务执行。三组均使用 DeepSeek Flash、JSON 模式、相同 MANUAL Prompt、8 轮模型交互与 8 次工具尝试上限。

| 指标 | 普通执行 | 网关，无终态修复 | 网关 + 一次终态修复 |
| --- | ---: | ---: | ---: |
| 成功任务 | 106/120 | 114/120 | 120/120 |
| 成功率 | 88.3% | 95.0% | 100% |
| 发生错误写入的任务 | 13 | 0 | 0 |
| 发生重复写入的任务 | 1 | 0 | 0 |
| HTTP 请求尝试 | 481 | 491 | 494 |

完整方案相较普通执行提高 **11.7 个百分点**，分层任务配对 bootstrap 95% 区间为 **7.5～16.7 个百分点**；配对 exact McNemar p = 0.000122。请求数增加 **2.7%**。完整方案拦截 11 次不合规工具提案，4 个任务使用一次终态反馈修复并通过最终评分。

- [完整结果与成本说明](results/gateway-ablation-v1/result.json)
- [冻结配置](results/gateway-ablation-v1/experiment_config.json)
- [普通执行轨迹](results/gateway-ablation-v1/plain_rows.json) · [网关轨迹](results/gateway-ablation-v1/guard_only_rows.json) · [修复轨迹](results/gateway-ablation-v1/guard_repair_rows.json)
- [请求级脱敏用量](results/gateway-ablation-v1/request_accounting.json)
- [该轮冻结源码](archive/gateway-ablation-v1-source/)

本轮累计 1,466/1,800 次请求。有一次 API 超时，其 tokens/费用未知，已计入请求上限。因此不声称精确的完整 token 成本增幅；`reported_tokens_overhead_pct` 仅比较已返回的用量，`token_overhead_pct` 保持 null。`result.raw.json` 保留成本展示修正前的原始报告，修正未改变任务结果或评分。

**结果边界**：120 条受控合成任务、每组每题单次执行；本批次 120/120 不代表真实线上或任意任务成功率。不同实验组是独立模型调用，组间通过数量差不等于某一组内触发修复次数。次要消融比较用于辅助分析。

## 任务是什么

用户要求“处理工单并完成分诊”，Agent 自行调用四个工具：

| 工具 | 职责 |
| --- | --- |
| `get_ticket` | 查询工单类别、等待时长、当前状态 |
| `get_customer` | 查询客户等级 |
| `get_policy` | 查询 SLA、优先级、团队路由及例外规则 |
| `update_ticket` | 实际更新优先级、负责团队与状态 |

任务覆盖普通分诊、VIP、SLA 超期、SLA 等号边界、动态路由、写入临时故障、缺失类别、缺失客户等级、幂等处理、读取临时故障、组合故障和政策例外等 12 类。新测试集每类 10 条，变化包含政策数值、客户等级、初始字段、故障位置与组合；排除仅更换实体 ID 的重复。

工具后端是本地模拟服务，模型请求是真实 API。评分要求最终状态、必要查询、有效澄清和副作用同时满足合同。JSON 合规或工具调用合法，并不等于业务完成。

## 核心实现

- **写入前校验**：网关只使用成功的工具响应建立证据状态，对照公开的结构化政策校验写入参数。缺少证据或不符合规则时，提案不会进入后端。
- **本地幂等检查**：已观察到目标字段一致时跳过写入；本轮真实修复组没有触发跳过，相关行为由本地测试覆盖。
- **终态验收与有界修复**：不接受缺少提交证据的完成声明。至多发起一次终态反馈修复，仍消耗原有模型轮数和工具尝试预算；修复后再次失败则明确拒绝完成声明。
- **完整审计**：记录被拦截、跳过的调用和被拒绝的原始完成提案，区分模型尝试的错误与实际产生的副作用。被阻止的操作不伪装成后端已执行的操作。
- **隔离与恢复**：任务与 Attempt 分离、独立消息和工单环境、ContextVar 请求归属；证据、审计、修复计数与消息/环境检查点共同恢复，已用预算不会重置。
- **标签隔离**：运行前剥离 `expected`、`family` 等评测字段；执行网关接收工具回调和已观察事实，不接收标准答案。评分器在外部使用标准答案。

这是一套面向固定结构化政策的应用规则引擎与执行管理机制，不理解任意自然语言政策。同步本地环境的去重不提供远程 exactly-once 保证，接入真实外部写入需要服务端幂等键、版本控制与对账。独立协程上下文也不是操作系统沙箱。

## 快速开始：不调用模型

需要 Python 3.11、Git。安装脚本下载固定 commit 的 OpenJudge 源码并安装依赖，不调用模型。

```bash
git clone https://github.com/shenchen44/OpenJudge-AgentLab.git
cd OpenJudge-AgentLab
bash scripts/setup.sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_published_results.py
.venv/bin/python -m agentlab.gateway_benchmark prepare
```

本地测试为 88 项；GitHub Actions 运行同样的零模型调用测试与结果核验。公开结果验证脚本会重新评分全部 360 条记录、回放 240 条网关轨迹、复算主要配对统计，并核对冻结源码、请求日志与已返回用量；这些离线校验不会生成新的模型效果成绩。

## 显式运行付费实验

将 `.env.example` 复制为 `.env`，只在本地填写 Key。环境变量优先；`.env` 不提交。配置按字面值读取，不执行 shell 内容。

```bash
cp .env.example .env
chmod 600 .env
# 编辑 .env，填写 AGENTLAB_API_KEY。不要将 Key 写入命令或日志。

# 明确接受最多 1,800 次 HTTP 尝试后再执行：
.venv/bin/python -m agentlab.gateway_benchmark run \
  --data data_gateway_v1 \
  --out runs/gateway-ablation-v1 \
  --max-api-calls 1800
```

这是 **360 次任务执行**，不保证 1,800 次请求一定足够。请求重试也计数，达到上限就暂停，不自动扩容。估算 token 预留不是人民币硬上限。`runs/` 为本地账本和检查点，公开的历史结果放在独立 `results/`，不会被运行命令当作可用余额。

任务顺序固定打乱，三组的六种执行先后顺序均衡轮换。相同目录恢复要求冻结配置一致，已完成的失败结果不会反复重跑挑最好成绩。代码修改后使用新实验目录。

## 目录与阅读顺序

| 路径 | 内容 |
| --- | --- |
| `agentlab/execution_gateway.py` | 证据状态、写入约束、幂等和终态检查 |
| `agentlab/agent.py` | Tool Calling 执行及一次终态修复 |
| `agentlab/graders.py` | OpenJudge 状态与轨迹评分器 |
| `agentlab/coordinator.py`、`storage.py` | 调度、Attempt、检查点和请求日志 |
| `agentlab/provider.py`、`budget.py` | 模型传输、脱敏错误与持久化预算 |
| `agentlab/gateway_benchmark.py` | 相同 Prompt 的三组架构消融 |
| `data_gateway_v1/` | 新种子下 24 条开发、120 条测试任务 |
| `data_expanded_v1/` | 较早的 Prompt 对比数据 |
| `data_v2/` | 小规模预实验与旧 CLI 数据 |
| `results/` | 脱敏后的已完成实验结果与轨迹 |
| `archive/*-source/` | 按配置 SHA-256 冻结的实验源码 |
| `tests/` | 本地软件行为与 HTTP Mock 测试 |

早期 Prompt 对比同样保留：[结果](results/prompt-comparison-v1/result.json)。它不能用于声称执行网关的架构收益。旧命令和早期设计记录见 [历史说明](docs/LEGACY_README.md)。

## 与开源项目的关系

**OpenJudge**：使用官方 `BaseGrader` 接口实现业务评分器，由 `GradingRunner` 批量运行。未修改上游核心，也不声称已有上游贡献。安装时拉取 [OpenJudge](https://github.com/agentscope-ai/OpenJudge) 固定 commit `c2fdc38314cd971b03d4d9333cd2c9f24d85c27f`，源码 checkout 位于忽略提交的 `upstream/`；来源见 [UPSTREAM.json](UPSTREAM.json)。默认不用额外 LLM judge 为客观状态打分。

**DeepSeek Super-agent**：参考其任务生命周期、执行尝试与上下文隔离思想，使用 Python 实现确定性 Coordinator；没有引入 DSH 运行时、复制其源码或完整复现其多智能体架构。参考：[原型](https://github.com/deepseek-ai/dsh-superagent-prototype)、[领域模型](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/main/docs/001-domain.md)、[执行设计](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/main/docs/003-execution.md)。

本项目提供 Prompt 搜索的早期实验入口，但核心公开架构收益来自执行网关与反馈修复；不将手工 Prompt 对比包装成自动调优，也不声称进行过 SFT/RL。

## License

Apache-2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。
