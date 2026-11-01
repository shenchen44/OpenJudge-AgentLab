# 历史设计与命令（非当前主页）

以下为早期项目说明，规模、状态和数字可能属于较早阶段；当前架构、真实结果与成本边界以根目录 README 及 results/gateway-ablation-v1 为准。

# OpenJudge-AgentLab v3

基于 OpenJudge 官方源码扩展的企业 Agent 评测与 Prompt 调优实验项目。

**先做预实验，再决定是否花额度优化。真实收益由数据决定，不能保证正向提升。**

## 评测设计（沿用v2）

- 数据改为12类机制、72项合成任务：预实验12、开发24、验证12、最终测试24。编号不携带场景标签。
- 增加VIP未超期、SLA等号、政策例外覆盖VIP、缺客户等级、不重复写已完成工单、读/写临时故障以及未见过的组合。
- 成功必须有真实状态、必要查询、有效澄清及无禁止副作用。写错再修复、不读工单空喊clarify，不算成功。
- 模型JSON格式/规划/轮数失败进入优化反馈；HTTP鉴权、持续网络错误等立即暂停，不继续扫描整个数据集。
- 所有模型请求（含优化器、重试）计入共享持久化预算；无全局Budget时禁止创建真实Provider。
- 阶段拆开：`pilot` → `optimize`（只冻结，不读测试）→ 显式 `finalize`。不满足预实验闸门不进入付费优化。
- 报告同时保留任务级成功率差异、回退任务、工具调用、tokens、异常及预算使用。可以报告无收益或成本收益，不自动筛掉负面结果。

## 已知边界

这是可执行的合成业务环境，不是企业生产系统。预实验和业务任务仍共享受控机制，测试只包含预先设计的规则组合/数值变化，不代表跨行业泛化。数据规模小，最终24题每题约4.17个百分点；不能据此声称小幅提升具有统计显著性。

`demo`与人工规则策略是脚本fixture，仅验证程序。离线得分不能写成大模型调优成绩。没有模型权重训练；当前只搜索系统Prompt，没有完整GEPA/OPRO实现。

## v3新增：参考Super-agent的执行管理

参考DeepSeek dsh-superagent-prototype的任务/执行尝试分离和上下文隔离思想，在Python评测层实现确定性Coordinator，不引入新的Router模型调用，也没有迁移DSH框架或复制其源码。

- EvaluationTask有稳定指纹；Attempt单独记录状态、时间和失败原因，重复投递不增加样本分母或再次执行。
- 每个Worker独立消息历史和工单环境。默认并发1，可显式设2（最高8），所有Worker及优化器共享原有预算。
- 每个HTTP尝试记录任务、Attempt、角色、状态和Token；使用ContextVar归属，不再用共享Provider计数差值统计任务用量。
- 在完成每轮工具执行后保存消息、环境状态、故障状态与步数。恢复从持久化边界继续，已完成本地工具步骤不重做。
- 中断的在途请求标为unknown，可能已收费；保留全部Attempt消耗，不用重跑后的单次成本掩盖恢复成本。
- 状态/请求/批次存入同一episodes.sqlite；结果缓存与task完成状态同事务提交。
- `tasks`可查看任务和Attempt；CLI停止后可以取消尚未完成的任务，或显式retry取消/阻塞/中断任务。已完成的模型失败不能通过retry反复重跑挑最好结果。

这是单进程、多个协程Worker的本地评测执行器，不是多租户服务或操作系统沙箱。CLI文件锁保护单一运行目录，不能把它当作跨主机分布式调度器。

## 启动

```bash
cd OpenJudge-AgentLab
source .venv/bin/activate
# 环境缺失时：bash scripts/setup.sh
agentlab plan
agentlab demo --out runs/v3-demo
open runs/v3-demo/report.html
agentlab calibrate --out runs/v3-calibration
python -m pytest -q tests
```

旧代码备份在 `archive/v1-20261101/` 和 `archive/v2-20261101/`；旧data和runs不覆盖。v3命令默认读取 `data_v2`，拒绝误用v1数据。

## DeepSeek真实预实验：只先跑这一阶段

项目会自动读取根目录 `.env` 中的 `AGENTLAB_API_KEY`，本地填写一次并保存即可。已设置的环境变量优先；`.env` 已被 Git 忽略，不应加入源码包。配置文件以字面值解析，不执行 shell 命令。

```bash
read -s "AGENTLAB_API_KEY?请输入DeepSeek官方API Key："
export AGENTLAB_API_KEY
export AGENTLAB_BASE_URL='https://api.deepseek.com'
export AGENTLAB_MODEL='deepseek-flash'
export AGENTLAB_OPTIMIZER_MODEL='deepseek-flash'

agentlab pilot --out runs/ds-pilot-v3 \
  --max-api-calls 150 --max-reserved-tokens 1000000
```

执行模型与优化器对DeepSeek官方hostname自动显式关闭thinking，保持固定比较条件。项目不读取余额、不自动充值。

预实验是同一组12项任务的baseline/manual对照，共24个episode。它不是最终测试，不涉及LLM judge。看 `runs/ds-pilot-v3/pilot.json` 的decision：

| decision | 意义 |
|---|---|
| ready | 人工方案有可观测的任务或成本收益，值得探索一轮自动优化；不是统计保证 |
| ready_efficiency | 基线接近满分，但人工方案成功率不降且Token至少少10%，可探索效率 |
| ceiling_or_no_headroom | 基线已至少11/12成功、且无足够成本空间，停止，不跑完整优化 |
| inspect_model_failures | 人工方案也有格式/截断等模型执行问题，先看轨迹修复协议或预算 |
| no_observed_gain | 目前未观察到可复现的改进空间，停止扩大投入 |

每任务会打印进度。只要接口异常或预算达到上限，阶段就停止并保存status.json。completed只表示命令完成，decision才决定是否值得继续。

## 预算说明（重要）

- `--max-api-calls`是整个阶段账本的**硬请求次数上限**，每次HTTP尝试前预占，重试也扣一次。
- `--max-reserved-tokens`是保守工程预估的Token预留上限：输入序列化UTF-8字节数+单次最大输出。它不是准确Tokenizer，也不是金额硬上限。
- 已返回的Token量、无法得知用量的失败请求、actor/optimizer分项，保存在budget.json。最终费用以官方账单为准。
- 重跑同一目录不会重置预算。提高上限是显式增加累计可用额度，不是新的一份独立额度。
- pilot有自己的目录与预算；search/finalize用同一目录与账本，二者共享累计上限。
- 单任务最多8轮模型交互、8次工具调用；actor输出上限900，optimizer2200。暂不支持思考模式比较。

## 并发与任务状态

第一轮建议保持并发1，不增加API预算。若要用2个Worker，pilot、optimize和finalize全部使用同一个 `--concurrency 2`，任务超时用 `--task-timeout 300`。并发只减少等待，既不自动省Token，也不提高任务成功率；同时在途的已预占请求可以完成，停止信号生效后不再发新请求。

```bash
agentlab tasks --out runs/ds-pilot-v3
# 先用Ctrl+C停止正在运行的CLI，再取消待执行任务：
agentlab cancel --out runs/ds-pilot-v3 --task-key '复制tasks输出中的完整key'
agentlab retry --out runs/ds-pilot-v3 --task-key '复制tasks输出中的完整key'
```

恢复使用原命令、原目录、原模型、原并发及任务预算。v2结果可以保留查看，但不能混用v3配置或缓存。v3双击 `run_pilot.command` 会使用新的ds-pilot-v3目录。

**恢复保证的边界**：当前工具是本地模拟服务，消息与本地状态可以共同恢复。请求已发出但未确认响应，或响应到达后检查点尚未提交的窗口，可能需要重发；未知请求仍可能已计费。接入真实远程写入工具之前必须增加服务端幂等键与对账，不能声称远程“恰好执行一次”。

## 闸门通过后：一轮调优，但先不碰最终测试

```bash
agentlab optimize --pilot runs/ds-pilot-v3 \
  --rounds 1 --repeats 1 --out runs/ds-experiment-v3 \
  --max-api-calls 800 --max-reserved-tokens 4000000
```

此阶段最多48个任务episode+1次优化请求（缓存可能减少），只用dev与validation。读取history.json查看候选有无被接受，frozen.json保存冻结的Prompt；没有正向收益也如实保留基线，不刻意削弱基线。

候选先按任务成功率选优，成功率不变时须轨迹不退步且tokens降低至少5%；demo没有tokens，只比较工具次数。该规则预先固定，并不保证最终测试也改善。

## 审查冻结版本后，显式运行最终测试

```bash
agentlab finalize --out runs/ds-experiment-v3 \
  --repeats 1 --max-steps 8 \
  --max-api-calls 800 --max-reserved-tokens 4000000
open runs/ds-experiment-v3/report.html
```

这一步在固定24个最终任务上比较baseline/manual/optimized共72个episode，同Prompt可复用缓存。总计search+finalize最多120个episode，加预实验为144个，少于v1的360个正式episode；这是设计规模差异，不是实际费用节省结论。

最终测试读完不再根据结果调Prompt并复用为新的独立成绩。负面或无收益结果也要保留。欲扩大实验需在开发阶段预先制定重复次数与新验收集，不要看到结果后只增加有利的一组试验。

## 中断与错误

- 预算耗尽：status.json标记paused，同参数重跑可以利用缓存；不主动提高额度。需要时你可显式提高累计上限。
- HTTP 401/403/400：立即停止，不自动重试整个任务集；日志只保留脱敏类别与状态，不保存服务端敏感错误体。
- JSON不合法、输出被截断、模型不按要求执行：作为模型失败缓存并进入优化反馈，不反复付费重跑直到碰巧成功。
- 代码/数据/模型/Prompt/预算条件变化要用新的实验目录。缓存和冻结绑定版本、工具schema、任务和请求配置。

## 代码与OpenJudge关系

保留 `upstream/` 的固定官方commit及Apache-2.0协议。使用公开BaseGrader接口新增业务评分器，实际由GradingRunner运行。自有代码在agentlab/，没有声称修改了上游核心或已有上游PR合并。

关键入口：data.py（机制和oracle）、graders.py（证据及副作用）、budget.py/provider.py（请求前额度）、experiment.py（分阶段优化和冻结）、cli.py（命令）。

`calibrate`默认的24个工程fixture是回归验证，不是独立人工标注。LLM judge不在默认路径中，避免为客观状态额外花钱。


## 设计参考

- 官方研究原型：https://github.com/deepseek-ai/dsh-superagent-prototype
- 任务与派工概念：https://github.com/deepseek-ai/dsh-superagent-prototype/blob/main/docs/001-domain.md
- 独立会话与恢复：https://github.com/deepseek-ai/dsh-superagent-prototype/blob/main/docs/003-execution.md

本项目只是借鉴并适配相关工程机制，不声称完整复现Super-agent或已获得其性能效果。最终离线验证见runs/v3-verification，真实模型指标需要你之后配置Key再测。

任务列表的completed只表示执行结束，不表示业务评分成功；同时查看reported_action、model_error与评测报告。单任务latency_ms是本次Attempt耗时，恢复任务不包含早期Attempt时长；批次耗时也是当前命令调用口径，不作为跨中断完整任务加速结论。

## 已跑完预实验后的短Prompt效率验证

如果已得到ds-pilot-v3的真实对照结果，不要改原始Prompt或重跑两组对照。双击 `run_efficiency.command`：仅评测一个固定190字符候选的12条pilot任务，沿用原目录及累计150次请求上限，不调用优化器，不读取最终测试集。

```bash
agentlab efficiency-plan --pilot runs/ds-pilot-v3 --out runs/ds-pilot-v3
agentlab efficiency --pilot runs/ds-pilot-v3 --out runs/ds-pilot-v3 --max-api-calls 150 --max-reserved-tokens 1000000
```

在当前已用98次请求的基础上，最多再用52次；如果剩余额度不足则暂停，不会自动扩容。原pilot.json及两组对照保留，结果另存efficiency.json、候选轨迹存efficiency_candidate.json。

候选是人工设计的Prompt压缩，不是自动调优结果。预先固定的探索标准为：不丢失人工对照的成功任务、没有未知用量，且相较原始Prompt总Token减少至少5%。没有满足也如实报告，不继续自动试候选。pilot已经被查看过，任何收益仍需后续独立验证，不能当最终测试成绩或人民币费用节省。


## 执行网关与有界终态修复

新模块 `execution_gateway.py` 从成功的工具响应维护工单、客户和政策证据，不接收任务标签或隐藏 world。`experiment.evaluate` 在进入 actor 前剥离 family/expected/split 等评测字段。写入前按公开的结构化业务政策校验参数、拦截缺少证据及不符合规则的操作；已满足目标时返回 NO_CHANGE，不调用后端写入。终态检查要求提交证据，最多允许一次终态反馈修复。被阻止、跳过的调用及被拒绝的原始完成声明单独记录，不能用“最终成功”掩盖危险提案。所有工具尝试（包括拒绝和跳过）计入同一个 8 次工具预算，终态修复也消耗原有 8 轮模型预算。

gateway 状态、审计轨迹、修复计数与消息/环境检查点一起保存，恢复不会重置限额。默认关闭，原有调用方式兼容；新的缓存指纹区分未启用、无终态修复、启用一次终态修复。

这是面向固定结构化政策的应用规则引擎；它没有训练模型，也不理解任意自然语言政策。同步本地环境的写入去重不构成外部系统 exactly-once 保证；外部并发需要服务端幂等键/版本控制。

`python -m agentlab.gateway_benchmark prepare`：零 API 生成并校验新种子下的 24 条开发和 120 条测试任务，检查与旧测试集的机制签名重叠。

`python -m agentlab.gateway_benchmark audit`：零 API 对历史轨迹做观测审计；结果不是网关重跑或模型修复效果。

`python -m agentlab.gateway_benchmark run --max-api-calls 1800`：需单独明确授权后才运行的付费实验。三组完全相同的 MANUAL Prompt：plain、guard_only（无额外终态反馈）、guard_repair（最多一次终态反馈）；120 条任务共 360 次执行。六种组顺序均衡交错，冻结源码/数据/配置，主要比较 guard_repair 与 plain，guard_only 用于次要消融。保存真实副作用、被拦截提案、成功率、用量开销及配对区间。超过请求上限暂停，不自动扩容。

上一轮仅 Prompt 对比的原始源码已保存在 `archive/expanded-fixed-v1-source`，可按原 config 哈希核对。其测量结果不能作为新网关的架构收益。
