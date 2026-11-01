# v3任务执行层

## 角色和归属

Coordinator是普通Python代码，不使用LLM。它将固定的评测输入注册为EvaluationTask，创建Attempt并分发给有限协程Worker。Worker拥有独立的消息列表、TicketEnvironment和ContextVar RequestScope。OpenJudge评分角色及Prompt优化器职责不变。

SQLite执行表：execution_tasks、attempts、checkpoints、requests、execution_events、batches、batch_tasks。episodes结果缓存保留。稳定key包括engine、工具schema、完整任务、Prompt、model/endpoint/request_options、重复编号及步骤预算；并发是实验条件，冻结配置绑定它。

状态：queued -> running -> completed；遇到预算或协作停止为interrupted；网络/鉴权/Worker超时为blocked。取消仅允许未运行且未完成的任务。重跑命令可恢复blocked/interrupted，显式retry可以重新排队取消的任务；completed模型失败是有价值的负面结果，不能重试直到碰巧成功。

## 请求和预算

发送前先预占共享Budget，再记录请求归属。每次HTTP尝试（包含重试）有独立id，携带task_key、task_id、attempt_id和role，不保存API Key、请求Authorization或服务端原始错误体。优化器归属对应candidate编号；任务用量从请求记录累计，不依赖共享Provider计数器差值。

并发在同一事件循环内，不同时运行多个CLI写同一目录；file lock保护CLI。全局预算耗尽或基础设施失败时发协作停止信号，允许已经在途的请求结算，但阻止新请求；不通过自动增加额度恢复。

取消或进程崩溃时，不知道供应商是否已经计费，标记unknown/未知用量。成功恢复后的episode包含同任务所有已记录Attempt的请求用量，不能只汇报最后一次尝试的较低成本。优化器及actor均在总预算中记录。

## 检查点与恢复

在一轮模型响应及其所有本地工具调用执行完毕后，同步保存消息、步数、工具事件、工单状态、剩余注入故障。新Attempt加载检查点，恢复独立环境和消息，沿用剩余步骤预算。terminal结果缓存与任务完成同一SQLite事务提交。

人工Ctrl+C会停止Worker并记录interrupted；强制kill来不及处理时，下次注册该任务将陈旧running Attempt和started Request标为interrupted/unknown。不会把它们抹掉。

这只保证本地持久化边界的恢复。模型响应已到达但快照尚未保存，或者请求仍在途的窗口，可能重复请求。真实外部工具写入需服务端幂等/对账，当前并不提供外部exactly-once保证。独立协程会话不是OS沙箱。

## 量化与证据

单任务：必要证据、业务成功、工具事件、所有已记录Attempt的tokens、unknown请求数。批次：提交数、去重数、缓存命中、实际Worker峰值、墙钟时间。缓存重播不能声称是推理加速。

tests/test_execution.py用HTTP MockTransport验证并发归属、跨任务上下文隔离、硬预算、401后续步恢复、取消在途请求、超时及陈旧Attempt恢复；没有真实模型调用。它可以证明程序性质，不能证明DeepSeek效果或线上加速。
