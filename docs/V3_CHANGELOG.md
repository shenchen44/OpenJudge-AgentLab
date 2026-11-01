# v3修订

参考DeepSeek Super-agent原型的任务、派工和独立会话设计，在现有OpenJudge评测层实现：

1. SQLite持久化任务、Attempt、请求、事件和检查点。
2. 稳定指纹去重，同结果同事务提交，不增加重复样本或反复调用。
3. ContextVar请求级用量归属，修复并发计数差值污染；执行与优化器共用原有预算。
4. 工具步骤检查点、剩余步数恢复、取消/超时/强制中断记录。
5. 确定性有限并发Worker（默认1），没有新增Router LLM请求。
6. tasks/cancel/retry命令与报告执行账本。
7. 阶段配置绑定concurrency与task-timeout；保持原有pilot闸门及显式finalize。

没有引入DSH运行时或复制其源代码，没有接入GitHub/飞书，没有自动启动真实实验。
验证结果以runs/v3-verification为准；不保证模型收益、API费用节省或远程恰好执行一次。
