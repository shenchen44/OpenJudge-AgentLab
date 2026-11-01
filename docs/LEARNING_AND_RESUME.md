# 项目学习与面试说明

当前项目定位：基于 OpenJudge 的 Agent 执行校验与有界反馈修复系统。先阅读根目录 README、execution_gateway.py、agent.py、graders.py，再阅读 coordinator.py 和 storage.py。

技术贡献包括：从已查询事实构造执行证据；写入前校验；终态验收；一次反馈修复；独立任务上下文；Attempt/请求归属；SQLite 检查点恢复和预算硬限制。

当前真实架构对比为同一 Prompt 下的三组 120 题实验，详见 results/gateway-ablation-v1。不能将手工 Prompt 对比说成自动调优，不能将本批 120/120 解释为线上保证，不能声称修改 OpenJudge 核心或完整复现 Super-agent。没有权重训练。

网关只理解固定的结构化政策，本地去重不构成外部 exactly-once；真实外部写入需服务端幂等和对账。实测增益、被拦截提案、失败记录和未知成本都应同时说明。
