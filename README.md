# llm-engineering-handbook

**构建生产级 AI 副本（LLM Twin）的完整教程**

作者：LichAmnesia ｜ 2026

---

## 为什么有这本书？

市面上讲大语言模型（LLM）的教程要么停在 prompt engineering，要么一上来就是 Transformer 数学推导，中间那段——**如何把一个原型拼成真正可上线、可维护、可扩展的系统**——被跳过了。

这本书只讲那段。

我把自己过去两年构建"LLM Twin"（AI 副本）过程里踩过的坑、权衡过的方案、最后跑到生产的架构，压成 10 章。每章都可以独立跟着动手做，连起来读，是一套完整的 FTI（Feature–Training–Inference）管线实现。

读完你能自己搭出：一个用你本人数据微调、用 RAG 增强、可以部署到云上、能持续评估和监控的个人 AI 副本。

---

## 目录

### Part I · 地基（Why & What）

- **[第 1 章 认识 LLM Twin — 构建生产级 AI 副本](chapter1_why_llm_twin.md)**
  什么是 LLM Twin，为什么不用 ChatGPT 就够了，适用场景与边界。

- **[第 2 章 系统架构 — FTI 三段式管线](chapter2_fti_architecture.md)**
  Feature / Training / Inference 三条独立管线的分工、接口、部署边界。

### Part II · 数据（Feature Pipeline）

- **[第 3 章 数据采集 — 从社交媒体到结构化语料](chapter3_data_collection.md)**
  ETL 管线设计、LinkedIn/Medium/GitHub 爬取、MongoDB 作为数据湖。

- **[第 4 章 特征管线 — 清洗、分块、向量化](chapter4_feature_pipeline.md)**
  文本清洗策略、chunk 策略、embedding 模型选型、Qdrant 向量库、CDC 实时同步。

### Part III · 训练（Training Pipeline）

- **[第 5 章 监督微调 — 用 QLoRA 把 LLM 调成"你"](chapter5_sft_qlora.md)**
  指令数据集构造、QLoRA 原理、超参数配置、用 AWS SageMaker 跑一次完整训练。

- **[第 6 章 偏好对齐 — DPO 让模型更懂分寸](chapter6_dpo_alignment.md)**
  从 RLHF 到 DPO 的演化、偏好对数据集构造、评估对齐效果。

### Part IV · 推理（Inference Pipeline）

- **[第 7 章 推理服务 — 实现一个真正好用的 RAG 系统](chapter7_rag_inference.md)**
  retrieval → rerank → generate 的完整链路、prompt 模板、FastAPI 封装。

- **[第 8 章 高级 RAG — Self-query、Hybrid、Rerank](chapter8_advanced_rag.md)**
  naive RAG 的 7 种失败模式和对应修法、混合检索、查询改写、上下文压缩。

### Part V · 上线（LLMOps）

- **[第 9 章 评估与可观测 — 让 LLM 系统不再是黑盒](chapter9_evaluation_monitoring.md)**
  离线 eval（RAGAS/LLM-as-judge）、在线监控（延迟/成本/幻觉率）、prompt 版本管理。

- **[第 10 章 部署与扩展 — 从单机原型到多租户服务](chapter10_deployment_scaling.md)**
  Modal/SageMaker/Bedrock 选型、灰度发布、成本控制、多 tenant 隔离。

### 附录

- **[结语 · 关于"把一个人训练成模型"这件事](chapter_epilogue.md)**
- **[参考资料 · 论文 / 开源项目 / 工具链](references.md)**

---

## 怎么读这本书

| 你是 | 建议路径 |
|------|----------|
| 第一次做 LLM 项目的工程师 | 按顺序读完 10 章，跟着每章末尾的"动手做"任务实现 |
| 已经在做 RAG，想提升生产级能力 | 跳过 Ch1-Ch4，直接从 Ch7 开始，回头补 Ch5-Ch6 |
| 只想了解架构决策逻辑 | 读 Ch2、Ch9、Ch10，再挑感兴趣的章节细看 |
| 想做代码落地 | 每章末尾都有 repo 里对应目录，fork 就能跑 |

---

## 约定

- **代码**：Python 3.11+。所有示例可在 Apple Silicon / Linux 下跑。
- **术语**：技术术语首次出现时给中英文双标（如"微调（fine-tuning）"）。
- **引用**：括号里带原始论文/文档链接，不用脚注。
- **免责**：生产部署前请根据自己的业务场景二次评估；书中的默认参数是起点，不是终点。

---

## 许可证

内容：CC BY-NC 4.0。代码：MIT。转载请注明出处。
