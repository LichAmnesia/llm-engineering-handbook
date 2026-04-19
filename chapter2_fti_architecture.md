# 第 2 章 系统架构 — FTI 三段式管线

## 2.1 为什么不能只写一个脚本

第一次做 LLM 应用的工程师，90% 走的路径是这样：

```
jupyter notebook → 拼 prompt → 调 API → 效果不错 → 部署成 FastAPI → 上线 → 3 个月后完全无法维护
```

问题不出在任何单一环节，出在**没有架构**。数据、训练、推理三件事的生命周期、硬件需求、更新频率完全不同，塞进一个仓库一个进程里，会在以下时刻爆掉：

- 你想换 embedding 模型 → 整个向量库要重建，但你发现重建脚本跟线上推理耦合了。
- 线上延迟突然涨 → 查日志发现是某次数据重爬卡在同一个进程里。
- 老板说"这个模型再训一版加上最近一周的数据" → 你发现训练代码依赖一个早就被改过的预处理函数。

FTI（Feature / Training / Inference）架构就是为了把这三件事**物理隔离**。

## 2.2 三条管线各自在做什么

```
┌──────────────────────────────────────────────────────────────┐
│                   FEATURE PIPELINE                           │
│   原始数据  →  清洗/chunk/embed  →  Feature Store            │
│   (crawler)    (CPU 密集)           (Qdrant + MongoDB)       │
│   跑频率：每日 / 每周                                         │
└──────────────────────────────────────────────────────────────┘
                              │ 读 feature store
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                   TRAINING PIPELINE                          │
│   指令数据集  →  SFT/DPO 微调  →  Model Registry             │
│   (from store)   (GPU 密集)        (HF Hub / S3)             │
│   跑频率：每月 / 按需                                         │
└──────────────────────────────────────────────────────────────┘
                              │ 拉 artifact
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                   INFERENCE PIPELINE                         │
│   用户请求  →  RAG + 生成  →  响应                            │
│   (FastAPI)    (GPU 常驻)       (JSON / stream)              │
│   跑频率：7x24 常驻                                           │
└──────────────────────────────────────────────────────────────┘
```

每条管线的三个关键属性：

| 属性 | Feature | Training | Inference |
|------|---------|----------|-----------|
| 触发方式 | 定时 (cron/ZenML) | 手动 / 数据漂移触发 | 请求驱动 |
| 硬件 | CPU + I/O | GPU 集群 | GPU 单机或小集群 |
| SLA | 最终一致性 OK | 无实时 SLA | p95 < 2s |
| 失败影响 | 数据延迟 | 下一版模型延迟 | 用户直接感知 |
| 版本化 | schema + pipeline code | model weights + data hash | API + prompt 模板 |

**核心洞察**：只有通过"持久化的 artifact"在三条管线之间传递状态——feature store、model registry、prompt registry——才能独立演进每一条。

## 2.3 接口契约

三条管线之间只通过以下几类 artifact 通信，别的都不行：

1. **Feature Store**（Qdrant collection + MongoDB collection）
   - schema：`{id, source, chunk_text, metadata, embedding_vector, created_at}`
   - 读者：Training pipeline（构造 SFT 数据集时用），Inference pipeline（检索时用）
   - 写者：只有 Feature pipeline

2. **Model Registry**（Hugging Face Hub 或 S3 bucket）
   - artifact：LoRA adapter 文件 + tokenizer + 训练 metadata（YAML）
   - 读者：Inference pipeline
   - 写者：只有 Training pipeline

3. **Prompt Registry**（本书用 Opik，也可用 LangSmith）
   - artifact：带版本号的 prompt 模板 + 对应的 eval 分数
   - 读者：Inference pipeline
   - 写者：开发者手动 push，CI 跑 eval 后自动 promote

> ⚠️ **反模式**：直接从 Training pipeline 的代码 import Inference pipeline 的函数。看起来省代码，实际上等你要把推理迁到另一台机器、另一个云，你会发现解耦不开。

## 2.4 目录结构

本书推荐的仓库布局（和 repo 里实际的布局一致）：

```
llm-engineering-handbook/
├── feature_pipeline/
│   ├── crawlers/              # LinkedIn, Medium, GitHub 爬虫
│   ├── etl/                   # 清洗、去重
│   ├── chunking/
│   ├── embedding/
│   └── pipeline.py            # ZenML @pipeline 入口
├── training_pipeline/
│   ├── dataset/               # 从 feature store 构造 SFT/DPO 数据集
│   ├── sft/                   # QLoRA 微调
│   ├── dpo/                   # 偏好对齐
│   └── pipeline.py
├── inference_pipeline/
│   ├── rag/                   # retrieve → rerank → compose
│   ├── llm/                   # vLLM 加载 adapter
│   ├── api/                   # FastAPI 路由
│   └── main.py
├── evaluation/                # RAGAS + 自定义 judge
├── ops/                       # k8s / modal 部署脚本
├── configs/                   # pydantic-settings 全局配置
└── scripts/
    ├── check_env.sh
    └── bootstrap_infra.sh
```

每个子目录都是一个独立 Python 包（`pyproject.toml` monorepo），避免跨管线的隐式 import。

## 2.5 全局配置

三条管线共享的参数（模型名、向量库地址、S3 bucket）放在 `configs/` 下用 Pydantic Settings 管理：

```python
# configs/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Feature store
    qdrant_url: str = "http://localhost:6333"
    mongo_uri: str = "mongodb://localhost:27017"
    corpus_name: str = "twin_corpus"

    # Model
    base_model: str = "Qwen/Qwen2.5-14B-Instruct"
    adapter_repo: str = "lichamnesia/twin-qwen2.5-14b-lora"
    embedding_model: str = "BAAI/bge-m3"

    # Infra
    s3_bucket: str = "twin-artifacts"
    hf_token: str

settings = Settings()
```

规则：**所有硬编码的路径、URL、模型名都必须通过 settings 读取**。方便本地 / CI / 生产三套环境切换，也是后面做多租户的前提。

## 2.6 触发与编排

本书用 ZenML 做管线编排（不用 Airflow 的原因：Airflow 对 ML artifact 不友好，log 和 artifact 分离得太远）。

```python
# feature_pipeline/pipeline.py
from zenml import pipeline, step

@step
def crawl(sources: list[str]) -> list[dict]: ...

@step
def clean_and_chunk(raw_docs: list[dict]) -> list[dict]: ...

@step
def embed_and_load(chunks: list[dict]) -> int:
    """写入 Qdrant；返回写入条数供监控。"""

@pipeline
def feature_etl(sources: list[str]):
    raw = crawl(sources)
    chunks = clean_and_chunk(raw)
    count = embed_and_load(chunks)
```

触发频率：
- **Feature**：每天凌晨跑一次（cron）+ 手动 on-demand。
- **Training**：手动。当 feature store 累积超过 N 条新数据，或 eval 分数漂移超过阈值时提醒。
- **Inference**：FastAPI + vLLM 常驻进程，CI/CD 推代码时蓝绿切换。

## 2.7 本地 vs 生产

开发时三条管线可以跑在一台 Mac 上（embedding 用小模型、微调用 Qwen2.5-1.5B 做 smoke test）。生产上三者分别部署：

| 管线 | 开发（Mac M3 64GB） | 生产 |
|------|---------------------|------|
| Feature | 本地 Docker 起 Qdrant + Mongo | Qdrant Cloud + MongoDB Atlas |
| Training | Qwen-1.5B LoRA 跑 MPS | Modal A100×4 / SageMaker |
| Inference | vLLM on Mac (gguf 量化) | Modal A10 常驻 |

Ch10 会讲每条线的具体部署选型和成本对比。

## 2.8 本章小结

- FTI 架构把一个 LLM 应用拆成三条独立生命周期的管线。
- 三者只通过 **Feature Store / Model Registry / Prompt Registry** 三类 artifact 通信。
- 配置统一、目录分离、触发解耦——这三件事做对了，后面每一章的工作都能并行推进。

> **动手做**：`git clone` 本仓库，看一眼 `configs/settings.py`，把 `.env.example` 复制成 `.env`，填好你自己的 HF token。这是贯穿全书的前置步骤。
