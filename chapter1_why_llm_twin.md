# 第 1 章 认识 LLM Twin — 构建生产级 AI 副本

## 1.1 问题的起点

ChatGPT 能帮你写一篇 LinkedIn 帖子。但那篇帖子读起来像 ChatGPT，不像你。

这是一个工程问题，不是调 prompt 的问题。prompt 可以把通用模型往你的方向推一点，推不到底——因为模型的权重里没有你。

LLM Twin（AI 副本）的目标：把你的写作风格、领域知识、语气偏好，**编码进模型权重和检索库**，让它生成内容时默认就是你，而不是被你反复纠正。

这本书讲如何把这件事做到生产级别：可复用、可评估、可部署、可维护。

## 1.2 LLM Twin 是什么

一句话：**一个用你的个人数据微调（fine-tuning）、并由你的知识库增强检索（RAG）的语言模型系统**。

它由三层组成：

```
┌──────────────────────────────────────────────────┐
│  推理层 (Inference)                              │
│  prompt → RAG 检索 → 微调模型生成 → 后处理      │
└──────────────────────────────────────────────────┘
                      ▲
                      │
┌──────────────────────────────────────────────────┐
│  知识层 (Feature Store)                          │
│  向量库 (Qdrant) + 文档库 (MongoDB)              │
└──────────────────────────────────────────────────┘
                      ▲
                      │
┌──────────────────────────────────────────────────┐
│  数据层 (Raw Data)                               │
│  LinkedIn / Medium / X / GitHub / 私人笔记       │
└──────────────────────────────────────────────────┘
```

- **风格**来自微调：模型权重学到了你的句式、词汇偏好、段落节奏。
- **事实**来自 RAG：模型不"记"事实，事实实时从向量库里检索出来塞进上下文。

这个分工很重要。记忆 + 风格全塞进权重，会让微调变贵、更新变慢；全靠 RAG 不微调，生成内容永远带着底模的"味道"。两者各司其职才是生产级方案。

## 1.3 为什么不只用 ChatGPT

通用大模型（ChatGPT、Claude、Gemini）在三件事上打不过 Twin：

1. **风格保真度**。通用模型的默认风格是"中性客气的美国职场英文翻译中文"，prompt 工程能压住一部分，压不住全部。微调直接改权重分布，风格才稳定。
2. **领域事实**。你写过的 200 篇技术博客、5 年的会议纪要、私人笔记——通用模型没见过。RAG 能把它们实时喂进上下文，通用 API 不能。
3. **成本与延迟**。高频小任务（日报生成、自动回复）用 API 调 GPT-5.2 既慢又贵；一个 7B 微调模型跑在一块 A10 GPU 上，延迟和单位成本都更可控。

反过来：**如果你的场景是低频、通用、需要最强推理能力**（一年用十次的战略决策建议），直接用 API 就够了，别做 Twin。判断标准是 ROI，不是"够不够酷"。

## 1.4 本书覆盖的技术栈

2026 年的主流组合，本书全部沿用：

| 层 | 选型 | 说明 |
|----|------|------|
| 基础模型 | Llama 3.3 70B / Qwen2.5 14B / Mistral Small 3 | 按显存预算选，中文偏好 Qwen |
| 微调方法 | QLoRA（4-bit + LoRA adapter） | 消费级 GPU 能训 |
| 偏好对齐 | DPO（Direct Preference Optimization） | 不用奖励模型，替代 RLHF |
| Embedding | BGE-M3 / E5-mistral-7b | 多语言 + 长上下文 |
| 向量库 | Qdrant | 开源，支持混合检索 |
| 文档库 | MongoDB Atlas | 半结构化数据湖 |
| 编排 | ZenML | 管线版本化，替代 Airflow |
| 推理服务 | vLLM + FastAPI | 吞吐比 HF pipeline 高 10x |
| 部署 | Modal / AWS SageMaker / Bedrock | 按成本与合规选 |
| 可观测 | Opik / LangSmith | prompt 版本 + trace |
| 评估 | RAGAS + LLM-as-judge | 离线 eval |

注意：LangChain/LlamaIndex 本书**只在 Ch8 做高级 RAG 时**用到，不用它们做训练（它们不做训练），也不用它们做大规模推理编排（吞吐不行）。工具定位要分清。

## 1.5 一个最小可跑的例子

给你一个直觉，剩下的章节我们慢慢展开：

```python
# 推理层最简版（完整版在 Ch7）
from vllm import LLM, SamplingParams
from qdrant_client import QdrantClient

llm = LLM(model="./out/twin-final")
qdrant = QdrantClient(url="http://localhost:6333")

def twin_generate(user_query: str) -> str:
    # 1. 检索你过去写过的相关内容
    hits = qdrant.search(
        collection_name="shen_corpus",
        query_vector=embed(user_query),
        limit=5,
    )
    context = "\n\n".join(h.payload["text"] for h in hits)

    # 2. 拼 prompt，喂进微调过的模型
    prompt = f"""基于以下我过去写过的内容，用我的风格回答。

参考:
{context}

问题: {user_query}
回答:"""

    out = llm.generate(prompt, SamplingParams(temperature=0.7, max_tokens=500))
    return out[0].outputs[0].text
```

这 20 行里每一行背后都有一段决策：向量库为什么选 Qdrant 不选 Pinecone？chunk 策略怎么定？retrieval 的 top-k 为什么是 5 不是 20？微调为什么选 LoRA 不全参？我们会在对应章节讲清楚。

## 1.6 需要提前准备什么

- **硬件**：本地开发一块 24GB 显存的卡（4090/A5000/M3 Max 64GB 统一内存）能跑通全流程 demo；训练 70B 模型要云 GPU。
- **软件**：Python 3.11+、Docker、uv（替代 pip/poetry）。
- **账号**：Hugging Face（下模型）、AWS 或 Modal（跑训练/推理）、一个 OpenAI/Anthropic key（做 LLM-as-judge）。
- **数据**：你自己的。本书全程用作者的 LinkedIn + Medium + GitHub 作为 demo 数据集；读完每章你可以换成自己的。

## 1.7 三个你需要提前知道的权衡

1. **微调不是越多越好**。过拟合的 Twin 会重复训练集里的句子，读起来像鹦鹉。我们会在 Ch5 讲如何用 early stopping 和 held-out eval 卡住。
2. **RAG 不是万能**。retrieval 失败（召回率低、rerank 错、chunk 切坏）带来的"幻觉"比模型本身的幻觉更隐蔽，因为用户看到"参考文献"反而更信。Ch8 专门讲这 7 种失败模式。
3. **伦理与合规不是章节结尾的套话**。用了私人消息数据？训练数据里有他人版权内容？部署后别人能不能 prompt 出你没授权的内容？这些是上线前必须过的关，Ch9 会给 checklist。

## 1.8 本章小结

LLM Twin = **微调（风格）+ RAG（事实）+ LLMOps（可运行可维护）** 的组合。

接下来 9 章是一次完整的工程落地：从数据采集到上线运维。每章都可以独立 clone 对应代码跑起来，连起来读是一条完整路径。

> **动手做**：clone 本仓库，运行 `./scripts/check_env.sh` 确认你本地环境满足 1.6 的前置条件。
