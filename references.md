# 参考资料

## 论文

### Transformer & LLM 基础
- Vaswani et al., *Attention Is All You Need* (2017). [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Brown et al., *Language Models are Few-Shot Learners* (GPT-3, 2020). [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- Qwen Team, *Qwen2.5 Technical Report* (2024). [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)

### PEFT / QLoRA
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021). [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023). [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

### 偏好对齐
- Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT, 2022). [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Rafailov et al., *Direct Preference Optimization* (DPO, 2023). [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

### RAG
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP* (2020). [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE, 2022). [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)
- Zheng et al., *Take a Step Back: Evoking Reasoning via Abstraction* (2023). [arXiv:2310.06117](https://arxiv.org/abs/2310.06117)

### Embedding & Rerank
- Chen et al., *BGE M3-Embedding* (2024). [arXiv:2402.03216](https://arxiv.org/abs/2402.03216)
- Wang et al., *E5: Text Embeddings by Weakly-Supervised Contrastive Pre-training* (2022). [arXiv:2212.03533](https://arxiv.org/abs/2212.03533)

### 推理优化
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM, 2023). [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- Dao et al., *FlashAttention-2* (2023). [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

## 开源项目

### 核心依赖
- [vllm-project/vllm](https://github.com/vllm-project/vllm) — 本书推理服务基石
- [huggingface/trl](https://github.com/huggingface/trl) — SFT / DPO Trainer
- [huggingface/peft](https://github.com/huggingface/peft) — LoRA 实现
- [TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — 4-bit 量化
- [qdrant/qdrant](https://github.com/qdrant/qdrant) — 向量库
- [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) — BGE 系列 embedding & rerank

### 编排 / 可观测
- [zenml-io/zenml](https://github.com/zenml-io/zenml) — MLOps 管线
- [comet-ml/opik](https://github.com/comet-ml/opik) — LLM 可观测
- [explodinggradients/ragas](https://github.com/explodinggradients/ragas) — RAG 评估

### 参考实现
- [decodingml/llm-twin-course](https://github.com/decodingml/llm-twin-course) — Paul Iusztin 的 LLM Twin 课程，本书架构深受启发
- [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — 一站式 SFT/DPO 训练框架

## 工具链
- Modal — [modal.com](https://modal.com/) 按秒计费的 serverless GPU
- AWS SageMaker — [aws.amazon.com/sagemaker](https://aws.amazon.com/sagemaker/)
- Hugging Face Hub — 模型和数据集托管
- Weights & Biases — 训练过程追踪

## 推荐阅读顺序（如果你想补课）

如果你从零开始，按这个顺序读论文比按时间顺序读效率高：

1. *Attention Is All You Need* — 建立 Transformer 直觉
2. *LoRA* — 理解为什么不用全参微调
3. *QLoRA* — 理解 4-bit 量化怎么和 LoRA 结合
4. *DPO* — 理解为什么不用 RLHF
5. *RAG (Lewis 2020)* — 原始 RAG 定义
6. *BGE M3* — 现代 embedding 模型的多功能合一
7. *vLLM paper* — PagedAttention 为什么重要
8. *HyDE* + *Step-back* — 高级 RAG 两个关键技巧

## 致谢

本书在写作过程中参考了 Paul Iusztin 的 LLM Twin 课程、Decodingml 团队的博客、以及无数开源社区的讨论。所有错误归作者本人。
