# 第 7 章 推理服务 — 实现一个真正好用的 RAG 系统

## 7.1 推理管线的职责

模型训好了，向量库建好了。推理管线要做的是把两者拼起来并暴露成一个**稳定、可观测、有 SLA** 的服务。

这章讲从请求到响应的完整链路：

```
HTTP Request
  → 输入校验
  → 查询改写 (可选)
  → 向量检索
  → 元数据过滤
  → rerank
  → 上下文压缩
  → prompt 组装
  → vLLM 生成
  → 后处理与引用抽取
  → HTTP Response
```

每一步都有失败模式和优化点。

## 7.2 向量检索

最朴素的 RAG 就是 `qdrant.search(top_k=5)`。但生产环境要处理以下情况：

- 用户用英文问关于中文语料的问题（跨语言检索）
- 用户的 query 太短，向量不稳定（`"最近工作"`）
- 用户想过滤"只看 Medium 的长文"
- 多租户场景下不能串数据

完整实现：

```python
# inference_pipeline/rag/retrieval.py
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from FlagEmbedding import BGEM3FlagModel

embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
client = QdrantClient(url="http://localhost:6333")

def retrieve(
    query: str,
    collection: str,
    top_k: int = 20,
    source: str | None = None,
    lang: str | None = None,
    tenant_id: str | None = None,
) -> list[dict]:
    must = []
    if source:
        must.append(FieldCondition(key="source", match=MatchValue(value=source)))
    if lang:
        must.append(FieldCondition(key="lang", match=MatchValue(value=lang)))
    if tenant_id:
        must.append(FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)))

    q_vec = embed_model.encode([query], max_length=512)["dense_vecs"][0].tolist()
    hits = client.search(
        collection_name=collection,
        query_vector=q_vec,
        limit=top_k,
        query_filter=Filter(must=must) if must else None,
    )
    return [{
        "text": h.payload["text"],
        "source": h.payload["source"],
        "url": h.payload["url"],
        "score": h.score,
        "id": h.id,
    } for h in hits]
```

**top_k 为什么是 20 不是 5**：我们要喂给 rerank 一个候选池，rerank 后再截到 top 5。一次检索出 5 个再 rerank 等于没 rerank。

## 7.3 Rerank

向量检索的分数是 cosine，它衡量"语义相似"，不等于"对回答问题最有用"。rerank 模型是一个**双塔交叉编码器**，把 query 和候选一起喂进去算相关性，比纯 embedding 准得多但慢 10-100x——所以只在 top-k 候选上跑。

本书用 `BAAI/bge-reranker-v2-m3`：

```python
# inference_pipeline/rag/rerank.py
from FlagEmbedding import FlagReranker

reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

def rerank(query: str, candidates: list[dict], keep_top: int = 5) -> list[dict]:
    if not candidates:
        return []
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.compute_score(pairs, normalize=True)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:keep_top]
```

效果：RAGAS 的 `context_precision` 指标通常从 0.55 提升到 0.75+。

## 7.4 上下文压缩

rerank 后的 5 个 chunk 总长可能 3000-5000 tokens。即使模型上下文窗足够，塞进去也有三个坏处：

1. **成本**：输入 tokens 越多 vLLM 吞吐越低。
2. **注意力稀释**：lost-in-the-middle 现象仍然存在。
3. **无关细节**：chunk 里有不相关的段落会干扰生成。

压缩两种做法：
- **抽句**：用一个小模型抽出 chunk 里与 query 最相关的 1-3 句。
- **提炼**：LLM 把 chunk 摘成 50 字结论。

这里用抽句（更快，不引入新的幻觉风险）：

```python
# inference_pipeline/rag/compress.py
from FlagEmbedding import BGEM3FlagModel
import re

embed = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

def compress(query: str, chunks: list[dict], max_sents_per_chunk: int = 3) -> list[dict]:
    q_vec = embed.encode([query])["dense_vecs"][0]
    out = []
    for c in chunks:
        sents = re.split(r"(?<=[。！？.!?])\s+", c["text"].strip())
        sents = [s for s in sents if len(s) > 10]
        if not sents:
            continue
        s_vecs = embed.encode(sents)["dense_vecs"]
        sims = (s_vecs @ q_vec).tolist()
        ranked = sorted(zip(sents, sims), key=lambda x: x[1], reverse=True)
        top = [s for s, _ in ranked[:max_sents_per_chunk]]
        # 恢复原文顺序
        order = {s: i for i, s in enumerate(sents)}
        top.sort(key=lambda s: order[s])
        c_new = {**c, "text": " ".join(top)}
        out.append(c_new)
    return out
```

压缩后上下文通常缩到原来的 1/3，检索质量基本不掉。

## 7.5 Prompt 组装

系统 prompt + 参考材料 + 用户问题三段式：

```python
# inference_pipeline/rag/prompt.py
SYSTEM = """你是 {author}。请用你本人的写作风格回答问题，保持你平时的节奏、用词和语气。

如果参考材料能支撑你的回答，在回答末尾用 [n] 标注引用（n 是参考材料编号）；
如果参考材料不足以回答，直接说"这点我之前没写过详细的"。"""

def compose(author: str, query: str, chunks: list[dict]) -> list[dict]:
    refs = []
    for i, c in enumerate(chunks, 1):
        refs.append(f"[{i}] ({c['source']}) {c['text']}")
    context = "\n\n".join(refs) if refs else "（无相关参考材料）"
    return [
        {"role": "system", "content": SYSTEM.format(author=author)},
        {"role": "user", "content": f"参考材料:\n{context}\n\n问题: {query}"},
    ]
```

**为什么要让模型明确引用**：
- 可观测：日志里能看到模型用了哪些 chunk。
- 可追溯：用户看到引用，点进去能读原文。
- 降低幻觉：带引用的模式让模型的生成更保守。

## 7.6 用 vLLM 做生成

vLLM 在吞吐上比 HuggingFace pipeline 高一个量级（PagedAttention + continuous batching）。单机部署：

```python
# inference_pipeline/llm/vllm_engine.py
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
    model="./out/twin-final",
    tokenizer="./out/twin-final",
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
    max_model_len=4096,
    enable_prefix_caching=True,
))

async def generate(messages: list[dict], request_id: str):
    # vLLM 0.7+ 的 AsyncLLMEngine 没有公开 .tokenizer 属性，
    # 必须 await get_tokenizer()。apply_hf_chat_template 的第 3 个参数
    # chat_template 必填（None 表示用 tokenizer 自带的）。
    tok = await engine.get_tokenizer()
    from vllm.entrypoints.chat_utils import apply_hf_chat_template
    prompt = apply_hf_chat_template(
        tok, messages, chat_template=None, add_generation_prompt=True,
    )
    sp = SamplingParams(
        temperature=0.7, top_p=0.9,
        max_tokens=800, repetition_penalty=1.05,
        stop=["<|im_end|>"],
    )
    async for out in engine.generate(prompt, sp, request_id):
        yield out.outputs[0].text
```

关键配置：
- `gpu_memory_utilization=0.85`：vLLM 会吃满这个比例的显存做 KV cache。
- `enable_prefix_caching=True`：system prompt 相同的请求共享前缀 KV，实测 qps 提升 30-50%。
- `repetition_penalty=1.05`：SFT 模型容易复读训练集，轻度惩罚。

## 7.7 FastAPI 层

把 RAG + 生成暴露成 HTTP 流式接口：

```python
# inference_pipeline/api/main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from uuid import uuid4
import json

from ..rag.retrieval import retrieve
from ..rag.rerank import rerank
from ..rag.compress import compress
from ..rag.prompt import compose
from ..llm.vllm_engine import generate

app = FastAPI()

class TwinRequest(BaseModel):
    query: str
    source: str | None = None
    lang: str | None = None
    stream: bool = True

@app.post("/twin/chat")
async def chat(req: TwinRequest):
    hits = retrieve(req.query, "twin_corpus", top_k=20,
                    source=req.source, lang=req.lang)
    hits = rerank(req.query, hits, keep_top=5)
    hits = compress(req.query, hits)
    msgs = compose(author="Shen", query=req.query, chunks=hits)

    rid = str(uuid4())
    if not req.stream:
        full = ""
        async for piece in generate(msgs, rid):
            full = piece
        return {"answer": full, "refs": [{"url": h["url"], "source": h["source"]} for h in hits]}

    async def sse():
        yield f"data: {json.dumps({'event':'refs','data':[{'url':h['url']} for h in hits]})}\n\n"
        last = ""
        async for piece in generate(msgs, rid):
            delta = piece[len(last):]
            last = piece
            if delta:
                yield f"data: {json.dumps({'event':'delta','data':delta})}\n\n"
        yield "data: {\"event\":\"done\"}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
```

## 7.8 性能目标与 SLA

单机（A10 24GB × 1）实测：

| 指标 | 目标 | 典型值 |
|------|------|--------|
| p50 first-token latency | < 400ms | 250ms |
| p95 first-token latency | < 1s | 750ms |
| 平均 tokens/sec | > 40 | 55 |
| 并发 qps（短请求） | > 8 | 12 |

慢在哪里怎么查：
- **embed + retrieval > 150ms**：检查 Qdrant 是否 co-located，网络延迟。
- **rerank > 200ms**：batch size、模型量化、是否 MPS/GPU 可用。
- **first token > 600ms**：vLLM prefix cache 有没有命中；system prompt 是否每次随机。

## 7.9 缓存策略

三层缓存：

1. **Prefix KV cache**（vLLM 自己管）：system prompt + refs 前缀复用。
2. **Embedding cache**（Redis）：同一 query 字符串的向量复用。
3. **Response cache**（Redis，可选）：完全相同 query + 相同过滤条件 的响应缓存 15 分钟。

```python
import hashlib, redis, json
r = redis.Redis()

def cache_key(query, source, lang):
    raw = f"{query}|{source}|{lang}"
    return "resp:" + hashlib.md5(raw.encode()).hexdigest()
```

注意：Twin 场景用户对"一字不差的重复回答"的容忍度比客服机器人高。但如果你的 Twin 用来做公开内容生成，response cache 要谨慎——同样的 query 生成重复内容是 SEO 灾难。

## 7.10 本章小结

- RAG 不是 `qdrant.search(5)`，是 `retrieve→filter→rerank→compress→compose→generate` 六步。
- 向量检索 top_k=20 给 rerank 留池子，rerank 截到 5。
- 用 vLLM 替代 transformers pipeline，prefix caching 要开。
- FastAPI 上做 SSE 流式，引用单独一个 event 提前吐出。
- p95 first-token < 1s 是 A10 单机的合理目标。

> **动手做**：`docker-compose up` 起 Qdrant + inference service，curl 一发请求看看端到端延迟和响应质量。Ch8 处理 naive RAG 的失败模式。
