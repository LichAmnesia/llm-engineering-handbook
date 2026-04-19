# 第 4 章 特征管线 — 清洗、分块、向量化

## 4.1 Feature Pipeline 做什么

Ch3 把原始数据落到 MongoDB。这章把原始数据变成**可检索、可训练**的 feature。三个动作：

```
raw_docs (MongoDB)  →  清洗  →  分块  →  向量化  →  Qdrant
                                                 ↘  MongoDB.processed_docs
```

一份 chunk 进两个库：
- **Qdrant**：向量 + 最小 payload，给 RAG 用。
- **MongoDB.processed_docs**：完整文本 + 元数据，给训练数据集构造用。

双写不是冗余，是两条下游管线对数据形态的需求不一样。

## 4.2 清洗

原始数据里永远有以下几种脏：

| 类型 | 例子 | 处理 |
|------|------|------|
| HTML 残余 | `<p>`, `&nbsp;` | `beautifulsoup4` 提纯文本 |
| Markdown 符号 | `**粗体**`, `[link](url)` | 保留 inline，去除 image 链接和 footnote |
| Emoji 与特殊字符 | 🚀, zero-width space | 保留 emoji（它们是风格的一部分），移除 ZWSP |
| URL 和 @提及 | `https://x.com/...`, `@someone` | URL 替换为 `<URL>`，@提及替换为 `<MENTION>` |
| 重复空行 | `\n\n\n\n` | 压成 `\n\n` |
| 语言判定 | 中英混排 | 用 `fasttext-langdetect`，按 char 比例打 `lang` 标签 |

核心实现：

```python
# feature_pipeline/etl/clean.py
import re
from bs4 import BeautifulSoup
from ftlangdetect import detect

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
ZWSP_RE = re.compile(r"[\u200B-\u200D\uFEFF]")

def clean_text(raw: str, source: str) -> tuple[str, str]:
    """返回 (clean_text, lang)."""
    if source == "linkedin" or source == "medium":
        raw = BeautifulSoup(raw, "html.parser").get_text()
    t = URL_RE.sub("<URL>", raw)
    t = MENTION_RE.sub("<MENTION>", t)
    t = ZWSP_RE.sub("", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()

    lang = detect(t[:500])["lang"]  # 只看前 500 字符判定
    return t, lang
```

注意**保留 emoji**。emoji 在 LinkedIn / X 语料里承载了大量语气信号（🔥 🚀 👀），清掉会让 Twin 的风格变平。

## 4.3 分块策略

分块（chunking）是 RAG 系统里被低估的环节。切错了，rerank 和 prompt engineering 都救不回来。

### 4.3.1 为什么不按固定长度切

最 naive 的做法：每 512 tokens 切一段。问题：
- 可能切断一个论点的结论句。
- 不同来源文本密度差异大（一条 tweet 就是一个完整单位，一篇 Medium 文章至少 3-5 段）。
- 不同语言的 token 密度不同（中文 1 字 ≈ 1 token，英文 1 字 ≈ 0.3 token）。

### 4.3.2 本书采用的策略：**语义 chunk + 硬上限**

```python
# feature_pipeline/chunking/semantic.py
from typing import Iterator
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def chunk_text(
    text: str,
    source: str,
    max_tokens: int = 400,
    overlap_tokens: int = 40,
) -> Iterator[str]:
    # 1) 短文本（tweet / linkedin 短帖）整条作为一个 chunk
    if len(enc.encode(text)) <= max_tokens:
        yield text
        return

    # 2) 按段落分
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    buf, buf_tokens = [], 0
    for p in paragraphs:
        ptok = len(enc.encode(p))
        if buf_tokens + ptok > max_tokens and buf:
            yield "\n\n".join(buf)
            # overlap：保留最后一段，带进下一个 chunk
            buf = buf[-1:] if buf[-1] and len(enc.encode(buf[-1])) < overlap_tokens * 3 else []
            buf_tokens = sum(len(enc.encode(b)) for b in buf)
        buf.append(p)
        buf_tokens += ptok
    if buf:
        yield "\n\n".join(buf)
```

三个关键设定：

- `max_tokens=400`：对 BGE-M3 这类 512-token embedding 模型是安全尺寸（留一点 margin 给特殊 token）。
- `overlap_tokens=40`：相邻 chunk 有约 10% 重叠，避免查询刚好命中边界而拿不到上下文。
- **短文本不切**：一条 tweet 整体进向量库，分开反而破坏语义单元。

### 4.3.3 代码 chunking

对 GitHub README / 代码块，不要按段落切——用 AST 或按 markdown heading 切。简单版：

```python
def chunk_markdown(md: str, max_tokens: int = 500) -> Iterator[str]:
    sections, current = [], []
    for line in md.splitlines():
        if line.startswith("## "):  # 顶级 H2 切分
            if current:
                sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))
    for sec in sections:
        yield from chunk_text(sec, "markdown", max_tokens=max_tokens)
```

## 4.4 向量化

### 4.4.1 选模型

2026 年中英混排场景的两个合理选择：

| 模型 | 维度 | 强项 | 何时选 |
|------|------|------|--------|
| BAAI/bge-m3 | 1024 | 多语言 + dense/sparse/multi-vec 三合一 | 本书默认 |
| intfloat/e5-mistral-7b-instruct | 4096 | 英文检索 SOTA，但模型大 | 纯英文语料 |

OpenAI 的 `text-embedding-3-large` 也能用，但需要数据出站 + 付费 + 单维度 3072，成本和合规都不友好。

### 4.4.2 批量 embed

```python
# feature_pipeline/embedding/bge.py
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

def embed_batch(texts: list[str]) -> list[list[float]]:
    out = model.encode(
        texts,
        batch_size=32,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return out["dense_vecs"].tolist()
```

`use_fp16=True` 在 Apple Silicon MPS 和 NVIDIA 上都能省一半显存，精度损失对检索任务忽略不计。

## 4.5 写入 Qdrant

Qdrant collection 初始化：

```python
# feature_pipeline/vectorstore/qdrant_ops.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def init_collection(client: QdrantClient, name: str, dim: int = 1024):
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    client.create_payload_index(name, field_name="source", field_schema="keyword")
    client.create_payload_index(name, field_name="lang", field_schema="keyword")

def upsert_chunks(
    client: QdrantClient, name: str,
    chunks: list[dict], vectors: list[list[float]],
):
    points = [
        PointStruct(
            id=c["chunk_id"],
            vector=v,
            payload={
                "text": c["text"],
                "source": c["source"],
                "url": c["url"],
                "lang": c["lang"],
                "parent_doc_id": c["parent_doc_id"],
            },
        )
        for c, v in zip(chunks, vectors)
    ]
    client.upsert(collection_name=name, points=points)
```

关键：`source` 和 `lang` 上建 payload index，后面 RAG 做 filter 时（"只检索 Medium 英文内容"）才跑得快。

## 4.6 整条管线串起来

```python
# feature_pipeline/pipeline.py
from zenml import pipeline, step
from .etl.clean import clean_text
from .chunking.semantic import chunk_text
from .embedding.bge import embed_batch
from .vectorstore.qdrant_ops import upsert_chunks, init_collection

@step
def process(raw_docs: list[dict]) -> list[dict]:
    out = []
    for d in raw_docs:
        cleaned, lang = clean_text(d["text"], d["source"])
        for i, chunk in enumerate(chunk_text(cleaned, d["source"])):
            out.append({
                "chunk_id": f"{d['_id']}::{i}",
                "parent_doc_id": d["_id"],
                "text": chunk,
                "source": d["source"],
                "url": d["url"],
                "lang": lang,
            })
    return out

@step
def vectorize_and_load(chunks: list[dict], qdrant_url: str, collection: str) -> int:
    from qdrant_client import QdrantClient
    client = QdrantClient(url=qdrant_url)
    init_collection(client, collection)
    vecs = embed_batch([c["text"] for c in chunks])
    upsert_chunks(client, collection, chunks, vecs)
    return len(chunks)

@pipeline(enable_cache=True)
def feature_etl(raw_docs: list[dict]):
    chunks = process(raw_docs)
    return vectorize_and_load(chunks, qdrant_url=..., collection=...)
```

ZenML 的 `enable_cache=True` 让相同输入不会重算——你加了 100 条新数据，只对这 100 条跑 embed，老数据缓存命中。

## 4.7 数据质量检查

入库前必过的三条硬门槛：

1. **去重**：chunk 的 MD5 hash 在当前 collection 中已存在则跳过。
2. **长度**：< 30 字符的 chunk 丢弃（embed 出来信息量太低）。
3. **语言一致性**：混杂语言比例 > 40% 的 chunk 打 `lang=mixed`，微调阶段再决定是否纳入。

```python
def quality_gate(chunks: list[dict], existing_hashes: set[str]) -> list[dict]:
    out = []
    for c in chunks:
        if len(c["text"]) < 30:
            continue
        import hashlib
        h = hashlib.md5(c["text"].encode()).hexdigest()
        if h in existing_hashes:
            continue
        c["content_hash"] = h
        out.append(c)
    return out
```

## 4.8 CDC 触发的实时更新

Ch3 铺了 MongoDB change stream 的钩子。这里接上：

```python
# feature_pipeline/cdc_worker.py
from pymongo import MongoClient
from .pipeline import feature_etl

def run():
    cli = MongoClient(...)
    col = cli["twin"]["raw_docs"]
    with col.watch([{"$match": {"operationType": "insert"}}]) as stream:
        batch = []
        for change in stream:
            batch.append(change["fullDocument"])
            if len(batch) >= 50:
                feature_etl(batch)
                batch = []
```

50 条 batch 触发一次，平衡实时性和 embedding 调用开销。

## 4.9 本章小结

- 清洗要保留 emoji，替换 URL/mention，判定语言。
- 分块按语义段落切，硬上限 400 tokens，短文本整条入库。
- Embedding 用 BGE-M3，fp16 + batch 跑。
- 双写 Qdrant（向量）和 MongoDB（完整文档），给下游两条管线各自用。
- 数据质量的三条硬门槛（去重、长度、语言）不能跳。

> **动手做**：跑 `python -m feature_pipeline.pipeline --collection twin_corpus`。完成后在 Qdrant dashboard 检查 collection 条数和 payload 分布。Ch5 会从这个 collection 出发构造 SFT 数据集。
