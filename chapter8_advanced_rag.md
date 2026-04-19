# 第 8 章 高级 RAG — Self-query、Hybrid、Rerank

## 8.1 Naive RAG 的 7 种失败模式

Ch7 的 pipeline 在 80% 的查询上跑得不错。剩下 20% 会以可预测的方式翻车。提前识别这些失败模式比盲目加组件更有用。

| # | 失败模式 | 现象 | 根因 | 本章方案 |
|---|---------|------|------|---------|
| 1 | 关键词漏检 | 查询包含专有名词（"LangGraph"），embedding 弱匹配 | dense vector 对精确 token 不敏感 | 8.2 Hybrid Search |
| 2 | 查询太短 | `"最近工作"` 向量不稳定，返回噪声 | 短 query 的 embedding 缺乏区分度 | 8.3 Query Rewrite |
| 3 | 元数据隐式需求 | `"去年 9 月我在 LinkedIn 写过的那篇"` | 时间/来源约束在用户 query 里，没有抽出来做 filter | 8.4 Self-query |
| 4 | 错误的 chunk 边界 | 检索命中了结论前半句，没有给出理由 | chunk 切在了段落中间 | 8.5 Parent-document retrieval |
| 5 | 多跳推理 | `"我和 A 讨论过的关于 B 的想法"` | 单次检索无法同时命中 A 和 B 的关联 | 8.6 Multi-hop / Step-back |
| 6 | 长尾稀疏问题 | 某主题全语料只有 2 条相关内容，被稀释淹没 | top_k 里相关 chunk 排序不稳 | 8.7 RRF & re-ranker |
| 7 | 幻觉"参考" | 模型编造了一个看起来像引用的回答 | prompt 模板未强制 grounding | 8.8 Grounded generation |

下面逐一展开。

## 8.2 Hybrid Search（dense + sparse）

dense embedding 擅长语义匹配，但对精确 token（版本号、专有名词、缩写）弱。sparse（BM25 风格）正相反。**二者融合**才是可靠的检索。

BGE-M3 本身就同时产出 dense 和 sparse 向量：

```python
# inference_pipeline/rag/hybrid.py
from FlagEmbedding import BGEM3FlagModel
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

def hybrid_search(client, collection, query, top_k=20):
    out = model.encode([query], return_dense=True, return_sparse=True)
    dense = out["dense_vecs"][0].tolist()
    lex = out["lexical_weights"][0]  # dict: token_id(str) -> weight(float)
    sparse_vec = SparseVector(
        indices=[int(k) for k in lex.keys()],
        values=[float(v) for v in lex.values()],
    )

    results = client.query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(query=dense, using="dense", limit=50),
            Prefetch(query=sparse_vec, using="sparse", limit=50),
        ],
        query=FusionQuery(fusion=Fusion.RRF),  # Reciprocal Rank Fusion
        limit=top_k,
    )
    return [{"text": p.payload["text"], "score": p.score, "id": p.id,
             "source": p.payload["source"], "url": p.payload["url"]}
            for p in results.points]
```

初始化 collection 时配置两路 vector：

```python
from qdrant_client.models import VectorParams, SparseVectorParams, Distance

client.create_collection(
    collection_name="twin_corpus",
    vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
    sparse_vectors_config={"sparse": SparseVectorParams()},
)
```

实测效果：在 "查询包含专有名词" 的 benchmark 上，hit@5 从 0.62 提升到 0.81。

## 8.3 Query Rewrite

短 query 的 embedding 不稳定，改写成更长、更具体的表达。

最简单有效的做法是 **HyDE (Hypothetical Document Embeddings)**：让 LLM 想象一段"如果我有答案，它会是什么样"，用这段假文档的 embedding 去检索。

```python
# inference_pipeline/rag/query_rewrite.py
from anthropic import Anthropic

cli = Anthropic()

def hyde(query: str, author: str) -> str:
    msg = cli.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{"role":"user","content":f"""你是 {author}。假设你要在自己的博客里回答这个问题，写一段 100 字左右的回答（不要多，只要一段）：

{query}"""}],
    )
    return msg.content[0].text.strip()
```

用法：

```python
if len(query) < 15:
    search_query = hyde(query, author="Shen")
else:
    search_query = query
hits = retrieve(search_query, ...)
```

只对短 query 用 HyDE，长 query 用原文——HyDE 对长 query 反而可能引入噪声。

## 8.4 Self-query：从自然语言里抽 filter

用户说 `"我去年 9 月在 LinkedIn 写的那篇关于 RAG 的"`，里面藏着三条 filter：`source=linkedin`, `date~=2025-09`, `topic=RAG`。让 LLM 把它们抽出来：

```python
# inference_pipeline/rag/self_query.py
import json
from anthropic import Anthropic

cli = Anthropic()
SCHEMA = """
字段:
- source: "linkedin" | "medium" | "github" | "x"
- lang: "zh" | "en"
- date_year: int
- date_month: int
- topic_query: str (用于向量检索的改写后语义查询)
"""

def parse(query: str) -> dict:
    msg = cli.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        messages=[{"role":"user","content":f"""从用户问题里抽取检索过滤字段。只输出 JSON，不要解释。

{SCHEMA}

用户问题: {query}

输出 JSON:"""}],
    )
    try:
        return json.loads(msg.content[0].text)
    except json.JSONDecodeError:
        return {"topic_query": query}
```

然后把 filter 接到 Ch7 的 `retrieve()`：

```python
parsed = parse(user_query)
hits = retrieve(
    query=parsed.get("topic_query", user_query),
    source=parsed.get("source"),
    lang=parsed.get("lang"),
    date_year=parsed.get("date_year"),
    ...
)
```

注意：LLM 抽 filter 偶尔出错（比如把"上个月"错解成 `date_month=1`）。**始终保留 topic_query 的 fallback**：如果带 filter 检索返回 0 条，去掉 filter 重试。

## 8.5 Parent-document retrieval

检索到的 chunk 是 400 tokens 的片段，有时上下文不够。解决办法：**检索用小 chunk（召回准），生成时用大段（信息全）**。

存储时额外记一个 `parent_doc_id`（Ch4 的 schema 里已经有）。检索后把每条 chunk 映射回父文档：

```python
# inference_pipeline/rag/parent.py
from pymongo import MongoClient

mongo = MongoClient(...)

def expand_to_parent(chunks: list[dict]) -> list[dict]:
    parent_ids = list({c["parent_doc_id"] for c in chunks if c.get("parent_doc_id")})
    if not parent_ids:
        return chunks
    parents = {d["_id"]: d for d in mongo["twin"]["raw_docs"].find({"_id": {"$in": parent_ids}})}
    out, seen = [], set()
    for c in chunks:
        pid = c.get("parent_doc_id")
        if pid in seen:
            continue
        seen.add(pid)
        parent = parents.get(pid)
        if parent:
            out.append({
                **c, "text": parent["clean_text"] or parent["text"],
                "is_parent": True,
            })
        else:
            out.append(c)
    return out
```

trade-off：父文档可能很长，需要配合 8.9 的 context window budgeting。对短文本来源（X 推文、LinkedIn 帖子）不做 expansion——它们本来就短。

## 8.6 Multi-hop / Step-back

多跳问题："我和 A 讨论过的关于 B 的想法"。第一次检索可能只命中 A 或 B 之一。两种思路：

**Step-back**（Google 的论文方法）：先问一个更抽象的上位问题检索出背景，再回到原问题。

```python
def step_back_retrieve(query: str, author: str):
    step_back_q = rewrite_to_stepback(query)     # LLM 把 query 抽象化
    background = retrieve(step_back_q, top_k=5)
    specific = retrieve(query, top_k=10)
    # 去重合并
    seen, merged = set(), []
    for h in specific + background:
        if h["id"] in seen: continue
        seen.add(h["id"]); merged.append(h)
    return merged
```

**Multi-hop**：把复杂 query 拆成子问题，分别检索再合并。实现复杂，适合深度问答；Twin 场景一般用 step-back 就够。

## 8.7 RRF (Reciprocal Rank Fusion)

多路检索（hybrid、step-back、不同 source filter）的结果要合并成一个有序列表。**RRF 是最简单、最稳健的方法**：不需要对不同路的 score 做归一化。

```python
def rrf(result_lists: list[list[dict]], k: int = 60, limit: int = 20) -> list[dict]:
    scores = {}
    for results in result_lists:
        for rank, item in enumerate(results):
            scores.setdefault(item["id"], {"score": 0, "item": item})
            scores[item["id"]]["score"] += 1 / (k + rank + 1)
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [r["item"] for r in ranked[:limit]]
```

k=60 是 RRF 论文里的默认值，90% 场景不用动。

## 8.8 Grounded Generation：不允许模型胡编引用

Ch7 的 prompt 要求模型标引用 `[n]`。但你还是可能看到模型标了 `[4]`，可你只给了 3 个参考——纯粹编造。

两层防御：

1. **Prompt 级**：明确告诉模型 refs 的数量。
2. **后处理级**：正则抽出 `[n]`，超出范围的标记替换或剔除。

```python
import re

def validate_citations(text: str, num_refs: int) -> str:
    def repl(m):
        n = int(m.group(1))
        return f"[{n}]" if 1 <= n <= num_refs else ""
    return re.sub(r"\[(\d+)\]", repl, text)
```

更强硬的方案：**constrained decoding**。vLLM 支持 `guided_decoding` 参数，可以限制模型只能输出合法的 `[1]..[num_refs]`。但对中文混排效果还不稳定，Twin 场景不推荐上。

## 8.9 上下文预算

所有压缩、扩展、rerank 最终都要塞进一个固定的上下文窗口。设一个 **budget** 显式管理：

```python
# inference_pipeline/rag/budget.py
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

def pack(chunks: list[dict], budget: int = 2500) -> list[dict]:
    """按 rerank_score 降序填进 budget。"""
    used = 0
    out = []
    for c in chunks:
        ntok = len(enc.encode(c["text"]))
        if used + ntok > budget:
            continue
        used += ntok
        out.append(c)
    return out
```

4K 上下文给推理留一半（2000 tokens 给 refs、500 给 system、500 给 query + 生成）。

## 8.10 把高级 RAG 整合进 Ch7 的 pipeline

改写 `chat` handler 里的 retrieval 段：

```python
parsed = self_query.parse(req.query)
base_q = parsed.get("topic_query", req.query)
if len(base_q) < 15:
    base_q = hyde(base_q, author="Shen")

dense_hits = retrieve(base_q, "twin_corpus", top_k=30, **filter_from(parsed))
hybrid_hits = hybrid_search(client, "twin_corpus", base_q, top_k=30)
step_back = retrieve(rewrite_to_stepback(base_q), "twin_corpus", top_k=10)

merged = rrf([dense_hits, hybrid_hits, step_back], limit=20)
reranked = rerank(req.query, merged, keep_top=8)
expanded = expand_to_parent(reranked[:3]) + reranked[3:]  # top 3 扩成父文档
compressed = compress(req.query, expanded)
packed = pack(compressed, budget=2500)
```

整条 pipeline 在 A10 上端到端 < 800ms（p50）。

## 8.11 本章小结

- Naive RAG 有 7 种可预测的失败模式，不要等用户投诉再修。
- Hybrid search（dense + sparse RRF）解决关键词漏检。
- HyDE 解决短 query、Self-query 解决隐式 filter、Parent retrieval 解决上下文碎片。
- Step-back 解决多跳，RRF 做多路融合，Grounded generation 防止编造引用。
- 用显式 token budget 管理上下文，别靠 "差不多够用"。

> **动手做**：在你的 Twin 上跑一批 30 条 eval query（包含短 query、隐式 filter、多跳），对比 Ch7 naive 版和 Ch8 advanced 版的 hit@5 和人工满意度。看到显著提升再进 Ch9。
