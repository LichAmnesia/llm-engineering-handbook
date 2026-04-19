# 第 9 章 评估与可观测 — 让 LLM 系统不再是黑盒

## 9.1 LLM 系统的可观测难在哪

传统 Web 服务的可观测是成熟的：latency、error rate、资源占用。LLM 服务多出三件难事：

1. **正确性没有确定性 ground truth**。同一个 query，昨天回答得好今天回答得差，不一定是 bug。
2. **故障是渐进的**。向量库漂移 + prompt 微调 + 底模升级，每一项单独看都没问题，组合起来质量降 10%。
3. **成本与质量 trade-off 是 runtime 决策**。要不要换更便宜的 reranker？单看 metric 不够，要看 dollar/质量曲线。

这章讲怎么把这三件难事变成可工程化的东西：**离线 eval（CI 跑）+ 在线 metric（持续跑）+ prompt / dataset 版本化**。

## 9.2 离线 eval：每次改动前跑

核心 metric 分三组。

### 9.2.1 检索质量（RAGAS + 自己扩展）

[RAGAS](https://github.com/explodinggradients/ragas) 定义了一套 reference-free 的 metric，Twin 场景主要看三个：

| metric | 衡量 | 目标 |
|--------|------|------|
| `context_precision` | 检索到的 chunk 里跟问题相关的比例 | > 0.75 |
| `context_recall` | 相关 chunk 里被检索到的比例（需要 ground truth chunks） | > 0.80 |
| `faithfulness` | 回答中声称的事实能否在 context 里找到支撑 | > 0.90 |

```python
# evaluation/offline/ragas_eval.py
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness
from datasets import Dataset

def run_eval(samples: list[dict]):
    """samples: [{'question','ground_truth','answer','contexts':[str,...]}, ...]"""
    ds = Dataset.from_list(samples)
    result = evaluate(ds, metrics=[context_precision, context_recall, faithfulness])
    return result
```

没有 human-labeled ground truth 的情况下，`context_recall` 跑不了。冷启动建议手工标 50-100 条 eval 集，以后每季度扩充 30 条。

### 9.2.2 风格保真度（LLM-as-judge）

Twin 的核心价值在风格，RAGAS 管不到。加一个自定义 judge：

```python
# evaluation/offline/style_judge.py
from anthropic import Anthropic
cli = Anthropic()

PROMPT = """下面是一批 {author} 真实写过的参考文本，和一段 AI 生成的回答。
请按 1-5 打分，判断 AI 回答的**风格**与参考文本的匹配度（不考虑事实准确性）。

参考文本（3 段）：
{refs}

AI 回答：
{answer}

评分标准：
1 = 完全不像 {author}（生硬、官方、翻译腔）
3 = 部分像，词汇或节奏某一方面接近
5 = 几乎难以和参考分辨

只回答 1-5 的整数。"""

def judge(answer: str, refs: list[str], author: str) -> int:
    msg = cli.messages.create(
        model="claude-opus-4-6", max_tokens=5,
        messages=[{"role":"user","content":PROMPT.format(
            author=author, refs="\n\n---\n\n".join(refs), answer=answer,
        )}],
    )
    try:
        return int(msg.content[0].text.strip()[0])
    except (ValueError, IndexError):
        return 3
```

refs 从语料里随机抽 3 段。SFT 之后 Twin 风格分应稳定在 **4.0 以上**。

### 9.2.3 生成质量与安全

| metric | 衡量 | 工具 |
|--------|------|------|
| answer_relevance | 回答和问题相关 | RAGAS |
| hallucination_rate | 无 context 支撑的事实比例 | faithfulness 反指标 |
| toxicity | 冒犯/有害内容 | `detoxify` |
| refusal_rate | 合理 prompt 被 refuse 的比例 | 自己写 classifier |

### 9.2.4 组成 eval set

```
data/eval/
├── retrieval_set.jsonl      # 50 条带 gt 的检索评估
├── style_set.jsonl          # 100 条风格评估 prompt
├── safety_set.jsonl         # 30 条边界 prompt
├── regression_set.jsonl     # 历史上出过 bug 的 prompt
```

每次改动（改 prompt / 换模型 / 调 rerank 参数）前，全套跑一遍。任何 metric 退步超过 **5%** 直接 block。

## 9.3 CI 集成

把 eval 接进 GitHub Actions / 自己的 CI：

```yaml
# .github/workflows/eval.yml
name: eval
on: pull_request
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -e ".[eval]"
      - env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          QDRANT_URL: ${{ secrets.QDRANT_URL }}
        run: python -m evaluation.offline.run --fail-threshold 0.05
      - uses: actions/upload-artifact@v4
        with: { name: eval-report, path: eval_report.html }
```

关键是产出 **HTML diff 报告**：当前 branch vs main 每条 metric、每条样本具体回答对比。PR reviewer 直接看。

## 9.4 在线可观测

离线 eval 是快照，线上行为是电影。持续跑的三层 metric：

### 9.4.1 基础 metric（Prometheus / CloudWatch）

- `twin_request_total{route,status}` — QPS / 错误率
- `twin_latency_seconds{phase}` — phase ∈ {retrieve, rerank, generate}，各段 latency 分别埋点
- `twin_tokens_total{kind}` — prompt vs completion tokens
- `twin_cost_usd_total` — 累计成本（按模型单价折算）

### 9.4.2 LLM 特有 metric

- `twin_empty_retrieval_total` — 检索返回空的请求数（每天超过阈值要告警）
- `twin_citation_coverage` — 回答里引用标注的 chunk 数 / 提供的 chunk 数（越高模型越 grounded）
- `twin_refusal_rate` — "这点我之前没写过详细的" 这类拒答的比例

### 9.4.3 Trace（Opik / LangSmith）

每一次用户请求落一条 trace，包含：

```json
{
  "request_id": "...",
  "query": "...",
  "retrieval": [{"id":"...","score":..., "selected":true}, ...],
  "rerank": {...},
  "prompt_template_version": "v1.3",
  "model_version": "twin-qwen2.5-14b-dpo-v2",
  "generation": {"text":"...","tokens":..., "duration_ms":...},
  "user_feedback": null
}
```

本书用 Opik（开源、自托管、对接 vLLM 顺手）：

```python
# inference_pipeline/observability/opik_tracer.py
from opik import Opik
cli = Opik()

def trace_request(req, retrieval, rerank, gen_out, prompt_version):
    cli.log_trace(
        name="twin_chat",
        input={"query": req.query},
        output={"text": gen_out},
        metadata={
            "retrieval_ids": [h["id"] for h in retrieval],
            "rerank_top5": [h["id"] for h in rerank[:5]],
            "prompt_version": prompt_version,
        },
    )
```

### 9.4.4 用户反馈闭环

API 层暴露一个 `feedback` 端点：

```python
@app.post("/twin/feedback")
def feedback(request_id: str, thumb: int, note: str | None = None):
    # thumb ∈ {-1, 1}
    cli.log_feedback(request_id=request_id, score=thumb, note=note)
```

👎 的请求单独落一个 bucket，每周人工审 20 条，找 eval set 里没覆盖的失败模式，扩充 `regression_set.jsonl`。

## 9.5 Prompt 版本化

prompt 是代码，不是 magic string。三条规矩：

1. 所有 prompt 模板进 `prompts/*.jinja` 文件，CI 跟代码一起走。
2. 每个模板带 frontmatter metadata：`version`, `author`, `eval_score_at_commit`。
3. 线上部署通过 prompt registry（Opik 有内置）读 `prompt:chat:production` 这个别名，版本切换一键回滚。

```yaml
# prompts/chat_rag.jinja
---
version: 1.3
updated: 2026-04-10
eval_scores:
  style: 4.2
  faithfulness: 0.92
---
你是 {{ author }}。...
```

## 9.6 Dataset 版本化

eval set、SFT dataset 都用 DVC 或者 `huggingface_hub` 版本化（不要塞 git）。数据变更触发模型重训，重训的 metadata 里记下 dataset hash，出问题能溯源。

## 9.7 成本看板

Twin 在线成本分三块：

| 项 | 单价（参考） | 占比 |
|----|-------------|------|
| vLLM GPU 常驻 | Modal A10 $1.10/h | 60% |
| Embedding 自托管 | 可忽略（CPU） | 5% |
| Judge / HyDE 用 Claude Haiku | $1/M in, $5/M out | 15% |
| Qdrant Cloud | $100/mo（小集群） | 20% |

做一个 Grafana panel：`cost_per_1k_requests`。一旦某天单位成本异常，立刻查 trace 看是谁在刷 HyDE。

## 9.8 Drift 检测

每周自动跑：

1. 抽样 200 条当周请求，跑 style judge，和上周对比。
2. 检索到的 chunk 的 source 分布漂移（比如 Medium 突然占比从 30% 涨到 60%，可能是某个新文章的 embedding 异常高分）。
3. top-1 retrieval score 的分布偏移。

Drift > 10% 触发告警，人工审。

## 9.9 合规 checklist

上线前过一遍：

- [ ] 训练数据里的 PII（电话、身份证、住址）已 scrub
- [ ] 第三方 API key 不在提交历史中
- [ ] 用户输入做长度上限（防 DoS）和 prompt injection 的基础过滤
- [ ] 输出过 toxicity classifier，分数超阈值的响应打 `flagged=true` 并记录
- [ ] Twin 被公开使用时，最终输出带 "AI 生成" 水印（法律要求）
- [ ] 用户数据存储有明确的保留期和删除流程

## 9.10 本章小结

- 离线 eval = RAGAS + 自定义风格 judge + safety set，CI 强制。
- 在线观测 = Prometheus 基础指标 + Opik trace + 用户反馈闭环。
- prompt 和 dataset 必须和代码一样版本化。
- drift 和成本是每周必看的看板。
- 合规不是章节脚注，是上线门禁。

> **动手做**：在 PR 上跑 `python -m evaluation.offline.run --head` vs main，拿到 diff 报告。线上部署后跑一周，看看 👎 反馈里有哪些模式你 eval set 没覆盖。
