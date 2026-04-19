# 第 10 章 部署与扩展 — 从单机原型到多租户服务

## 10.1 这章要做的决策

前九章在一台机器上跑通了完整链路。这章做三件上线前必须做的决策：

1. **部署平台选哪个**：Modal / AWS SageMaker / Bedrock / 自己搭 k8s。
2. **扩展模式**：单租户 vs 多租户，垂直扩展 vs 水平扩展。
3. **成本控制**：GPU 空转、缓存策略、冷热分层。

每个决策都给判断标准，不是单推某一个方案。

## 10.2 部署平台对比

按"你是谁"选：

| 场景 | 推荐 | 理由 |
|------|------|------|
| 独立开发者，按需用，不想管运维 | **Modal** | 按秒计费，scale-to-zero，部署代码就是一段 Python decorator |
| 已在 AWS 生态，企业合规要求 | **SageMaker Endpoint** | IAM、VPC、KMS 齐全 |
| 已在 AWS 且想用托管 API（不自管模型） | **Bedrock** + custom import | 管 prompt 和 eval 不管 infra |
| 有 DevOps 团队、量大要精细化控制 | **自托管 k8s + KServe** | 总成本最低但要能 hold 住 |
| 个人试验，不想付云钱 | **Ollama + 反代** | 本地跑，代价是没 SLA |

本章分别给出前两种的最小可用部署代码。

## 10.3 Modal 部署

Modal 适合 Twin 的 90% 场景。一份文件起一个生产级服务：

```python
# ops/modal/deploy_twin.py
import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        "vllm==0.7", "fastapi", "uvicorn",
        "qdrant-client", "FlagEmbedding", "anthropic", "opik",
    )
    .env({"HF_HOME": "/cache/huggingface"})
)

app = modal.App("twin-inference")

MODEL_VOL = modal.Volume.from_name("twin-model", create_if_missing=True)

@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/cache": MODEL_VOL},
    secrets=[
        modal.Secret.from_name("qdrant"),
        modal.Secret.from_name("anthropic"),
        modal.Secret.from_name("opik"),
    ],
    min_containers=1,             # Modal 1.4+：最少 1 个常驻（旧 API 叫 keep_warm）
    scaledown_window=300,         # 5 分钟无请求后缩容
)
@modal.concurrent(max_inputs=10)  # 单实例吃 10 个并发（1.4+ 独立装饰器，旧 API 写在 @app.cls 里）
class Twin:
    @modal.enter()
    def load(self):
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model="lichamnesia/twin-qwen2.5-14b-final",
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            enable_prefix_caching=True,
            max_model_len=4096,
        ))
        # embed + qdrant 客户端初始化略

    @modal.fastapi_endpoint(method="POST", docs=True)  # 1.4+；旧 API 叫 web_endpoint
    async def chat(self, req: dict):
        # 这里调 Ch7/Ch8 的 pipeline
        ...
```

部署：`modal deploy ops/modal/deploy_twin.py`。5 分钟后拿到一个 HTTPS 端点。

配置要点：
- `min_containers=1`：Twin 是面向终端用户的服务，冷启延迟（A10 拉模型 ~40s）不能让用户感知。保 1 个常驻。
- `scaledown_window=300`：流量低谷自动缩回 1 个，省钱。
- `@modal.concurrent(max_inputs=10)`：vLLM 的 continuous batching 能吃 10 个并发不降 p95。

成本估算：1 个常驻 A10G @ $1.10/h × 720h = **$790/月**。突发流量多启一个 A10 按分钟计费。

## 10.4 SageMaker Endpoint 部署

合规要求高的团队走 SageMaker。流程比 Modal 繁，但 VPC/IAM/加密都现成：

```python
# ops/sagemaker/deploy.py
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

role = "arn:aws:iam::123:role/SageMakerExecutionRole"

hf_model = HuggingFaceModel(
    model_data="s3://twin-artifacts/twin-qwen2.5-14b-final.tar.gz",
    role=role,
    image_uri=sagemaker.image_uris.retrieve(
        framework="djl-lmi", region="us-east-1",  # DJL LMI 内置 vLLM backend
        version="0.29.0", image_scope="inference",
    ),
    env={
        "HF_MODEL_ID": "/opt/ml/model",
        "OPTION_DTYPE": "bf16",
        "OPTION_ROLLING_BATCH": "vllm",
        "OPTION_MAX_MODEL_LEN": "4096",
    },
)

predictor = hf_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge",   # A10G
    endpoint_name="twin-prod",
    container_startup_health_check_timeout=600,
)
```

`ml.g5.2xlarge` 按小时计费 ~$1.52/h，比 Modal 贵 38%，但合规收益值得。

Auto-scaling：

```python
import boto3
cli = boto3.client("application-autoscaling")
cli.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/twin-prod/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=1, MaxCapacity=4,
)
cli.put_scaling_policy(
    PolicyName="twin-tgt",
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/twin-prod/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 40.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
        },
        "ScaleInCooldown": 300, "ScaleOutCooldown": 60,
    },
)
```

## 10.5 多租户

如果 Twin 是 SaaS 产品（不是个人玩具），多租户是必考题。三层隔离：

### 10.5.1 数据隔离

Qdrant collection 的两种方式：

- **共享 collection + payload filter**（tenant_id）——省运维，中等隔离
- **每 tenant 一个 collection**——强隔离，但 collection 数量过千时 Qdrant 管理变难

本书推荐 **10 人以上大客户独立 collection，其他走共享 + filter**。

### 10.5.2 模型隔离

LoRA adapter 的优势在这里兑现：底模共享，每个 tenant 一个 adapter。vLLM 原生支持多 adapter 热切：

```python
from vllm import AsyncEngineArgs
engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
    model="Qwen/Qwen2.5-14B-Instruct",
    enable_lora=True,
    max_loras=8,
    max_lora_rank=32,
))

# 请求时带上 LoRA id
request_output = engine.generate(
    prompt, sampling_params, request_id,
    lora_request=LoRARequest(
        lora_name=f"twin-{tenant_id}",
        lora_int_id=hash(tenant_id) % 2**31,
        lora_path=f"s3://twin-adapters/{tenant_id}/",
    ),
)
```

一块 A100 能同时 hold 8-16 个 adapter。tenant 规模 < 100 时一个 cluster 够用。

### 10.5.3 成本隔离

每 request 记 tenant_id + tokens，按 token + GPU 分钟做 billing。Grafana 按 tenant 拆面板，超阈值告警。

## 10.6 蓝绿部署 / 灰度发布

模型升级不能直接替换线上版本。两种标准方案：

**蓝绿**：两个完全一样的环境并存，DNS / LB 切流量。回滚是改 DNS 一步。

**灰度**（推荐）：同一个 endpoint 按比例分流新老版本：

```python
# 在 API 层用 tenant_id 或 random 决定版本
@app.post("/twin/chat")
async def chat(req, x_user_id: str = Header(...)):
    version = "v2" if hash(x_user_id) % 100 < 10 else "v1"   # 10% 灰度
    return await engines[version].generate(...)
```

灰度比例：**1% → 10% → 50% → 100%**，每档观察 24h。同时看 style judge / faithfulness / 用户反馈三个指标。任何一个劣化 > 5% 立刻回滚。

## 10.7 成本优化

### 10.7.1 GPU 空转是最大的钱坑

scale-to-zero 不总是可行（冷启 40s）。折中：

- **低峰期缩容到 1**：Modal 的 `scaledown_window` 或 SageMaker auto-scaling 的 min=1。
- **off-peak 时间用 CPU 降级方案**：夜里把请求转给 Claude Haiku API + 给 grounding prompt 加 "模仿以下风格" 一段。延迟和成本都 OK，风格打 7 折。

### 10.7.2 量化

线上用 INT8 / FP8 推理，吞吐 +30-50%，风格 eval 分数掉 < 0.1：

```python
engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
    model="lichamnesia/twin-qwen2.5-14b-final",
    quantization="fp8",  # 需要 Hopper (H100) 或 Ada Lovelace (L40)
    ...
))
```

A10 上没有 FP8，用 AWQ：

```bash
pip install autoawq
# 离线量化一次
python -m awq.quantize lichamnesia/twin-qwen2.5-14b-final \
    --output ./out/twin-awq-int4
```

然后 vLLM 加载 `--quantization awq`。

### 10.7.3 Embedding 和 Rerank 下沉到 CPU 或小 GPU

Embedding 和 rerank 不需要大 GPU。单独起一个 t3.xlarge CPU 服务跑 BGE-M3 + reranker-v2-m3，给 inference 服务调。成本从 GPU 侧省下来。

### 10.7.4 Response cache

Ch7 讲过。对于公开 Twin（比如博客的 "问我任何事"），相同 query 缓存 1 小时能省 30-50% 的生成成本。

## 10.8 灾备

最小化：

- 模型 artifact 同步到两个区域的 S3 bucket。
- Qdrant 用 snapshot 每日备份。
- Prompt registry 和 eval set 在 git 里。
- 本地 Ollama + gguf 版的 Twin 作为"掉电预案"——真当 Modal 挂了，起本地。

## 10.9 上线前 72 小时 checklist

- [ ] 从干净机器 `git clone` 能跑通全链路（docs + scripts 覆盖到）。
- [ ] 离线 eval 在生产镜像里跑通。
- [ ] 压测：500 QPS × 30 分钟，p95 < 2s，零 5xx。
- [ ] prompt injection 测试集（50 条）过一遍，无越权输出。
- [ ] 回滚脚本跑通（把 prod 版本改回前一版 < 2 分钟）。
- [ ] alerts 配齐：error rate > 1%、p95 > 3s、drift > 10%、每日成本超预算。
- [ ] 人工读 100 条 real traffic 生成（dry-run 灰度流量）。

## 10.10 本章小结

- Modal 是大多数独立开发者的最优解；企业合规选 SageMaker。
- 多租户隔离靠 Qdrant collection/filter + vLLM LoRA adapter 热切。
- 灰度永远 1% → 10% → 50% → 100%，三档指标不劣化才全量。
- 成本优化的顺序：scale-to-low → 量化 → embedding 下沉 → response cache。
- 上线前 72 小时过完 checklist，别靠"感觉没问题"。

> **动手做**：`modal deploy ops/modal/deploy_twin.py`，跑 `scripts/load_test.py --qps 100 --duration 1800`。看监控面板 24h，确认 p95、成本、错误率全绿后，把真实流量放 1% 上来。
