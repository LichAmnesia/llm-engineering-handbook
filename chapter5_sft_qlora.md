# 第 5 章 监督微调 — 用 QLoRA 把 LLM 调成"你"

## 5.1 微调要解决什么

前面四章都在数据侧。这一章进入模型侧。

微调（SFT, Supervised Fine-Tuning）要解决的是 **RAG 解决不了的问题**：模型生成时默认的**风格、用词、段落结构**。检索得再准，底模还是 Qwen 官方 instruct 的味道。微调改的是模型权重里那部分"怎么说话"的分布。

我们不从零训，也不做全参微调。用 QLoRA：4-bit 量化底模 + 训练一个小的 LoRA adapter。消费级 GPU（24GB 显存）能训 Qwen2.5-14B。

## 5.2 QLoRA 原理（能用得上的那部分）

不展开数学。你需要记住三件事：

1. **4-bit NF4 量化底模**：把原始 fp16 权重压到 4-bit NormalFloat，显存降 4x，训练时 dequant 一小块做计算，再丢回 4-bit 存储。
2. **LoRA adapter**：在每个 attention 和 MLP 投影矩阵旁边，挂一个低秩（rank=16 或 32）矩阵对。训练时只更新这些低秩矩阵，原始权重冻住。
3. **最终 merge**：训练完可以把 LoRA 矩阵 merge 回底模，推理时和全参微调一样快。

参数量对比（Qwen2.5-14B）：

| 方案 | 可训参数 | 单卡显存峰值 |
|------|---------|-------------|
| 全参微调 | 14B | ~120GB（多卡） |
| LoRA r=16 | ~14M (0.1%) | ~28GB |
| QLoRA r=16 | ~14M (0.1%) | ~16GB ✅ |

## 5.3 构造 SFT 数据集

底模是 chat 模型（Qwen2.5-14B-Instruct），数据必须是 `{instruction, input, output}` 或 chat messages 格式。原始语料是单向的帖子/文章，要**反向构造成指令对**。

### 5.3.1 反向指令生成

思路：用一个强模型（GPT-5.2 / Claude Opus 4.6）读你的某条帖子，反推"什么 prompt 会让人写出这段"。

```python
# training_pipeline/dataset/build_sft.py
from anthropic import Anthropic

cli = Anthropic()

REVERSE_PROMPT = """你将看到一段 {author} 在 {source} 上发表的文字。
请反推：什么样的用户 prompt 会让 {author} 写出这段回复？

要求：
1. 指令要具体（不能是"写一段话"），要带话题和场景。
2. 指令不能提到原文里出现过的独特措辞。
3. 用中文写指令。

原文：
{text}

直接输出指令，不要解释。"""

def reverse_instruction(text: str, author: str, source: str) -> str:
    msg = cli.messages.create(
        model="claude-opus-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": REVERSE_PROMPT.format(
            author=author, source=source, text=text,
        )}],
    )
    return msg.content[0].text.strip()
```

对每条 chunk 调一次，生成 `(instruction, original_text)` 对作为 SFT 样本。

### 5.3.2 数据集质量审计

反向生成有两个典型失败：
- **指令泄露原文独特措辞**（"写一段关于 ... 的帖子，用'坦白讲'开头"）→ 用 LLM-as-judge 再审一遍。
- **指令过于通用**（"谈谈你对 AI 的看法"）→ 按指令长度 + n-gram 多样性过滤。

审计代码：

```python
def audit(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    kept = []
    for instr, out in pairs:
        if len(instr) < 15:              # 太短
            continue
        if any(w in instr for w in ["写一段", "谈谈你对"]):  # 模板太通用
            continue
        # 指令 5-gram 不能在 output 里出现
        instr_5grams = {instr[i:i+5] for i in range(len(instr)-4)}
        if any(g in out for g in instr_5grams if len(g) >= 5):
            continue
        kept.append((instr, out))
    return kept
```

审完保留率通常 60-75%。

### 5.3.3 数据量与切分

经验法则：
- **最少 500 条高质量 pair** 才能明显感觉到风格迁移。
- **2000-5000 条**是 sweet spot，更多性价比快速下降。
- Train / val / test 按 90 / 5 / 5 切。val 用于 early stopping，test 永远不看直到最终 eval。

保存格式（chat messages，配 Qwen chat template）：

```python
import json
def to_chatml(pairs, system_prompt, out_path):
    with open(out_path, "w") as f:
        for instr, resp in pairs:
            sample = {"messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instr},
                {"role": "assistant", "content": resp},
            ]}
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
```

system prompt 建议：
```
你是 {用户姓名}。请用你本人的写作风格回答问题，保持你平时的节奏、用词和语气。
```

## 5.4 训练配置

用 `trl` 的 `SFTTrainer` + `peft` + `bitsandbytes`：

```python
# training_pipeline/sft/train.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import torch

BASE = "Qwen/Qwen2.5-14B-Instruct"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=bnb,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)

ds = load_dataset("json", data_files={
    "train": "data/sft_train.jsonl",
    "validation": "data/sft_val.jsonl",
})

cfg = SFTConfig(
    output_dir="./out/twin-qwen2.5-14b-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps", eval_steps=100,
    save_strategy="steps", save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    max_seq_length=2048,
    packing=True,
    report_to="wandb",
)

trainer = SFTTrainer(
    model=model, args=cfg, peft_config=lora_cfg,
    train_dataset=ds["train"], eval_dataset=ds["validation"],
    processing_class=tokenizer,   # trl >= 0.12 改名，旧版是 tokenizer=
)
trainer.train()
trainer.save_model()
```

## 5.5 关键超参数的经验值

| 参数 | 经验值 | 为什么 |
|------|--------|--------|
| `r` (LoRA rank) | 16 | 8 风格学不进，32+ 开始过拟合；14B 模型 16 是甜点 |
| `lora_alpha` | 2×r = 32 | 约定俗成，别乱动 |
| `learning_rate` | 2e-4 | QLoRA 比全参高一个量级；1e-4 偏保守，5e-4 训崩 |
| `epochs` | 3 | 超过 3 轮 val loss 开始反弹 |
| `batch_size × grad_accum` | effective 16 | 太小 loss 抖，太大 update 不够细 |
| `max_seq_length` | 2048 | 覆盖 99% 样本；长样本 packing 提效率 |
| `warmup_ratio` | 0.03 | 3% step 线性 warmup，防止初期乱跳 |

## 5.6 监控训练

两个必看指标：

1. **train_loss / val_loss**：val_loss 应该先降后平。连续两个 eval point 反弹就该 early stop。
2. **sample 生成**：每 100 step 从 val 里抽 3 条 instruction 跑一次生成，肉眼看风格有没有漂过来。

肉眼 eval 的脚本：

```python
# training_pipeline/sft/sample_during_train.py
from transformers.trainer_callback import TrainerCallback

class SampleCallback(TrainerCallback):
    def __init__(self, eval_prompts, tokenizer):
        self.eval_prompts = eval_prompts
        self.tok = tokenizer

    def on_evaluate(self, args, state, control, model=None, **kw):
        model.eval()
        for p in self.eval_prompts:
            inputs = self.tok.apply_chat_template(
                [{"role":"user","content":p}],
                return_tensors="pt", add_generation_prompt=True,
            ).to(model.device)
            out = model.generate(inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
            print(f"\n[step {state.global_step}] {p}\n{self.tok.decode(out[0][inputs.shape[1]:])}")
```

## 5.7 Merge & Push

训完 merge LoRA 回底模，推到 HF Hub：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "./out/twin-qwen2.5-14b-lora")
merged = model.merge_and_unload()
merged.save_pretrained("./out/twin-qwen2.5-14b-merged")
tok = AutoTokenizer.from_pretrained(BASE)
tok.save_pretrained("./out/twin-qwen2.5-14b-merged")

# 推送
merged.push_to_hub("lichamnesia/twin-qwen2.5-14b-merged")
tok.push_to_hub("lichamnesia/twin-qwen2.5-14b-merged")
```

**只推送 adapter 不推送 merged**：省空间，加载时 `PeftModel.from_pretrained` 自己拼。但 vLLM 对 adapter-only 支持不如 merged 稳，生产部署我们推 merged。

## 5.8 在云上跑

本地 4090 跑一次 3-epoch 训练大概 4-6 小时。云上 A100 或者 Modal：

```python
# training_pipeline/sft/modal_runner.py
import modal

image = modal.Image.debian_slim().pip_install(
    "transformers==4.47", "trl==0.12", "peft==0.13",
    "bitsandbytes", "accelerate", "datasets", "flash-attn",
).apt_install("git")

app = modal.App("twin-sft")

@app.function(
    gpu="A100-80GB",
    image=image,
    timeout=60*60*6,
    volumes={"/cache": modal.Volume.from_name("hf-cache", create_if_missing=True)},
    secrets=[modal.Secret.from_name("hf-token"), modal.Secret.from_name("wandb")],
)
def train():
    from training_pipeline.sft.train import main
    main()

@app.local_entrypoint()
def go():
    train.remote()
```

Modal A100-80GB 按秒计费，6 小时大约 $24，比买显卡划算。

## 5.9 常见翻车

- **数据里混了指令泄露** → 模型学到 "看到 xxx 就抄原文"，eval 看着漂亮上线翻车。用 5.3.2 的 audit。
- **忘记 `prepare_model_for_kbit_training`** → 训练 loss 不降。
- **`attn_implementation="flash_attention_2"` 忘开** → 训练慢 2-3x。
- **eval_loss 和 train_loss 同向下降且差距很大** → val 集有重复 / 污染，重新切。

## 5.10 本章小结

- QLoRA = 4-bit 底模 + LoRA adapter，消费级 GPU 能训 14B。
- 数据集用反向指令生成 + LLM audit，500-5000 条高质量 pair。
- 关键超参有经验值，别瞎调。
- 训练过程跑 sample callback，肉眼监控风格是否在收敛。
- 上云跑 A100，6 小时训完 Qwen-14B，$24 级别。

> **动手做**：跑 `python -m training_pipeline.sft.train`，确认 val_loss 3 epoch 下降且 sample 有风格感知。adapter 推到 HF Hub。Ch6 在此基础上做偏好对齐。
