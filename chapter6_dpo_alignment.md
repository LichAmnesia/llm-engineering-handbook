# 第 6 章 偏好对齐 — DPO 让模型更懂分寸

## 6.1 SFT 之后为什么还要再对齐

Ch5 的 SFT 让模型学会"像你说话"。但有几类问题 SFT 不能单靠一次训练解决：

- **风格过头**：学到你写短帖子的风格后，连正式邮件也开始用"说真的"开头。
- **Tone 不分场合**：技术文档和朋友闲聊都用同一种节奏。
- **边界失守**：生成了你本人不会说的话（过激、抱怨、引战）。

这些问题的共性是：**同一个指令下，存在"更像你"和"不太像你"两种可能回答**。SFT 只教模型"输出这个"，不教它"这个比那个好"。需要偏好对齐（preference alignment）。

## 6.2 RLHF vs DPO：为什么本书选 DPO

RLHF（InstructGPT 那套）三步走：SFT → 训奖励模型 → PPO。工程复杂度高，训练不稳定，需要同时加载 4 份模型（policy、reward、value、reference）。

DPO（Direct Preference Optimization，2023）做了一个漂亮的等价变换：**不用奖励模型，直接用偏好对数据优化**。只需要两份模型（policy + frozen reference），训练稳定，代码量是 RLHF 的 1/5。

效果上，2024-2026 年大量实验显示：**数据量小（< 10k pairs）时 DPO ≥ RLHF**；数据量大且任务复杂时 RLHF 稍有优势。Twin 场景属于前者，DPO 是默认选择。

## 6.3 偏好对数据集

格式：`(prompt, chosen, rejected)`。对每个 prompt，chosen 是"更像你"的回答，rejected 是"不太像你"的回答。

### 6.3.1 三种生成方式

| 方式 | 成本 | 质量 | 适用 |
|------|------|------|------|
| A. 人工标注 | 高（几小时/千条） | 最好 | 有时间 |
| B. SFT 模型生成 N 个 → 用 judge 排序 | 中 | 中 | 推荐 |
| C. SFT 模型 vs 底模对比 | 低 | 低 | 冷启动 |

本书主推 B。流程：

```
prompt  →  SFT 模型 temperature 采样 4 次  →  LLM-as-judge 打分  →  取最高最低作为 chosen/rejected
```

### 6.3.2 实现

```python
# training_pipeline/dataset/build_dpo.py
from vllm import LLM, SamplingParams
from anthropic import Anthropic

llm = LLM(model="./out/twin-qwen2.5-14b-merged")
judge = Anthropic()

JUDGE_PROMPT = """你是 {author} 本人。评判下面两段回答，哪一段**更像你自己写的**？
评判维度：用词习惯、句式节奏、语气边界、话题切入角度。

Prompt: {prompt}

A:
{a}

B:
{b}

直接回答 "A"、"B" 或 "TIE"。理由在 <reason> 标签里写一行。"""

def build_pair(prompt: str, author: str) -> dict | None:
    sp = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=500, n=4)
    outs = llm.generate(prompt, sp)[0].outputs
    candidates = [o.text for o in outs]
    # 两两 judge，选出最偏好和最不偏好
    # 为简洁示意，这里只对比 candidate[0] 和 candidate[-1]
    import random
    a, b = random.sample(candidates, 2)
    msg = judge.messages.create(
        model="claude-opus-4-6",
        max_tokens=100,
        messages=[{"role":"user","content": JUDGE_PROMPT.format(
            author=author, prompt=prompt, a=a, b=b,
        )}],
    )
    winner = msg.content[0].text.strip().split()[0]
    if winner == "TIE":
        return None
    chosen, rejected = (a, b) if winner == "A" else (b, a)
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
```

### 6.3.3 数据集规模和来源

- **prompts 来源**：SFT 的 val 集 + 一批"边界类"新 prompt（正式邮件、公关回应、道歉、争议话题）——后者专门考验风格是否收敛。
- **规模**：1000-3000 pairs。更多对 small rank LoRA 无明显提升。
- **拒绝的 rejected 要多样**：不能全是"底模 refuse 风格"。混入一些"对，但不是你风格"、"风格对但信息错"、"风格对但长度离谱"。

### 6.3.4 常见数据陷阱

- **位置偏差**：judge 倾向选 A。随机化 A/B 顺序。
- **长度偏差**：judge 倾向选更长的回答。chosen/rejected 长度差要控制在 ±30%。
- **judge 和 SFT 用同一个底模家族** → judge 和 policy 有共谋。用不同家族的强模型做 judge（我们用 Qwen 训的 twin，用 Claude 当 judge）。

## 6.4 DPO 训练

`trl` 的 `DPOTrainer` 直接支持。

```python
# training_pipeline/dpo/train.py
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

BASE = "./out/twin-qwen2.5-14b-merged"  # SFT 后的模型作为起点

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")

lora_cfg = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05, task_type="CAUSAL_LM",
)

cfg = DPOConfig(
    output_dir="./out/twin-dpo",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    beta=0.1,                   # KL 惩罚强度，见 6.5
    max_prompt_length=1024,
    max_length=2048,
    bf16=True,
    logging_steps=10,
    eval_steps=50,
    save_steps=50,
    report_to="wandb",
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # peft 模式下 trl 自动用 adapter 禁用状态当 reference
    args=cfg,
    peft_config=lora_cfg,
    processing_class=tok,   # trl >= 0.12 改名；旧版用 tokenizer=
    train_dataset=...,      # load_dataset("json", ...) 对应 6.3.2 产出的 jsonl
    eval_dataset=...,
)
trainer.train()
trainer.save_model()
```

## 6.5 关键超参：`beta`

DPO 里 `beta` 控制**偏离 reference 模型的惩罚**：

- `beta=0.01`：几乎自由漂，风格变化大，容易忘记 SFT 学到的东西。
- `beta=0.1`（本书默认）：适度约束，偏好显著但风格不丢。
- `beta=0.3+`：过度保守，看不到明显对齐效果。

判断 `beta` 对不对的两条信号：
1. train 中 `rewards/margins` 上升但幅度合理（不应直冲 20+）。
2. eval 时 SFT 阶段学到的风格没崩（抽 10 条老 prompt 生成，肉眼看）。

## 6.6 学习率与 epochs

DPO 的学习率**比 SFT 小一个量级**（5e-6 vs 2e-4）。DPO 是在 SFT 之上的微调的微调，改动要小。

`num_train_epochs=1` 足够。DPO 训练过头会 reward hacking：模型为了让 chosen 概率高而胡乱放大它的 logit，生成质量崩盘。

## 6.7 评估

DPO 训练收敛不等于你想要的对齐真的发生了。必须做**独立 eval**：

### 6.7.1 自动 eval

从 held-out 测试集取 100 个 prompt，跑 SFT 和 SFT+DPO 两个版本各生成一次，LLM-as-judge 盲对：

```python
def compare(prompt, answer_sft, answer_dpo, author):
    msg = judge.messages.create(
        model="claude-opus-4-6", max_tokens=50,
        messages=[{"role":"user","content":f"""{author} 写的两段回答，哪段更像本人？
Prompt: {prompt}
A: {answer_sft}
B: {answer_dpo}
回答 A/B/TIE。"""}],
    )
    return msg.content[0].text.strip().split()[0]
```

**目标**：DPO 胜率 ≥ 60%。低于 55% 说明数据有问题，高于 80% 警惕 reward hacking。

### 6.7.2 人工 eval

100 条里挑 30 条最边界的（正式、敏感、长文）自己读。肉眼 eval 发现的问题往往是 judge 发现不了的——比如模型在正式场合用了你从来不用的"众所周知"。

## 6.8 安全性与红线

偏好对齐的另一个作用：**压掉一些你本人绝不会说的内容**。在数据集里加一类"safety pair"：

- prompt：一个容易诱导模型说出激进/违法/诽谤内容的问题。
- chosen：你本人会怎么 gracefully 拒绝或绕开。
- rejected：底模可能生成的不合适回答。

这部分数据量不用大（100-200 条），但训进去能显著减少后续被用户 jailbreak 的概率。

## 6.9 Merge 与交付

DPO 完了同样 merge：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("./out/twin-qwen2.5-14b-merged",
                                            torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "./out/twin-dpo")
final = model.merge_and_unload()
final.save_pretrained("./out/twin-final")
tok = AutoTokenizer.from_pretrained("./out/twin-qwen2.5-14b-merged")
tok.save_pretrained("./out/twin-final")
```

`./out/twin-final` 是接下来 Ch7 推理管线加载的 artifact。若要上云（Ch10），同步推 HF Hub：

```python
final.push_to_hub("lichamnesia/twin-qwen2.5-14b-final")
tok.push_to_hub("lichamnesia/twin-qwen2.5-14b-final")
```

## 6.10 本章小结

- DPO = 不训奖励模型的偏好对齐，适合 Twin 这种小数据场景。
- 数据构造用 SFT 采样 + LLM-as-judge 两两比较，1000-3000 pairs 够用。
- 训练关键超参：`beta=0.1`、`lr=5e-6`、`epochs=1`，更小改动更稳。
- Eval 看 judge 胜率（目标 ≥ 60%）+ 人工抽查边界 prompt。
- 加一批 safety pairs 能显著降低 jailbreak 风险。

> **动手做**：跑完 DPO 后，用 `scripts/compare_sft_dpo.py` 跑 100 条 judge 比较，确认胜率 ≥ 60%。否则回到 6.3 检查数据质量再来。
