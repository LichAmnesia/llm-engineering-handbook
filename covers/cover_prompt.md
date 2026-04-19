# 封面 Prompt（ChatGPT Images · gpt-image-1）

**用法**：粘贴进 https://chatgpt.com/images 的输入框生成。需要的话再做 minor 手动微调。

---

Create a cinematic, vintage-toned book cover image. Vertical portrait orientation, 3:4 aspect ratio (e.g. 1086 × 1448 px), for a Chinese technical book titled《构建生产级 AI 副本：LLM 工程手册》.

Visual style:
- Moody, warm, museum-lit still life. Deep blackish-green canvas backdrop, edges in heavy shadow.
- A strong diagonal beam of warm golden light falls from the upper left onto the center of the composition, bathing the props in rich amber and honey tones.
- Overall palette: near-black forest green, aged parchment cream, brass, copper, antique gold. No bright blues, no modern neon.
- Textures: visible paper grain, coffee-stained parchment, brushed metal, soft film grain.
- Shot on a medium-format camera feel, f/2.8 shallow depth of field, rim lighting on metallic objects.

Central subject: a pile of aged, slightly overlapping sheets of parchment / graph paper resting on a dark green felt desk. On the paper, rendered as if sketched in dark ink with a fountain pen:
- A hand-drawn transformer / attention-block architecture diagram (boxes labeled Q, K, V with arrows)
- A small handwritten cosine-similarity formula and a softmax equation
- A diagram of a retrieval-augmented generation (RAG) pipeline: labeled boxes "Retrieve → Rerank → Generate" with arrows
- A scatter of mathematical notations and annotations in English mixed with a few Chinese characters
- Corner doodles of small neural-network node graphs

Foreground props arranged like a vintage scholar's desk:
- A brass antique microscope on the left, partially in shadow, its eyepiece catching a sliver of golden light.
- A polished brass pocket watch with an exposed gear face, lying on the papers, chain trailing off.
- A set of interlocking brass gears and a small sculpted golden helix-like spiral (evoking a neural network topology rather than DNA) resting on the pages.
- A fountain pen, an ink bottle, a stack of hardcover leather books in the far corner, slightly out of focus.

Typography layout (very important):
- Top third of the cover reserves empty dark space for the main title, which should appear in elegant gold-colored Chinese serif type: large "构建生产级 AI 副本" on the first line, smaller "LLM 工程手册" on a second line below it.
- Between title and the still life, a thin horizontal gold rule, with a one-line subtitle in smaller gold text: "从数据采集到生产部署 · FTI 架构全流程".
- Bottom edge: author signature "LichAmnesia 著" in small gold type, centered.
- All text should feel hand-set and letterpress-like, not computer-generated. Make sure Chinese characters are rendered accurately and elegantly, no garbled glyphs.

Mood: scholarly, timeless, serious but warm. The book should look like a high-end O'Reilly-meets-Penguin-Classics hybrid, evoking both engineering craftsmanship and literary gravitas. Avoid sci-fi chrome, avoid purple/cyan neon, avoid obvious AI-generated plastic look.

Output: a single, centered, symmetrical composition suitable as a vertical book cover JPEG.

---

## 备用 variation（如果首次出图不满意）

调整以下几处再生成：

- 换主道具：把 microscope 改成 `brass vintage typewriter keys` 或 `antique mechanical calculator`，更贴合"工程"主题。
- 换光源角度：`top-right rim light` → 画面左侧更柔和。
- 强化中文字形：把 typography 段落加一句 `Use Noto Serif SC or a similar CJK serif typeface with clean stroke contrast; no AI-rendered broken characters.`
- 调比例：换成 `1536×2048 (9:12)` 或 `1024×1536 (2:3)`，看哪个导出更清晰。
