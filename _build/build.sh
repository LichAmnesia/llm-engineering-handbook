#!/bin/bash
set -e
cd "$(dirname "$0")/.."

BUILD=_build
COVER=covers/cover_v1.png

cp "$COVER" "$BUILD/cover.png"
COVER_B64=$(base64 -i "$BUILD/cover.png")

cat > "$BUILD/book.md" <<HEADER
<div class="cover-page"><img src="data:image/png;base64,$COVER_B64" alt="cover" /></div>

<div class="cover-info-page">
<h1 class="book-title-main">构建生产级 AI 副本</h1>
<div class="subtitle">LLM 工程手册 · 从数据采集到生产部署 · FTI 架构全流程</div>
<div class="author">LichAmnesia　著</div>
<div class="date">2026 年 4 月</div>
</div>

HEADER

for f in \
  README.md \
  chapter1_why_llm_twin.md \
  chapter2_fti_architecture.md \
  chapter3_data_collection.md \
  chapter4_feature_pipeline.md \
  chapter5_sft_qlora.md \
  chapter6_dpo_alignment.md \
  chapter7_rag_inference.md \
  chapter8_advanced_rag.md \
  chapter9_evaluation_monitoring.md \
  chapter10_deployment_scaling.md \
  chapter_epilogue.md \
  references.md
do
  echo "" >> "$BUILD/book.md"
  cat "$f" >> "$BUILD/book.md"
  echo "" >> "$BUILD/book.md"
done

echo "Built $BUILD/book.md"
wc -c "$BUILD/book.md"
