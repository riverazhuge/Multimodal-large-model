#!/usr/bin/env bash
set -e
PROMPT="${1:-一只小狗在海边奔跑}"
TMP=tmp_prompts.txt
echo "$PROMPT" > $TMP
python inference/generate.py --config configs/infer.yaml --prompt_file $TMP
echo "Done A/B. 修改配置后再次运行以对比 outputs/ 视频。"