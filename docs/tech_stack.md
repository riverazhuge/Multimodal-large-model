# 技术栈

## 核心依赖
- PyTorch 2.4+ / CUDA 12+
- diffusers / transformers / accelerate
- xFormers / FlashAttention
- peft (LoRA)
- decord / ffmpeg

## 能力点映射
| 能力 | 代码位置 | 说明 |
|------|----------|------|
| σ 调度 | sampling/sigma_schedules.py | Karras/余弦/线性 |
| 采样器切换 | inference/generate.py | DPM++ / UniPC / Heun / Euler / DDIM |
| CFG 日程 | sampling/cfg_schedule.py | 分步调度与 rescale |
| 分镜 DSL | conditions/shot_dsl.py | JSON → 场景列表 |
| 时序一致性 | temporal/* | 相关噪声 + 滑窗 |
| LoRA 微调 | training/train_lora.py | 占位骨架 |
| 评测 | eval/metrics/* | CLIP 相似度等 |

> FVD / 光流 / ID 一致性可扩展加入。