# Text-to-Video Sampler & Temporal Consistency Lab

demo导向的文生视频工程实践仓库。目标：在免费或小显存 GPU 上复现并展示以下能力：
- 采样与噪声日程：Karras / 余弦 / 线性 σ 调度，DPM-Solver++ / UniPC / Heun / Euler / DDIM 对比
- CFG 分步调度与 rescale 防过曝
- 参数化与损失：ε / x0 / v 切换、SNR 加权（示意）、动态阈值
- 条件与控制：ControlNet (Depth/Edge) 与 IP-Adapter、分镜 DSL 解析
- 时序一致性：相关噪声、滑动窗口、插帧、关键帧 → Refiner 二阶段结构（接口钩子）
- 潜空间与解码：VAE 替换、tile 超分/去 banding（占位）
- 训练效率：LoRA 微调（accelerate）、混精度、梯度累积、Checkpoint
- 可复现与评测：固定种子、指标脚本 (CLIP-Text / FVD 占位 / 闪烁率 / ID 一致性占位)

## 快速开始
```bash
pip install -r requirements.txt
python inference/generate.py --config configs/infer.yaml --prompt_file eval/prompts.txt
```

## 目录说明
- `configs/`：推理与训练配置
- `sampling/`：σ 调度与 CFG 日程
- `temporal/`：时序相关噪声与滑动窗口
- `conditions/`：分镜 DSL 与控制条件入口
- `training/`：LoRA 微调骨架
- `eval/`：提示集与指标脚本
- `docs/`：技术栈与路线图文档

## A/B 示例（预期展示）
| 改动 | CLIP-Text ↑ | 闪烁率 ↓ | 时延（s/视频） | 备注 |
|------|-------------|----------|---------------|------|
| Euler 30步 + 线性σ | 0.27 | 基线 | 18 | 基线 |
| DPM++ 20步 + Karras(ρ=7) | 0.30 | -15% | 14 | 更少步更好 |
| DPM++ 20步 + CFG 衰减 | 0.31 | -20% | 14 | 过曝减少 |
| + Temporal Attention + 相关噪声 | 0.30 | -30% | 15 | 一致性提升 |

（实际数值需实验填充）

## 核心要点
1. 采样器与 σ 调度代码 → 指标表。
2.  CFG schedule/rescale 实现与前后对比帧。
3. 分镜 DSL JSON 与多镜头合成视频。
4. 时序一致性策略（相关噪声、滑窗、插帧）。
5. 种子重现与版本化配置。

## 后续扩展
- 接 FVD 计算（需要 Inception I3D 特征）
- 接真实 ControlNet/Depth 预处理
- 完整光流一致性与 ID embedding 指标

> 说明：部分模块为占位/骨架，需思考改进空间与真实实现方向.
