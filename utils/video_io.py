import os
import numpy as np
import imageio
import torch

def to_uint8(frames):
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().float().cpu().numpy()
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 1) if frames.max() <= 1.0 else np.clip(frames / 255.0, 0, 1)
        frames = (frames * 255.0 + 0.5).astype(np.uint8)
    return frames

def save_mp4(frames, path, fps=12, quality=8):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arr = to_uint8(frames)
    try:
        import imageio_ffmpeg  # noqa: F401
    except Exception:
        pass
    imageio.mimwrite(path, arr, fps=fps, quality=quality, codec="libx264")
    return path
