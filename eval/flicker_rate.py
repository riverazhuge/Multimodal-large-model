import numpy as np

def flicker_rate(frames):
    """
    简易“闪烁率”估计：相邻帧差的全局均方差均值，数值越大闪烁越明显。
    frames: np.ndarray [T,H,W,C] (uint8 or float[0..1])
    """
    x = frames.astype(np.float32)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 1)
        x = (x * 255.0).astype(np.uint8)
    diffs = []
    for t in range(1, x.shape[0]):
        d = x[t].astype(np.int16) - x[t-1].astype(np.int16)
        diffs.append((d**2).mean())
    return float(np.mean(diffs)) if diffs else 0.0