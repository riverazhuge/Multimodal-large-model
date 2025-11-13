def prepare_control(cfg):
    # 占位：应根据启用的 depth/edge 生成控制图像（PIL 或 tensor）
    if not (cfg["control"]["depth"]["enabled"] or cfg["control"]["edge"]["enabled"]):
        return None
    return None
