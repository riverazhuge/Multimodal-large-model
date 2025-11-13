def make_cfg_schedule(steps:int, start:float=4.0, end:float=6.5):
    if steps <= 1: return [end]
    return [start + (end-start)*i/(steps-1) for i in range(steps)]

def cfg_rescale(guided, unguided, scale:float):
    return unguided + scale * (guided - unguided)