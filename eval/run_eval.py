import os, yaml, csv, time
import numpy as np
from PIL import Image
from eval.metrics.clip_score import clip_text_score
from eval.flicker_rate import flicker_rate
from utils.video_io import save_mp4

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, UniPCMultistepScheduler, EulerAncestralDiscreteScheduler, HeunDiscreteScheduler, DDIMScheduler
from sampling.cfg_schedule import make_cfg_schedule

SCHED_MAP = {
  "dpmpp_2m": DPMSolverMultistepScheduler,
  "unipc": UniPCMultistepScheduler,
  "euler": EulerAncestralDiscreteScheduler,
  "heun": HeunDiscreteScheduler,
  "ddim": DDIMScheduler
}

def set_scheduler(pipe, s_type, use_karras):
    sched = SCHED_MAP[s_type].from_config(pipe.scheduler.config)
    if hasattr(sched, "use_karras_sigmas"):
        sched.use_karras_sigmas = use_karras
    pipe.scheduler = sched

@torch.inference_mode()
def gen_once(pipe, prompt, steps, guidance_scale, seed=1234):
    g = torch.Generator(pipe.device).manual_seed(seed)
    out = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance_scale, generator=g)
    return out.videos[0]

def main(config="configs/eval.yaml", infer_cfg="configs/infer.yaml"):
    ecfg = yaml.safe_load(open(config))
    icfg = yaml.safe_load(open(infer_cfg))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(icfg["model"]["repo_id"], torch_dtype=torch.float16).to(device)

    prompts = [l.strip() for l in open(ecfg["eval"]["prompts_file"]).read().splitlines() if l.strip()]
    seed = ecfg["eval"]["seed"]
    out_csv = ecfg["eval"]["output_csv"]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    rows = []
    for setting in ecfg["eval"]["sampler_set"]:
        s_type = setting["type"]
        steps = int(setting["steps"])
        use_karras = (setting.get("sigma_schedule", "karras") == "karras")
        set_scheduler(pipe, s_type, use_karras)
        cfg_scales = make_cfg_schedule(steps, 4.0, 6.5)
        gs = cfg_scales[-1]

        for i, prompt in enumerate(prompts):
            t0 = time.time()
            vid = gen_once(pipe, prompt, steps, gs, seed + i)
            elapsed = time.time() - t0
            mid = vid.shape[0] // 2
            frame = (vid[mid] * 255).astype(np.uint8)
            clip_t = clip_text_score(Image.fromarray(frame), prompt, device=device)
            flick = flicker_rate((vid * 255).astype(np.uint8))
            out_path = f"outputs/eval_{s_type}_{steps}_{i:03d}.mp4"
            save_mp4(vid, out_path, fps=12)
            rows.append({
                "sampler": s_type, "steps": steps, "prompt_id": i,
                "clip_text": clip_t, "flicker": flick, "latency_sec": elapsed,
                "video": out_path
            })
            print(f"[{{s_type}}/{{steps}}] #{{i}} CLIP={{clip_t:.3f}} Flicker={{flick:.1f}} Time={{elapsed:.2f}}s")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved eval to:", out_csv)

if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--config", type=str, default="configs/eval.yaml")
    a.add_argument("--infer_cfg", type=str, default="configs/infer.yaml")
    main(**vars(a.parse_args()))