import torch, yaml, os
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, UniPCMultistepScheduler, EulerAncestralDiscreteScheduler, HeunDiscreteScheduler, DDIMScheduler
from sampling.sigma_schedules import sigma_schedule_karras, sigma_schedule_cosine, sigma_schedule_linear
from sampling.cfg_schedule import make_cfg_schedule
from utils.video_io import save_mp4

SCHED_MAP = {
  "dpmpp_2m": DPMSolverMultistepScheduler,
  "unipc": UniPCMultistepScheduler,
  "euler": EulerAncestralDiscreteScheduler,
  "heun": HeunDiscreteScheduler,
  "ddim": DDIMScheduler
}

def set_scheduler(pipe, cfg):
    s_type = cfg["sampler"]["type"]
    sched = SCHED_MAP[s_type].from_config(pipe.scheduler.config)
    if hasattr(sched, "use_karras_sigmas"):
        sched.use_karras_sigmas = (cfg["sampler"]["sigma_schedule"] == "karras")
    pipe.scheduler = sched

@torch.inference_mode()
def generate_one(pipe, prompt, cfg, generator):
    steps = cfg["sampler"]["steps"]
    cfg_scales = make_cfg_schedule(steps, cfg["cfg"]["start"], cfg["cfg"]["end"])
    out = pipe(prompt=prompt,
               num_inference_steps=steps,
               guidance_scale=cfg_scales[-1],
               generator=generator)
    return out.videos[0]

def main(config="configs/infer.yaml", prompt_file=None):
    cfg = yaml.safe_load(open(config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(cfg["model"]["repo_id"], torch_dtype=torch.float16).to(device)
    set_scheduler(pipe, cfg)

    generator = torch.Generator(device).manual_seed(cfg["seed"])
    prompts = [l.strip() for l in open(prompt_file or "eval/prompts.txt").read().splitlines() if l.strip()]

    os.makedirs(cfg["output_dir"], exist_ok=True)
    for i, prompt in enumerate(prompts):
        video = generate_one(pipe, prompt, cfg, generator)
        out_path = os.path.join(cfg["output_dir"], f"sample_{i:03d}.mp4")
        save_mp4(video, out_path, fps=cfg["video"]["fps"])
        print("Saved:", out_path)

if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--config", type=str, default="configs/infer.yaml")
    a.add_argument("--prompt_file", type=str, default="eval/prompts.txt")
    main(**vars(a.parse_args()))