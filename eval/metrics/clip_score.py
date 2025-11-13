import torch, torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

@torch.inference_mode()
def clip_text_score(frame, prompt, device="cuda"):
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    inputs = proc(text=[prompt], images=[frame], return_tensors="pt").to(device)
    out = model(**inputs)
    sim = F.normalize(out.text_embeds, dim=-1) @ F.normalize(out.image_embeds, dim=-1).T
    return sim.item()