import torch

def make_correlated_noise(T:int, H:int, W:int, rho:float=0.9, device="cuda", C:int=4):
    eps0 = torch.randn(1, C, H//8, W//8, device=device)
    noise = [eps0]
    for _ in range(1, T):
        eps = rho * noise[-1] + (1-rho**2)**0.5 * torch.randn_like(noise[-1])
        noise.append(eps)
    return torch.stack(noise, dim=0)