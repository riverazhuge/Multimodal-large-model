import numpy as np

def sigma_schedule_karras(n:int, sigma_min:float, sigma_max:float, rho:float=7.0):
    ramp = np.linspace(0, 1, n, dtype=np.float32)
    return ((sigma_max**(1/rho) + ramp*(sigma_min**(1/rho) - sigma_max**(1/rho)))**rho).astype(np.float32)

def sigma_schedule_cosine(n:int, sigma_min:float=0.02, sigma_max:float=14.0, s:float=0.008):
    t = np.linspace(0, 1, n, dtype=np.float32)
    f = np.cos(((t + s)/(1+s)) * np.pi * 0.5)**2
    return (sigma_min + (sigma_max - sigma_min) * f).astype(np.float32)

def sigma_schedule_linear(n:int, sigma_min:float, sigma_max:float):
    return np.linspace(sigma_max, sigma_min, n, dtype=np.float32)