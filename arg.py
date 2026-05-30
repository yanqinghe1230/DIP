import torch

ckpt = torch.load("checkpoints/errnet_gate/errnet_latest.pt", map_location="cpu")
state = ckpt["icnn"] if isinstance(ckpt, dict) and "icnn" in ckpt else ckpt
alpha = state["gate_alpha"].item()
print("gate_alpha =", alpha)