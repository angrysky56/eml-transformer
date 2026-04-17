import torch

from eml_transformer.models.decoder import ModelConfig

checkpoint = torch.load("evaluator.pt", map_location="cpu", weights_only=False)
config = checkpoint["config"]
print(f"Evaluator Config: {config}")
if "model_state_dict" in checkpoint:
    params = sum(p.numel() for p in checkpoint["model_state_dict"].values())
    print(f"Total Parameters: {params:,}")
