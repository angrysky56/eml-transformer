import os
import sys

import torch

# Add src to path
sys.path.append(os.path.abspath("src"))

from eml_transformer.models.decoder import ModelConfig, TinyDecoder
from eml_transformer.models.effort_head import EffortHead

# Register ModelConfig as safe global
torch.serialization.add_safe_globals([ModelConfig])


def get_params(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    if isinstance(cfg, dict):
        # Handle potential key differences
        if "self_aware" in cfg and "ffn_mode" not in cfg:
            cfg["ffn_mode"] = "delta" if cfg.pop("self_aware") else "vanilla"
        cfg = ModelConfig(**cfg)
    model = TinyDecoder(cfg)
    return sum(p.numel() for p in model.parameters())


files = ["main_v7_vanilla.pt", "main_v7_delta.pt", "main_v7_film.pt"]

for f in files:
    if not os.path.exists(f):
        print(f"{f}: Not found")
        continue
    try:
        p = get_params(f)
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
        cfg = ckpt["config"]
        if isinstance(cfg, dict):
            mode = cfg.get("ffn_mode", "unknown")
            expansion = cfg.get("ffn_expansion", "unknown")
        else:
            mode = cfg.ffn_mode
            expansion = cfg.ffn_expansion
        print(f"{f}: mode={mode}, expansion={expansion}, params={p:,}")
    except Exception as e:
        print(f"{f}: Error {e}")
