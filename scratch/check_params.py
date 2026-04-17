import torch

from eml_transformer.data.tokenizer import EMLTokenizer
from eml_transformer.models import EffortHead, TinyDecoder, make_config


def count_params(mode, expansion, max_depth=5):
    tokenizer = EMLTokenizer.from_variables(["x", "y"])
    cfg = make_config(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        ffn_mode=mode,
        ffn_expansion=expansion,
        num_bins=max_depth + 1,
    )
    model = TinyDecoder(cfg)
    head = EffortHead(d_model=cfg.d_model, num_bins=cfg.num_bins)
    return model.num_parameters() + sum(p.numel() for p in head.parameters())


print(f"Delta (expansion 4): {count_params('delta', 4):,}")
print(f"FiLM  (expansion 4): {count_params('film', 4):,}")
for e in range(5, 12):
    print(f"FiLM  (expansion {e}): {count_params('film', e):,}")
    print(f"Vanilla (expansion {e}): {count_params('vanilla', e):,}")
