from __future__ import annotations

import torch
from torch.utils.data import Dataset
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eml_transformer.layer2.dataset import SignatureProgramPair
    from eml_transformer.layer2.tokenizer import Layer2Tokenizer


class SignatureProgramDataset(Dataset):
    """PyTorch Dataset for signature-to-program pairs."""

    def __init__(
        self,
        pairs: list[SignatureProgramPair],
        tokenizer: Layer2Tokenizer,
        max_target_length: int = 16,
    ):
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        
        self.data = []
        for p in pairs:
            # Encode without special tokens to check length against cap
            token_ids = tokenizer.encode(p.rpn, add_special=False)
            # Total length will be token_ids + BOS + EOS
            if len(token_ids) + 2 <= max_target_length:
                # Convert complex signature to 12-dim real vector
                # Convention: [real0, imag0, real1, imag1, ...]
                sig_tensor = torch.zeros(12, dtype=torch.float32)
                for i, val in enumerate(p.signature):
                    sig_tensor[2*i] = float(val.real)
                    sig_tensor[2*i+1] = float(val.imag)
                
                # Store full IDs (with BOS/EOS)
                full_ids = tokenizer.encode(p.rpn, add_special=True)
                
                self.data.append({
                    "signature": sig_tensor,
                    "ids": full_ids
                })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        return {
            "signature": item["signature"],
            "ids": torch.tensor(item["ids"], dtype=torch.long)
        }


def collate_signature_fn(pad_id: int):
    """Returns a collation function for the SignatureProgramDataset.
    
    Implements:
    - Left-padding for input_ids (causal transformer standard).
    - Right-shifted labels for teacher forcing.
    - Masking of PAD tokens in labels using -100.
    """
    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        signatures = torch.stack([item["signature"] for item in batch])
        ids_list = [item["ids"] for item in batch]
        
        # Max length of sequences in this batch
        batch_max_len = max(len(ids) for ids in ids_list)
        B = len(batch)
        
        # 1. Create a full padded tensor [B, batch_max_len]
        # We use left-padding
        full_padded = torch.full((B, batch_max_len), pad_id, dtype=torch.long)
        for i, ids in enumerate(ids_list):
            L = len(ids)
            full_padded[i, batch_max_len - L:] = ids
            
        # 2. Shift for teacher forcing
        # input_ids: [0, ..., N-1]
        # labels:    [1, ..., N]
        input_ids = full_padded[:, :-1]
        labels = full_padded[:, 1:].clone()
        
        # 3. Mask labels where input is PAD
        # This ensures we don't calculate loss for predicting BOS from PAD, etc.
        labels[input_ids == pad_id] = -100
        
        # 4. Attention mask (True for non-PAD tokens)
        attention_mask = (input_ids != pad_id)
        
        return {
            "signature": signatures,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }
        
    return collate
