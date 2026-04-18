"""Supervised training for Layer 2 signature-to-program translation."""

from __future__ import annotations

import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from eml_transformer.compiler import build_registry, load_catalog, signature_bindings, TEST_POINTS
from eml_transformer.layer2.dataset import generate_pairs, GeneratorConfig, _evaluate_tree
from eml_transformer.layer2.tokenizer import Layer2Tokenizer
from eml_transformer.layer2.torch_dataset import SignatureProgramDataset, collate_signature_fn
from eml_transformer.layer2.model import SignatureProgramModel, Layer2Config


def get_call_count(rpn: str) -> int:
    """Number of catalog calls in the RPN program."""
    # x, y, z are variables. Constants like '0', '1', 'e' are technically in catalog but arity 0.
    # In dataset.py, depth 0 is variables. depth 1 is calls.
    return sum(1 for t in rpn.split() if t not in ("x", "y", "z"))


@torch.no_grad()
def evaluate(model: SignatureProgramModel, loader: DataLoader, tokenizer: Layer2Tokenizer, device: str):
    model.eval()
    total_loss = 0
    total_exact_match = 0
    total_sig_match = 0
    total_count = 0
    
    # We'll use a registry for signature match verification
    registry = build_registry(tokenizer.catalog)
    
    for batch in loader:
        signatures = batch["signature"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # 1. Compute Loss
        logits = model(signatures, input_ids, attention_mask=attention_mask)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        total_loss += loss.item() * signatures.size(0)
        
        # 2. Greedy Decode for Exact/Signature Match
        # To save time, we only evaluate a subset or just run generate
        gen_ids = model.generate(
            signatures, 
            max_length=input_ids.size(1), 
            bos_id=tokenizer.bos_id, 
            eos_id=tokenizer.eos_id
        )
        
        for i, pred_ids in enumerate(gen_ids):
            target_rpn = tokenizer.decode(labels[i].tolist()) # Note: labels has -100 for pad, decode needs to handle it or we use input_ids
            # Cleaner: decode input_ids but strip BOS/PAD
            target_ids = input_ids[i].tolist()
            # target_ids has BOS and PAD. 
            target_rpn = tokenizer.decode([t for t in target_ids if t not in (tokenizer.pad_id, tokenizer.bos_id)])
            pred_rpn = tokenizer.decode(pred_ids)
            
            if pred_rpn == target_rpn:
                total_exact_match += 1
                total_sig_match += 1
            else:
                # Signature match check
                try:
                    # Target signature (already in batch, but let's re-eval or compare)
                    # For simplicity, we compare pred_sig to target_sig in signatures[i]
                    target_sig_complex = signatures[i].view(-1, 2)
                    target_sig = [complex(r.item(), j.item()) for r, j in target_sig_complex]
                    
                    # Compute pred sig
                    tree = tokenizer.parse_to_tree(pred_rpn)
                    if tree:
                        # Find vars in tree
                        vars_in_tree: set[str] = set()
                        def collect_vars(n):
                            from eml_transformer.compiler import TokenKind
                            if n.kind is TokenKind.VAR: vars_in_tree.add(n.var_name)
                            elif n.kind is TokenKind.EML:
                                collect_vars(n.left)
                                collect_vars(n.right)
                        collect_vars(tree)
                        vars_sorted = tuple(sorted(vars_in_tree))
                        
                        pred_sig = []
                        for p in TEST_POINTS:
                            bindings = signature_bindings(p, vars_sorted)
                            val = _evaluate_tree(tree, bindings)
                            pred_sig.append(val)
                        
                        # Compare
                        is_match = True
                        for p_val, t_val in zip(pred_sig, target_sig):
                            if abs(p_val - t_val) > 1e-8:
                                is_match = False
                                break
                        if is_match:
                            total_sig_match += 1
                except Exception:
                    pass
            
            total_count += 1

    return total_loss / total_count, total_exact_match / total_count, total_sig_match / total_count


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--depth2_ratio", type=float, default=0.0, help="Ratio of depth-2 to include in training (0.0 = pure generalization test)")
    parser.add_argument("--max_samples", type=int, default=100000, help="Cap total dataset size for speed")
    parser.add_argument("--max_depth", type=int, default=2, help="Max composition depth for generation")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Data
    print(f"Generating programs (max_depth={args.max_depth})...")
    catalog = load_catalog()
    gen_cfg = GeneratorConfig(max_composition_depth=args.max_depth)
    all_pairs = generate_pairs(gen_cfg)
    random.shuffle(all_pairs)
    
    # Split by call count
    train_pairs = []
    val_pairs = []
    
    for p in all_pairs:
        if get_call_count(p.rpn) <= 1:
            train_pairs.append(p)
        else:
            if args.depth2_ratio > 0 and random.random() < args.depth2_ratio:
                train_pairs.append(p)
            else:
                val_pairs.append(p)
                
    # Fallback if val_pairs is empty (e.g. max_depth=1)
    if not val_pairs and train_pairs:
        random.shuffle(train_pairs)
        split_idx = int(0.8 * len(train_pairs))
        val_pairs = train_pairs[split_idx:]
        train_pairs = train_pairs[:split_idx]

    # Limit val size for faster evaluation
    if len(val_pairs) > 1000:
        val_pairs = val_pairs[:1000]

    print(f"Train samples: {len(train_pairs)}")
    print(f"Val samples (depth-2): {len(val_pairs)}")

    # 2. Tokenizer & Datasets
    tokenizer = Layer2Tokenizer.from_catalog(catalog)
    train_ds = SignatureProgramDataset(train_pairs, tokenizer)
    val_ds = SignatureProgramDataset(val_pairs, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_signature_fn(tokenizer.pad_id))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_signature_fn(tokenizer.pad_id))

    # 3. Model
    model_cfg = Layer2Config(vocab_size=tokenizer.vocab_size)
    model = SignatureProgramModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    best_sig_match = -1.0
    
    # 4. Loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            signatures = batch["signature"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            logits = model(signatures, input_ids, attention_mask=attention_mask)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # Evaluation
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            val_loss, exact_match, sig_match = evaluate(model, val_loader, tokenizer, device)
            print(f"Val Loss: {val_loss:.4f} | Exact Match: {exact_match:.2%} | Sig Match: {sig_match:.2%}")
            
            if sig_match > best_sig_match:
                best_sig_match = sig_match
                torch.save(model.state_dict(), "layer2_model_best.pt")
                print("Saved new best model.")

    print(f"Training complete. Best Sig Match: {best_sig_match:.2%}")


if __name__ == "__main__":
    train()
