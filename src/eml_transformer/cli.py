import argparse
import random
import sys

import torch
from torch.utils.data import DataLoader

from eml_transformer.data.dataset import DEPTH_IGNORE_INDEX, EffortDataset, collate_effort_batch
from eml_transformer.data.tokenizer import EMLTokenizer
from eml_transformer.data.trees import linearize, random_tree
from eml_transformer.models.decoder import ModelConfig, TinyDecoder
from eml_transformer.models.effort_head import EffortHead
from eml_transformer.training import (
    GlobalMeanBaseline,
    TokenClassBaseline,
    TrainConfig,
    evaluate,
    pretty_print_metrics,
    train,
)

DEFAULT_VARIABLES: list[str] = ["x", "y"]


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Print a handful of generated EML trees with tokens and depth labels."""
    tokenizer = EMLTokenizer.from_variables(DEFAULT_VARIABLES)
    dataset = EffortDataset(
        tokenizer=tokenizer,
        variables=DEFAULT_VARIABLES,
        num_samples=args.samples,
        max_depth=args.max_depth,
        seed=args.seed,
    )
    for i in range(args.samples):
        sample = dataset[i]
        print(f"\n--- sample {i} | tree_depth={sample.tree_depth} ---")
        print(f"  tokens:  {sample.tokens}")
        print(f"  depths:  {[d for d in sample.depth_labels if d != DEPTH_IGNORE_INDEX]}")
        print(f"  ids:     {sample.input_ids}")
        # Reconstruct the expression for human readability.
        rng = random.Random(f"{args.seed}:{i}")
        tree = random_tree(
            max_depth=args.max_depth,
            variables=DEFAULT_VARIABLES,
            rng=rng,
        )
        print(f"  expr:    {tree.to_expression()}")
    print(f"\nvocab_size = {tokenizer.vocab_size}")
    print(f"pad_id     = {tokenizer.pad_id}")
    return 0


def _cmd_baseline(args: argparse.Namespace) -> int:
    """Evaluate simple baselines on the effort-prediction task."""
    tokenizer = EMLTokenizer.from_variables(DEFAULT_VARIABLES)
    train_ds = EffortDataset(tokenizer, DEFAULT_VARIABLES, args.train_samples, seed=args.seed)
    eval_ds = EffortDataset(tokenizer, DEFAULT_VARIABLES, args.eval_samples, seed=args.seed + 1)

    train_loader = DataLoader(
        train_ds, batch_size=32, collate_fn=lambda b: collate_effort_batch(b, tokenizer.pad_id)
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=32, collate_fn=lambda b: collate_effort_batch(b, tokenizer.pad_id)
    )

    print("Fitting GlobalMeanBaseline...")
    gm = GlobalMeanBaseline.fit(train_loader)
    print(f"Global prediction: {gm.prediction}")

    print("\nFitting TokenClassBaseline...")
    tc = TokenClassBaseline.fit(train_loader, tokenizer)
    print(f"E-node prediction: {tc.e_prediction}")
    print(f"Leaf prediction:   {tc.leaf_prediction}")

    # Evaluate
    from eml_transformer.training.metrics import compute_metrics

    def eval_baseline(name, baseline):
        all_preds = []
        all_targets = []
        for batch in eval_loader:
            preds = baseline.predict(batch["input_ids"])
            all_preds.append(preds)
            all_targets.append(batch["depth_labels"])
        metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
        pretty_print_metrics(metrics, prefix=name)

    eval_baseline("GlobalMean", gm)
    eval_baseline("TokenClass", tc)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Train the tiny transformer decoder to predict EML tree depth."""
    tokenizer = EMLTokenizer.from_variables(DEFAULT_VARIABLES)
    train_ds = EffortDataset(
        tokenizer, DEFAULT_VARIABLES, args.train_samples, max_depth=args.max_depth, seed=args.seed
    )
    eval_ds = EffortDataset(
        tokenizer,
        DEFAULT_VARIABLES,
        args.eval_samples,
        max_depth=args.max_depth,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_effort_batch(b, tokenizer.pad_id),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_effort_batch(b, tokenizer.pad_id),
    )

    m_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    model = TinyDecoder(m_config)
    head = EffortHead(d_model=args.d_model, num_bins=args.max_depth + 1)

    t_config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        verbose=True,
    )

    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Starting training on {args.device}...")
    train(model, head, train_loader, eval_loader, t_config)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate a fresh (untrained) model; placeholder for checkpoint loading."""
    tokenizer = EMLTokenizer.from_variables(DEFAULT_VARIABLES)
    eval_ds = EffortDataset(
        tokenizer, DEFAULT_VARIABLES, args.eval_samples, max_depth=args.max_depth, seed=args.seed
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_effort_batch(b, tokenizer.pad_id),
    )

    m_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    model = TinyDecoder(m_config)
    head = EffortHead(d_model=args.d_model, num_bins=args.max_depth + 1)

    print(f"Evaluating model on {args.device}...")
    metrics = evaluate(model, head, eval_loader, device=args.device)
    pretty_print_metrics(metrics, prefix="Evaluation")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="eml-transformer",
        description="Effort-gated compiled substrate research prototype",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: inspect
    inspect = subparsers.add_parser(
        "inspect",
        help="Print generated EML training samples for manual verification.",
    )
    inspect.add_argument("--samples", type=int, default=5, help="Number of trees to print.")
    inspect.add_argument("--max-depth", type=int, default=5, help="Max tree depth.")
    inspect.add_argument("--seed", type=int, default=0, help="Random seed.")
    inspect.set_defaults(func=_cmd_inspect)

    # Subcommand: baseline
    baseline = subparsers.add_parser(
        "baseline",
        help="Evaluate simple baselines on the effort-prediction task.",
    )
    baseline.add_argument("--train-samples", type=int, default=1000, help="Training samples.")
    baseline.add_argument("--eval-samples", type=int, default=500, help="Evaluation samples.")
    baseline.add_argument("--seed", type=int, default=0, help="Random seed.")
    baseline.set_defaults(func=_cmd_baseline)

    # Subcommand: train
    train_parser = subparsers.add_parser(
        "train",
        help="Train the tiny transformer decoder.",
    )
    train_parser.add_argument("--train-samples", type=int, default=5000, help="Training samples.")
    train_parser.add_argument("--eval-samples", type=int, default=1000, help="Evaluation samples.")
    train_parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    train_parser.add_argument("--d-model", type=int, default=128, help="Model dimension.")
    train_parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads.")
    train_parser.add_argument("--n-layers", type=int, default=4, help="Number of layers.")
    train_parser.add_argument("--max-depth", type=int, default=5, help="Max tree depth.")
    train_parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda).")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    train_parser.set_defaults(func=_cmd_train)

    # Subcommand: eval
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate a model on the effort-prediction task.",
    )
    eval_parser.add_argument("--eval-samples", type=int, default=1000, help="Evaluation samples.")
    eval_parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    eval_parser.add_argument("--d-model", type=int, default=128, help="Model dimension.")
    eval_parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads.")
    eval_parser.add_argument("--n-layers", type=int, default=4, help="Number of layers.")
    eval_parser.add_argument("--max-depth", type=int, default=5, help="Max tree depth.")
    eval_parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda).")
    eval_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    eval_parser.set_defaults(func=_cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
