import argparse
import random
import sys

import torch
from torch.utils.data import DataLoader

from eml_transformer.data.dataset import (
    DEPTH_IGNORE_INDEX,
    EffortDataset,
    collate_effort_batch,
)
from eml_transformer.data.tokenizer import EMLTokenizer
from eml_transformer.data.trees import random_tree
from eml_transformer.models import (
    EffortHead,
    EMLTransformer,
    LMHead,
    ModelConfig,
    TinyDecoder,
)
from eml_transformer.training import (
    GlobalMeanBaseline,
    TokenClassBaseline,
    TrainConfig,
    evaluate,
    evaluate_lm,
    load_checkpoint,
    pretty_print_metrics,
    save_checkpoint,
    train,
    train_lm,
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
        print(
            f"  depths:  {[d for d in sample.depth_labels if d != DEPTH_IGNORE_INDEX]}"
        )
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
    train_ds = EffortDataset(
        tokenizer, DEFAULT_VARIABLES, args.train_samples, seed=args.seed
    )
    eval_ds = EffortDataset(
        tokenizer, DEFAULT_VARIABLES, args.eval_samples, seed=args.seed + 1
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        collate_fn=lambda b: collate_effort_batch(b, tokenizer.pad_id),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=32,
        collate_fn=lambda b: collate_effort_batch(b, tokenizer.pad_id),
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

    def eval_baseline(name: str, baseline) -> None:
        all_preds = []
        all_targets = []
        for batch in eval_loader:
            preds = baseline.predict(batch["input_ids"])
            # Flatten and move to CPU to ensure concatenation is robust to
            # varying sequence lengths and device mismatches.
            all_preds.append(preds.view(-1).cpu())
            all_targets.append(batch["depth_labels"].view(-1).cpu())

        if not all_preds:
            print(f"No samples found for {name} evaluation.")
            return

        preds_tensor = torch.cat(all_preds, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(preds_tensor, targets_tensor)
        pretty_print_metrics(metrics, prefix=name)

    eval_baseline("GlobalMean", gm)
    eval_baseline("TokenClass", tc)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Train the tiny transformer decoder to predict EML tree depth."""
    tokenizer = EMLTokenizer.from_variables(DEFAULT_VARIABLES)
    train_ds = EffortDataset(
        tokenizer,
        DEFAULT_VARIABLES,
        args.train_samples,
        max_depth=args.max_depth,
        seed=args.seed,
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
        num_bins=args.max_depth + 1,
    )
    model = TinyDecoder(m_config)
    head = EffortHead(d_model=args.d_model, num_bins=m_config.num_bins)

    if args.load_path:
        print(f"Loading checkpoint from {args.load_path}...")
        load_checkpoint(model, head, args.load_path, device=args.device)

    t_config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        verbose=True,
    )

    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Starting training on {args.device}...")
    train(model, head, train_loader, eval_loader, t_config)

    if args.save_path:
        print(f"Saving checkpoint to {args.save_path}...")
        save_checkpoint(model, head, m_config, args.save_path)

    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate a trained model from a checkpoint."""
    tokenizer = EMLTokenizer.from_variables(DEFAULT_VARIABLES)
    eval_ds = EffortDataset(
        tokenizer,
        DEFAULT_VARIABLES,
        args.eval_samples,
        max_depth=args.max_depth,
        seed=args.seed,
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

    if args.load_path:
        print(f"Loading checkpoint from {args.load_path}...")
        load_checkpoint(model, head, args.load_path, device=args.device)
    else:
        print("Warning: Evaluating an untrained model (no --load-path provided).")

    print(f"Evaluating model on {args.device}...")
    metrics = evaluate(model, head, eval_loader, device=args.device)
    pretty_print_metrics(metrics, prefix="Evaluation")
    return 0


def _cmd_train_main(args: argparse.Namespace) -> int:
    """Train the self-aware EML Transformer (Main Decoder)."""
    tokenizer = EMLTokenizer.from_variables(DEFAULT_VARIABLES)

    # 1. Dataset & Loaders
    train_ds = EffortDataset(
        tokenizer,
        DEFAULT_VARIABLES,
        args.train_samples,
        max_depth=args.max_depth,
        seed=args.seed,
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

    # 2. Model setup
    main_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        self_aware=not args.no_self_aware,  # Allow disabling self-awareness
    )

    # Load EMLTransformer using the evaluator checkpoint
    print(f"Loading evaluator from {args.evaluator_path}...")
    model = EMLTransformer.from_checkpoints(
        evaluator_path=args.evaluator_path,
        main_config=main_config,
        device=args.device,
    )

    # Add LMHead
    model.task_head = LMHead(d_model=args.d_model, vocab_size=tokenizer.vocab_size)

    t_config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        verbose=True,
    )

    print(f"Main Decoder parameters: {model.main_decoder.num_parameters():,}")
    print(f"Starting Phase 2 training on {args.device}...")
    train_lm(model, train_loader, eval_loader, t_config)

    if args.save_path:
        # We reuse save_checkpoint but we need to be careful about what we save.
        # For now, let's just save the whole EMLTransformer state.
        # Note: save_checkpoint expects (model, head, config, path)
        # We'll pass model.task_head as the head.
        print(f"Saving main model checkpoint to {args.save_path}...")
        save_checkpoint(model, model.task_head, main_config, args.save_path)

    return 0


def _cmd_eval_main(args: argparse.Namespace) -> int:
    """Evaluate a trained self-aware EML Transformer on the LM task."""
    tokenizer = EMLTokenizer.from_variables(DEFAULT_VARIABLES)

    eval_ds = EffortDataset(
        tokenizer,
        DEFAULT_VARIABLES,
        args.eval_samples,
        max_depth=args.max_depth,
        seed=args.seed,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_effort_batch(b, tokenizer.pad_id),
    )

    # 1. We need the config from the checkpoint to reconstruct the model correctly.
    # However, our current EMLTransformer.from_checkpoints is a bit limited.
    # Let's manually load for now to be safe.

    checkpoint = torch.load(args.load_path, map_location=args.device, weights_only=True)
    m_config_data = checkpoint["config"]
    m_config = (
        ModelConfig(**m_config_data)
        if isinstance(m_config_data, dict)
        else m_config_data
    )

    # We need the evaluator too.
    print(f"Loading evaluator from {args.evaluator_path}...")
    model = EMLTransformer.from_checkpoints(
        evaluator_path=args.evaluator_path,
        main_config=m_config,
        device=args.device,
    )

    # Add LMHead and load its weights
    model.task_head = LMHead(d_model=m_config.d_model, vocab_size=tokenizer.vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Evaluating main model on {args.device}...")
    acc = evaluate_lm(model, eval_loader, device=args.device)
    print(f"Main Decoder LM Accuracy: {acc:.2%}")
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
    inspect.add_argument(
        "--samples", type=int, default=5, help="Number of trees to print."
    )
    inspect.add_argument("--max-depth", type=int, default=5, help="Max tree depth.")
    inspect.add_argument("--seed", type=int, default=0, help="Random seed.")
    inspect.set_defaults(func=_cmd_inspect)

    # Subcommand: baseline
    baseline = subparsers.add_parser(
        "baseline",
        help="Evaluate simple baselines on the effort-prediction task.",
    )
    baseline.add_argument(
        "--train-samples", type=int, default=1000, help="Training samples."
    )
    baseline.add_argument(
        "--eval-samples", type=int, default=500, help="Evaluation samples."
    )
    baseline.add_argument("--seed", type=int, default=0, help="Random seed.")
    baseline.set_defaults(func=_cmd_baseline)

    # Subcommand: train
    train_parser = subparsers.add_parser(
        "train",
        help="Train the tiny transformer decoder.",
    )
    train_parser.add_argument(
        "--train-samples", type=int, default=5000, help="Training samples."
    )
    train_parser.add_argument(
        "--eval-samples", type=int, default=1000, help="Evaluation samples."
    )
    train_parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    train_parser.add_argument(
        "--d-model", type=int, default=128, help="Model dimension."
    )
    train_parser.add_argument(
        "--n-heads", type=int, default=4, help="Number of attention heads."
    )
    train_parser.add_argument(
        "--n-layers", type=int, default=4, help="Number of layers."
    )
    train_parser.add_argument(
        "--max-depth", type=int, default=5, help="Max tree depth."
    )
    train_parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu/cuda)."
    )
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    train_parser.add_argument(
        "--save-path", type=str, help="Path to save the model checkpoint."
    )
    train_parser.add_argument(
        "--load-path", type=str, help="Path to load a model checkpoint."
    )
    train_parser.set_defaults(func=_cmd_train)

    # Subcommand: eval
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate a model on the effort-prediction task.",
    )
    eval_parser.add_argument(
        "--eval-samples", type=int, default=1000, help="Evaluation samples."
    )
    eval_parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    eval_parser.add_argument(
        "--d-model", type=int, default=128, help="Model dimension."
    )
    eval_parser.add_argument(
        "--n-heads", type=int, default=4, help="Number of attention heads."
    )
    eval_parser.add_argument(
        "--n-layers", type=int, default=4, help="Number of layers."
    )
    eval_parser.add_argument("--max-depth", type=int, default=5, help="Max tree depth.")
    eval_parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu/cuda)."
    )
    eval_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    eval_parser.add_argument(
        "--load-path", type=str, help="Path to load a model checkpoint."
    )
    eval_parser.set_defaults(func=_cmd_eval)

    # Subcommand: train-main
    train_main = subparsers.add_parser(
        "train-main",
        help="Train the self-aware main transformer decoder.",
    )
    train_main.add_argument(
        "--evaluator-path",
        type=str,
        required=True,
        help="Path to Effort Evaluator checkpoint.",
    )
    train_main.add_argument(
        "--train-samples", type=int, default=5000, help="Training samples."
    )
    train_main.add_argument(
        "--eval-samples", type=int, default=1000, help="Evaluation samples."
    )
    train_main.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    train_main.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    train_main.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    train_main.add_argument("--d-model", type=int, default=128, help="Model dimension.")
    train_main.add_argument(
        "--n-heads", type=int, default=4, help="Number of attention heads."
    )
    train_main.add_argument("--n-layers", type=int, default=4, help="Number of layers.")
    train_main.add_argument("--max-depth", type=int, default=5, help="Max tree depth.")
    train_main.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu/cuda)."
    )
    train_main.add_argument("--seed", type=int, default=42, help="Random seed.")
    train_main.add_argument(
        "--save-path", type=str, help="Path to save the model checkpoint."
    )
    train_main.add_argument(
        "--no-self-aware",
        action="store_true",
        help="Disable self-aware modulation (baseline).",
    )
    train_main.set_defaults(func=_cmd_train_main)

    # Subcommand: eval-main
    eval_main = subparsers.add_parser(
        "eval-main",
        help="Evaluate a self-aware main transformer on the LM task.",
    )
    eval_main.add_argument(
        "--load-path", type=str, required=True, help="Path to main model checkpoint."
    )
    eval_main.add_argument(
        "--evaluator-path",
        type=str,
        required=True,
        help="Path to Effort Evaluator checkpoint.",
    )
    eval_main.add_argument(
        "--eval_samples", type=int, default=1000, help="Evaluation samples."
    )
    eval_main.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    eval_main.add_argument("--max-depth", type=int, default=5, help="Max tree depth.")
    eval_main.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu/cuda)."
    )
    eval_main.add_argument("--seed", type=int, default=42, help="Random seed.")
    eval_main.set_defaults(func=_cmd_eval_main)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
