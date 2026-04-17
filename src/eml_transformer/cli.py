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
    VALID_FFN_MODES,
    EffortHead,
    EMLTransformer,
    LMHead,
    ModelConfig,
    TinyDecoder,
    make_config,
)

torch.serialization.add_safe_globals([ModelConfig])
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
        ffn_mode=args.ffn_mode,
        ffn_expansion=args.ffn_expansion,
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

    # 1. Setup model and config first.
    if args.load_path:
        print(f"Loading checkpoint from {args.load_path}...")
        ckpt = torch.load(args.load_path, map_location=args.device, weights_only=True)
        raw_cfg = ckpt.get("config")
        saved_config = make_config(**raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg

        if saved_config is None:
            print(
                "Warning: Checkpoint missing config. Falling back to default ModelConfig."
            )
            saved_config = ModelConfig()

        # Match the evaluation depth to the model capacity if not explicitly specified
        # (check if args.max_depth is the default value of 5).
        if args.max_depth == 5 and saved_config.num_bins != 6:
            args.max_depth = saved_config.num_bins - 1
            print(
                f"Auto-adjusting --max-depth to {args.max_depth} to match checkpoint."
            )

        model = TinyDecoder(saved_config)
        head = EffortHead(d_model=saved_config.d_model, num_bins=saved_config.num_bins)
        load_checkpoint(model, head, args.load_path, device=args.device)
    else:
        print("Warning: Evaluating an untrained model (no --load-path provided).")
        saved_config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            num_bins=args.max_depth + 1,
        )
        model = TinyDecoder(saved_config)
        head = EffortHead(d_model=saved_config.d_model, num_bins=saved_config.num_bins)

    # 2. Setup data using the potentially-updated max_depth.
    eval_ds = EffortDataset(
        tokenizer,
        DEFAULT_VARIABLES,
        args.eval_samples,
        max_depth=args.max_depth,
        branch_schedule_depth=args.branch_schedule_depth,
        seed=args.seed,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_effort_batch(b, tokenizer.pad_id),
    )

    print(f"Evaluating model on {args.device} (max_depth={args.max_depth})...")
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

    # 2. Model setup. ``--ffn-mode`` is the modern flag; ``--no-self-aware``
    # is kept as a deprecated alias that forces vanilla mode.
    ffn_mode = "vanilla" if args.no_self_aware else args.ffn_mode
    main_config = make_config(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_mode=ffn_mode,
        ffn_expansion=args.ffn_expansion,
    )

    # Load EMLTransformer using the evaluator checkpoint
    print(f"Loading evaluator from {args.evaluator_path}...")
    model = EMLTransformer.from_checkpoints(
        evaluator_path=args.evaluator_path,
        main_config=main_config,
        device=args.device,
    )

    # Add LMHead and ensure the entire model is on the correct device.
    model.task_head = LMHead(d_model=args.d_model, vocab_size=tokenizer.vocab_size)
    model.to(args.device)

    t_config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        verbose=True,
    )

    # Report trainable parameter count — this is the fair number to cite
    # across modes because it excludes the frozen evaluator.
    print(f"Config: ffn_mode={ffn_mode}, ffn_expansion={args.ffn_expansion}")
    print(f"Trainable parameters: {model.num_trainable_parameters():,}")
    print(f"Starting Phase 2 training on {args.device}...")
    train_lm(model, train_loader, eval_loader, t_config)

    if args.save_path:
        print(f"Saving main model checkpoint to {args.save_path}...")
        save_checkpoint(model, model.task_head, main_config, args.save_path)

    return 0


def _cmd_eval_depth(args: argparse.Namespace) -> int:
    """Evaluate a trained Effort Evaluator on held-out deeper trees.

    Trains nothing. Loads the model from ``--load-path`` and reports
    per-``max-depth`` accuracy so we can see whether the learned depth
    signal generalizes from the training distribution (typically
    max_depth=5) to deeper, unseen trees (max_depth=6, 7, ...).

    The random-tree generator's ``branch_schedule_depth`` is held fixed at
    the training value, so shallow trees have the *same* distribution in
    every eval bucket — we only change the cap.
    """
    tokenizer = EMLTokenizer.from_variables(DEFAULT_VARIABLES)

    # Load checkpoint once to read the saved config (it tells us how to
    # size the head and whether the model was trained at depth 5 or another).
    ckpt = torch.load(args.load_path, map_location=args.device, weights_only=True)
    saved_config = ckpt.get("config")
    if isinstance(saved_config, dict):
        saved_config = make_config(**saved_config)
    if saved_config is None:
        saved_config = ModelConfig()

    model = TinyDecoder(saved_config)
    head = EffortHead(d_model=saved_config.d_model, num_bins=saved_config.num_bins)
    load_checkpoint(model, head, args.load_path, device=args.device)

    # The schedule used at training time anchors the tree distribution. Any
    # eval bucket with a larger cap draws trees from the same conditional
    # distribution at shallow depths — only the tail extends.
    branch_schedule = args.branch_schedule_depth

    print(
        f"Held-out depth evaluation"
        f" | model num_bins={saved_config.num_bins}"
        f" | branch_schedule_depth={branch_schedule}"
    )
    for eval_depth in args.eval_max_depths:
        ds = EffortDataset(
            tokenizer,
            DEFAULT_VARIABLES,
            args.eval_samples,
            max_depth=eval_depth,
            branch_schedule_depth=branch_schedule,
            seed=args.seed,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            collate_fn=lambda b: collate_effort_batch(b, tokenizer.pad_id),
        )
        metrics = evaluate(model, head, loader, device=args.device)
        pretty_print_metrics(metrics, prefix=f"max_depth={eval_depth}")
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

    # ``weights_only=False`` because we saved a ``ModelConfig`` dataclass —
    # our own checkpoints only.
    checkpoint = torch.load(args.load_path, map_location=args.device, weights_only=True)
    raw_cfg = checkpoint["config"]
    m_config = make_config(**raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg

    print(f"Loading evaluator from {args.evaluator_path}...")
    model = EMLTransformer.from_checkpoints(
        evaluator_path=args.evaluator_path,
        main_config=m_config,
        device=args.device,
    )
    model.task_head = LMHead(d_model=m_config.d_model, vocab_size=tokenizer.vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    print(f"Evaluating main model on {args.device}...")
    acc = evaluate_lm(model, eval_loader, device=args.device)
    print(f"Main Decoder LM Accuracy: {acc:.2%}")
    return 0


def _cmd_compare_modes(args: argparse.Namespace) -> int:
    """Parameter-matched ablation across FFN modes, averaged over seeds.

    Runs training for each of ``--modes`` on identical data and the same list
    of seeds. Prints a side-by-side summary with mean±std of final eval
    accuracy and trainable parameter count. Use this instead of single-run
    comparisons — the prior report's 0.23% gap was inside expected noise.

    This runs the *Effort Evaluator* task (depth prediction on RPN sequences).
    Comparing LM-task main-decoder modes would require a trained evaluator
    first; run that loop manually via ``train-main`` with different modes.
    """
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

    results: dict[str, dict[str, list[float] | int]] = {}
    for mode in args.modes:
        accs: list[float] = []
        maes: list[float] = []
        param_count = 0

        # Calculate expansion for this mode. If --parity is set, we double the
        # expansion for vanilla/film to match delta's parameter count.
        eff_expansion = args.ffn_expansion
        if args.parity and mode in ("vanilla", "film"):
            eff_expansion *= 2

        for seed in args.seeds:
            torch.manual_seed(seed)
            cfg = make_config(
                vocab_size=tokenizer.vocab_size,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                ffn_mode=mode,
                ffn_expansion=eff_expansion,
                num_bins=args.max_depth + 1,
            )
            model = TinyDecoder(cfg)
            head = EffortHead(d_model=cfg.d_model, num_bins=cfg.num_bins)
            param_count = model.num_parameters() + sum(
                p.numel() for p in head.parameters()
            )
            t_config = TrainConfig(
                epochs=args.epochs,
                lr=args.lr,
                device=args.device,
                verbose=False,
            )
            print(
                f"  [{mode} seed={seed}] training ({param_count:,} params, exp={eff_expansion})..."
            )
            train(model, head, train_loader, eval_loader, t_config)
            metrics = evaluate(model, head, eval_loader, device=args.device)
            accs.append(metrics.accuracy)
            maes.append(metrics.mae)
        results[mode] = {"accs": accs, "maes": maes, "params": param_count}

    # Pretty summary.
    print("\n=== compare-modes summary ===")
    print(
        f"{'mode':<10} {'params':>10}  {'acc_mean':>10} {'acc_std':>8} "
        f"{'mae_mean':>10} {'mae_std':>8}"
    )
    for mode in args.modes:
        r = results[mode]
        accs = torch.tensor(r["accs"])  # type: ignore[arg-type]
        maes = torch.tensor(r["maes"])  # type: ignore[arg-type]

        # Handle single-seed case (std=0 instead of nan)
        acc_std = accs.std().item() if len(accs) > 1 else 0.0
        mae_std = maes.std().item() if len(maes) > 1 else 0.0

        print(
            f"{mode:<10} {r['params']:>10,}  "
            f"{accs.mean().item():>10.4%} {acc_std:>8.4%} "
            f"{maes.mean().item():>10.4f} {mae_std:>8.4f}"
        )
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
        "--ffn-mode",
        type=str,
        default="vanilla",
        choices=list(VALID_FFN_MODES),
    )
    train_parser.add_argument("--ffn-expansion", type=int, default=4)
    train_parser.add_argument("--branch-schedule-depth", type=int, default=5)
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
    eval_parser.add_argument(
        "--branch-schedule-depth",
        type=int,
        default=5,
        help="Tree distribution schedule (usually match training depth).",
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
    train_main.add_argument(
        "--ffn-mode",
        type=str,
        default="delta",
        choices=list(VALID_FFN_MODES),
        help="Modulation variant for main decoder FFNs. Overridden by --no-self-aware.",
    )
    train_main.add_argument(
        "--ffn-expansion",
        type=int,
        default=4,
        help="FFN expansion ratio. Raise for vanilla to parameter-match delta (try 6-7).",
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
        "--eval-samples", type=int, default=1000, help="Evaluation samples."
    )
    eval_main.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    eval_main.add_argument("--max-depth", type=int, default=5, help="Max tree depth.")
    eval_main.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu/cuda)."
    )
    eval_main.add_argument("--seed", type=int, default=42, help="Random seed.")
    eval_main.set_defaults(func=_cmd_eval_main)

    # Subcommand: eval-depth — held-out generalization test for the evaluator.
    eval_depth = subparsers.add_parser(
        "eval-depth",
        help="Evaluate a trained Effort Evaluator on held-out deeper trees.",
    )
    eval_depth.add_argument(
        "--load-path", type=str, required=True, help="Path to evaluator checkpoint."
    )
    eval_depth.add_argument(
        "--eval-samples", type=int, default=2000, help="Eval samples per depth bucket."
    )
    eval_depth.add_argument(
        "--eval-max-depths",
        type=int,
        nargs="+",
        default=[5, 6, 7],
        help="Max-depth buckets to evaluate (e.g. 5 6 7).",
    )
    eval_depth.add_argument(
        "--branch-schedule-depth",
        type=int,
        default=5,
        help="Must match the training-time schedule, otherwise the distribution shifts.",
    )
    eval_depth.add_argument("--batch-size", type=int, default=64)
    eval_depth.add_argument("--device", type=str, default="cpu")
    eval_depth.add_argument("--seed", type=int, default=123)
    eval_depth.set_defaults(func=_cmd_eval_depth)

    # Subcommand: compare-modes — parameter-matched ablation over FFN modes.
    compare = subparsers.add_parser(
        "compare-modes",
        help="Train multiple FFN modes over multiple seeds and report mean±std.",
    )
    compare.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["vanilla", "delta", "film"],
        choices=list(VALID_FFN_MODES),
        help="FFN modes to compare (any subset of vanilla/delta/film).",
    )
    compare.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds to run per mode (mean±std is over these).",
    )
    compare.add_argument("--train-samples", type=int, default=5000)
    compare.add_argument("--eval-samples", type=int, default=1000)
    compare.add_argument("--epochs", type=int, default=5)
    compare.add_argument("--batch-size", type=int, default=64)
    compare.add_argument("--lr", type=float, default=1e-3)
    compare.add_argument("--d-model", type=int, default=128)
    compare.add_argument("--n-heads", type=int, default=4)
    compare.add_argument("--n-layers", type=int, default=4)
    compare.add_argument("--max-depth", type=int, default=5)
    compare.add_argument("--ffn-expansion", type=int, default=4)
    compare.add_argument(
        "--parity",
        action="store_true",
        help="Double FFN expansion for vanilla/film to match delta parameter count.",
    )
    compare.add_argument("--device", type=str, default="cpu")
    compare.add_argument(
        "--seed", type=int, default=42, help="Data seed (not run seed)."
    )
    compare.set_defaults(func=_cmd_compare_modes)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
