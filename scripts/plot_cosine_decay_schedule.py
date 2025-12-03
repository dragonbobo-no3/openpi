#!/usr/bin/env python3
"""
    Plot the CosineDecaySchedule learning rate curve.

    python3 scripts/plot_cosine_decay_schedule.py \
        --warmup_steps 1000 \
        --peak_lr 1e-4 \
        --decay_steps 3000 \
        --decay_lr 1e-6 \
        --total_steps 3000 \
        --show
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

# Ensure the repo's src/ is importable when running this script directly.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from openpi.training.optimizer import (
        CosineDecaySchedule as RepoCosineDecaySchedule,  # type: ignore
    )
except Exception:  # pragma: no cover - soft fallback for missing deps
    RepoCosineDecaySchedule = None  # type: ignore


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return ivalue


def parse_args() -> argparse.Namespace:
    defaults = (
        RepoCosineDecaySchedule()
        if RepoCosineDecaySchedule is not None
        else argparse.Namespace(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        )
    )
    parser = argparse.ArgumentParser(
        description="Plot the learning rate curve produced by CosineDecaySchedule."
    )
    parser.add_argument(
        "--warmup_steps",
        type=positive_int,
        default=defaults.warmup_steps,
        help=f"Number of warmup steps (default: {defaults.warmup_steps})",
    )
    parser.add_argument(
        "--peak_lr",
        type=float,
        default=defaults.peak_lr,
        help=f"Peak learning rate reached after warmup (default: {defaults.peak_lr})",
    )
    parser.add_argument(
        "--decay_steps",
        type=positive_int,
        default=defaults.decay_steps,
        help=f"Number of cosine decay steps (default: {defaults.decay_steps})",
    )
    parser.add_argument(
        "--decay_lr",
        type=float,
        default=defaults.decay_lr,
        help=f"Final learning rate after decay (default: {defaults.decay_lr})",
    )
    parser.add_argument(
        "--total_steps",
        type=positive_int,
        default=None,
        help=(
            "Total steps to plot. Defaults to warmup_steps + decay_steps. "
            "Steps beyond warmup+decay hold at the final LR."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cosine_decay_lr.png",
        help="Where to save the plot (default: cosine_decay_lr.png)",
    )
    parser.add_argument(
        "--dpi",
        type=positive_int,
        default=200,
        help="DPI for the saved figure (default: 200)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window after saving (may require a GUI).",
    )
    return parser.parse_args()


def cosine_decay_with_warmup(
    step: int,
    warmup_steps: int,
    peak_lr: float,
    decay_steps: int,
    decay_lr: float,
) -> float:
    """Pure-Numpy reproduction of optax.warmup_cosine_decay_schedule."""
    if warmup_steps < 0 or decay_steps <= 0:
        raise ValueError("warmup_steps must be >=0 and decay_steps must be >0")

    init_lr = peak_lr / (warmup_steps + 1)
    if warmup_steps > 0 and step < warmup_steps:
        return init_lr + (peak_lr - init_lr) * step / warmup_steps

    progress = max(step - warmup_steps, 0)
    progress = min(progress, decay_steps)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * progress / decay_steps))
    return decay_lr + (peak_lr - decay_lr) * cosine_decay


def get_lr_fn(
    warmup_steps: int,
    peak_lr: float,
    decay_steps: int,
    decay_lr: float,
):
    """Prefer the repo's schedule; fall back to a NumPy reimplementation if deps are missing."""
    if RepoCosineDecaySchedule is not None:
        try:
            schedule = RepoCosineDecaySchedule(
                warmup_steps=warmup_steps,
                peak_lr=peak_lr,
                decay_steps=decay_steps,
                decay_lr=decay_lr,
            )
            return schedule.create()
        except Exception as exc:  # pragma: no cover - last-resort fallback
            print(f"Falling back to NumPy schedule (reason: {exc})")
    return lambda step: cosine_decay_with_warmup(step, warmup_steps, peak_lr, decay_steps, decay_lr)


def main() -> None:
    args = parse_args()
    total_steps = args.total_steps or args.warmup_steps + args.decay_steps
    steps = np.arange(total_steps)

    lr_fn = get_lr_fn(
        warmup_steps=args.warmup_steps,
        peak_lr=args.peak_lr,
        decay_steps=args.decay_steps,
        decay_lr=args.decay_lr,
    )
    lrs = np.array([float(lr_fn(step)) for step in steps])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, lrs, label="Learning rate")
    ax.axvline(args.warmup_steps, color="tab:orange", linestyle="--", label="Warmup end")
    ax.set_title(

            "CosineDecaySchedule | "
            f"warmup={args.warmup_steps}, peak_lr={args.peak_lr}, "
            f"decay_steps={args.decay_steps}, decay_lr={args.decay_lr}, "
            f"total_steps={total_steps}"

    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning rate")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)

    print(f"Saved plot to {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
