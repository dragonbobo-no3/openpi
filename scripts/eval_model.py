#!/usr/bin/env python3
"""Evaluate a single dataset index using policy.infer, save metrics, and plot actions.

Example:
uv run scripts/eval_model.py \
  --checkpoint-dir /path/to/ckpt \
  --data-root /path/to/data \
  --episode-index 0 \
  --num-sample-steps 10 \
  --output eval_pi0.npz \
  --plot \
  --plot-output eval_plot.png \
  pi05_agileX_test_effort \
  --data.repo-id lerobot/test \
  --exp-name my_exp
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import platform
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Ensure repo root is importable for "scripts.*" imports when executed via uv/run.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.shared.effort_type import EffortType


def init_logging() -> None:
    formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.handlers[0].setFormatter(formatter)


def parse_eval_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint-dir", required=True, help="Path to the checkpoint directory to restore.")
    parser.add_argument("--data-root", required=True, help="Path to the dataset root directory.")
    parser.add_argument("--checkpoint-step", type=int, default=None, help="Checkpoint step to restore (default: latest).")
    parser.add_argument("--episode-index", type=int, required=True, help="Dataset index to evaluate.")
    parser.add_argument("--num-sample-steps", type=int, default=10, help="Number of diffusion steps during eval.")
    parser.add_argument("--output", type=str, default="eval_results.npz", help="Path to save evaluation npz.")
    parser.add_argument("--plot", action="store_true", help="Save a PNG of GT vs Pred actions.")
    parser.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="Output PNG path (defaults to output basename + '_actions.png').",
    )
    parser.add_argument("--skip-norm-stats", action="store_true", help="Skip applying normalization stats.")
    parser.add_argument(
        "--use-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="(Compatibility flag) recorded in npz; does not change loading path.",
    )
    return parser.parse_known_args()


def _ensure_dir(path: str | pathlib.Path) -> None:
    path = pathlib.Path(path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)


def _resolve_checkpoint_dir(base_dir: pathlib.Path, checkpoint_step: int | None) -> pathlib.Path:
    if (base_dir / "params" / "_METADATA").exists():
        return base_dir
    step_dirs: list[tuple[int, pathlib.Path]] = []
    for child in base_dir.iterdir():
        if child.is_dir():
            try:
                step_val = int(child.name)
            except ValueError:
                continue
            if (child / "params" / "_METADATA").exists():
                step_dirs.append((step_val, child))
    if checkpoint_step is not None:
        target = base_dir / str(checkpoint_step)
        if (target / "params" / "_METADATA").exists():
            return target
        raise FileNotFoundError(f"Checkpoint step {checkpoint_step} not found under {base_dir}")
    if not step_dirs:
        raise FileNotFoundError(f"No checkpoint with params found under {base_dir}")
    return max(step_dirs, key=lambda x: x[0])[1]


def load_policy(
    config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path,
    checkpoint_step: int | None,
) -> _policy_config._policy.Policy:
    ckpt_dir = _resolve_checkpoint_dir(checkpoint_dir, checkpoint_step)
    return _policy_config.create_trained_policy(config, ckpt_dir)


def strip_future_effort(
    effort: np.ndarray,
    *,
    effort_type: EffortType,
    history_len: int,
) -> np.ndarray:
    if effort_type in (EffortType.EXPERT_FUT, EffortType.EXPERT_HIS_C_FUT, EffortType.EXPERT_HIS_C_L_FUT):
        return effort[:history_len, :]
    return effort


def plot_action_compare(gt: np.ndarray, pred: np.ndarray, path: pathlib.Path, title: str = "") -> None:
    T, A = gt.shape
    fig, axes = plt.subplots(A, 1, figsize=(10, max(2.5, 2 * A)), sharex=True)
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        ax.plot(gt[:, i], label=f"GT action {i}", linestyle="--")
        ax.plot(pred[:, i], label=f"Pred action {i}")
        ax.set_ylabel(f"Dim {i}")
        ax.legend(loc="best")
    axes[-1].set_xlabel("Step")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    logging.info(f"Saved action compare plot to {path}")
    plt.close(fig)


def main() -> None:
    eval_args, remaining = parse_eval_args()
    sys.argv = [sys.argv[0]] + remaining
    config = _config.cli()
    checkpoint_dir = pathlib.Path(eval_args.checkpoint_dir).expanduser().resolve()
    data_root = pathlib.Path(eval_args.data_root).expanduser().resolve()

    init_logging()
    logging.info(f"Running eval on: {platform.node()}")

    policy = load_policy(config, checkpoint_dir, eval_args.checkpoint_step)

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.rlds_data_dir is not None:
        raise NotImplementedError("episode-index is only supported for torch datasets.")
    data_config = dataclasses.replace(data_config, root=str(data_root))

    dataset = _data_loader.create_torch_dataset(
        data_config,
        action_horizon=config.model.action_horizon,
        model_config=config.model,
    )
    # Do NOT apply transform_dataset here to avoid double-normalization; policy.infer will apply transforms.

    sample = dataset[eval_args.episode_index]
    # Pick action key from dataset; prefer action_sequence_keys, then common fallback.
    action_keys = list(data_config.action_sequence_keys) if hasattr(data_config, "action_sequence_keys") else []
    action_keys += ["actions", "action"]
    action_key = None
    for k in action_keys:
        if k in sample:
            action_key = k
            break
    if action_key is None:
        raise KeyError(f"No action key found in sample. Checked: {action_keys}")

    actions_np = np.asarray(sample[action_key])[None, ...]  # keep GT

    history_len = len(data_config.effort_history)
    effort = sample.get("effort")
    if effort is not None:
        sample["effort"] = strip_future_effort(
            np.asarray(effort),
            effort_type=config.model.effort_type,
            history_len=history_len,
        )
    # Remove actions before infer
    sample_for_infer = {k: v for k, v in sample.items() if k != action_key}

    tic = time.time()
    outputs = policy.infer(sample_for_infer)
    infer_times = [(time.time() - tic) * 1e3]
    preds_np = np.asarray(outputs["actions"])[None, ...]  # add batch dim for consistency

    mse = np.mean((preds_np - actions_np) ** 2, axis=(1, 2))
    losses = mse.tolist()
    pred_samples = [preds_np]
    gt_samples = [actions_np]

    action_plot_path = (
        pathlib.Path(eval_args.plot_output)
        if eval_args.plot_output
        else pathlib.Path(eval_args.output).with_name(pathlib.Path(eval_args.output).stem + "_actions.png")
    )
    if eval_args.plot:
        plot_action_compare(
            actions_np[0],
            preds_np[0],
            action_plot_path,
            title=f"Episode index {eval_args.episode_index}",
        )

    losses_arr = np.asarray(losses, dtype=np.float32)
    mean_mse = float(np.mean(losses_arr))
    std_mse = float(np.std(losses_arr))

    output_path = pathlib.Path(eval_args.output)
    _ensure_dir(output_path)
    np.savez_compressed(
        output_path,
        batch_mse=losses_arr,
        mean_mse=np.float32(mean_mse),
        std_mse=np.float32(std_mse),
        pred_samples=np.asarray(pred_samples, dtype=np.float32),
        gt_samples=np.asarray(gt_samples, dtype=np.float32),
        infer_times_ms=np.asarray(infer_times, dtype=np.float32),
        checkpoint_dir=str(checkpoint_dir),
        data_root=str(data_root),
        checkpoint_step=np.int32(eval_args.checkpoint_step or -1),
        use_ema=np.bool_(eval_args.use_ema),
        config=str(config),
        episode_index=np.int32(eval_args.episode_index),
    )
    logging.info(f"Saved eval results to {output_path} (mean MSE={mean_mse:.4f}, std={std_mse:.4f})")

    if eval_args.plot:
        plot_path = pathlib.Path(eval_args.plot_output or output_path.with_suffix(".png"))
        _ensure_dir(plot_path)
        plt.figure(figsize=(8, 4))
        plt.plot(losses_arr, label="MSE")
        plt.axhline(mean_mse, color="tab:orange", linestyle="--", label="Mean MSE")
        plt.xlabel("Sample")
        plt.ylabel("MSE")
        plt.title(f"Eval MSE | sample={eval_args.episode_index} mean={mean_mse:.4f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        logging.info(f"Saved plot to {plot_path}")
        plt.close()


if __name__ == "__main__":
    main()
