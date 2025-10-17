#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法：
uv run scripts/evaluate_all.py run --episode_id 0 --out ./all_temp.npz
uv run scripts/evaluate_all.py plot --inp ./all_temp.npz --out ./all_temp.png --pred_start 10
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.models.tokenizer import PaligemmaTokenizer


def _select_episode_indices(dataset, episode_id: int):
    ds = dataset.hf_dataset
    ep_col = np.asarray(ds["episode_index"])
    idxs = np.nonzero(ep_col == episode_id)[0].tolist()
    if not idxs:
        raise ValueError(f"Episode {episode_id} not found.")
    return idxs

def _ensure_dir(path: str):
    if path and os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

def _to_batch_np(arr, dtype=None):
    a = np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)
    return a[None, ...]


def run_infer_and_save(args):
    _ensure_dir(args.out)
    cfg = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(cfg, args.checkpoint_dir)
    tokenizer = PaligemmaTokenizer()
    if hasattr(policy, "reset"):
        policy.reset()
    dataset = lerobot_dataset.LeRobotDataset(args.repo_id, root=args.root)
    episode_steps = _select_episode_indices(dataset, args.episode_id)
    gt_actions_list = []
    pred_actions_list = []  # [T, N, action_dim]
    infer_times_ms = []
    infer_states_list = []

    N = getattr(args, 'pred_len', 50)  # 默认N=50，可通过plot参数传递
    for t, idx in enumerate(episode_steps):
        step = dataset[idx]
        gt_actions_list.append(np.asarray(step["action"]))
        prompt = step.get("task") or args.default_prompt
        tokenized, mask = tokenizer.tokenize(prompt)
        tokenized = _to_batch_np(tokenized, dtype=np.int32)
        mask = _to_batch_np(mask, dtype=bool)
        cur_state = np.asarray(step["observation.state"])
        infer_states_list.append(cur_state)
        obs = {
            "images": {
                "camera0": step["observation.images.camera0"],
                "camera1": step["observation.images.camera1"],
                "camera2": step["observation.images.camera2"],
                "camera3": step["observation.images.camera3"],
            },
            "image_masks": {
                "camera0": np.array([True], dtype=bool),
                "camera1": np.array([True], dtype=bool),
                "camera2": np.array([True], dtype=bool),
                "camera3": np.array([True], dtype=bool),
            },
            "state": cur_state,
            "tokenized_prompt": tokenized,
            "tokenized_prompt_mask": mask,
            "token_ar_mask": None,
            "token_loss_mask": None,
        }
        tic = time.time()
        result = policy.infer(obs)
        infer_times_ms.append((time.time() - tic) * 1e3)
        pred_seq = np.asarray(result["actions"])  # shape [N, action_dim]
        # 截断到episode结尾
        remain = len(episode_steps) - t
        if pred_seq.shape[0] > remain:
            pred_seq = pred_seq[:remain]
        # 若不足N步则pad
        if pred_seq.shape[0] < N:
            pad = np.zeros((N - pred_seq.shape[0], pred_seq.shape[1]), dtype=pred_seq.dtype)
            pred_seq = np.concatenate([pred_seq, pad], axis=0)
        pred_actions_list.append(pred_seq)
        print(f"step {t}/{len(episode_steps)}")

    gt_actions = np.stack(gt_actions_list)
    pred_actions = np.stack(pred_actions_list)  # [T, N, action_dim]
    infer_times_ms = np.asarray(infer_times_ms, dtype=np.float32)
    infer_states = np.stack(infer_states_list, axis=0) if infer_states_list else np.zeros((0, gt_actions.shape[1]), dtype=gt_actions.dtype)

    np.savez_compressed(
        args.out,
        gt_actions=gt_actions,
        pred_actions=pred_actions,
        episode_id=np.int32(args.episode_id),
        infer_times_ms=infer_times_ms,
        infer_states=infer_states,
        repo_id=args.repo_id,
        root=args.root,
        checkpoint_dir=args.checkpoint_dir,
        config=args.config,
        pred_len=N,
    )
    print(f"[RUN] saved npz -> {args.out} | gt={gt_actions.shape} pred={pred_actions.shape} infer_states={infer_states.shape}")
    del policy

    if args.plot_after_run:
        class P:
            pass
        p = P()
        setattr(p, 'inp', args.out)
        if args.out_png:
            setattr(p, 'out', args.out_png)
        else:
            base = os.path.splitext(args.out)[0]
            setattr(p, 'out', base + "_action_compare_all.png")
        setattr(p, 'dpi', args.dpi)
        setattr(p, 'pred_start', getattr(args, 'pred_start', 0))
        plot_saved(p)


def plot_saved(args):
    _ensure_dir(args.out)
    # 支持多条预测曲线
    pred_files = getattr(args, 'pred_files', None)
    pred_labels = getattr(args, 'pred_labels', None)
    if pred_files is None:
        pred_files = [args.inp]
    if pred_labels is None or len(pred_labels) != len(pred_files):
        pred_labels = [os.path.splitext(os.path.basename(f))[0] for f in pred_files]

    # 读取GT
    data = np.load(pred_files[0], allow_pickle=True)
    gt = data["gt_actions"]
    infer_states = data["infer_states"]
    T, N, A = data["pred_actions"].shape
    pred_start = getattr(args, 'pred_start', 0)
    stride = getattr(args, 'pred_len', N)
    pred_start = max(0, min(pred_start, T))
    stride = max(1, min(stride, N))

    # 生成所有预测曲线
    plot_preds = []
    infer_idxs = []
    for pf in pred_files:
        pdata = np.load(pf, allow_pickle=True)
        pred = pdata["pred_actions"]
        plot_pred = np.copy(gt)
        infer_idx = []
        t = pred_start
        while t < T:
            n = min(stride, T - t)
            plot_pred[t:t+n] = pred[t, :n]
            infer_idx.append(t)
            t += n
        plot_preds.append(plot_pred)
        infer_idxs.append(infer_idx)

    fig, axes = plt.subplots(A, 1, figsize=(10, 4 * A), sharex=True)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    for i in range(len(gt)):
        if gt[i].max() > 1e8:
            gt[i] = gt[i-1] if i > 0 else gt[i+1]
    for i, ax in enumerate(axes):
        ax.plot(gt[:, i], label=f"GT action {i}", linestyle="--")
        for j, plot_pred in enumerate(plot_preds):
            ax.plot(plot_pred[:, i], label=f"{pred_labels[j]} pred", alpha=0.8)
            # 高亮推理节点
            idxs = infer_idxs[j]
            if len(idxs) > 0:
                ax.scatter(idxs, plot_pred[idxs, i], s=30, label=f"{pred_labels[j]} inf node", zorder=5)
        ax.set_ylabel(f"Action dim {i}")
        ax.set_title(f"GT vs Predicted Actions (dim {i})")
        ax.legend(loc="best")
    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"[PLOT] saved figure -> {args.out}")
    plt.show()
    plt.close(fig)


def build_cli():
    parser = argparse.ArgumentParser(
        description="Run policy inference (all frames) or plot from saved npz."
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    # run
    p_run = subparsers.add_parser("run", help="Run inference for all frames and save results to .npz")
    p_run.add_argument("--config", default="pi05_agileX")
    p_run.add_argument("--checkpoint_dir", default="/home/test/jemotor/jemodel/pi05/1014_pi05_test/12500/")
    p_run.add_argument("--repo_id", default="lerobot/test")
    p_run.add_argument("--root", default="/home/test/jemotor/jedata/test_0928_100_v2/")
    p_run.add_argument("--episode_id", type=int, default=54)
    p_run.add_argument("--default_prompt", default="pick up the circular chip and place it on the yellow pot")
    p_run.add_argument("--out", default="./all_save2.npz")
    p_run.add_argument("--plot-after-run", action="store_true", help="After saving npz, immediately plot.")
    p_run.add_argument("--out-png", default="", help="If --plot-after-run, output PNG path (optional).")
    p_run.add_argument("--dpi", type=int, default=150)
    p_run.add_argument("--pred_start", type=int, default=0, help="Start frame for using pred_actions in plot.")
    p_run.set_defaults(func=run_infer_and_save)
    # plot
    p_plot = subparsers.add_parser("plot", help="Plot GT vs Pred from saved .npz, with pred_start and pred_len option, and support multi pred_files")
    p_plot.add_argument("--inp", default="./all_save2.npz")
    p_plot.add_argument("--out", default="./all_save2.png")
    p_plot.add_argument("--dpi", type=int, default=150)
    p_plot.add_argument("--pred_start", type=int, default=20, help="Start frame for using pred_actions in plot.")
    p_plot.add_argument("--pred_len", type=int, default=50, help="Number of frames to use pred_actions in plot (-1 means to end)")
    p_plot.add_argument("--pred_files", nargs="*", default=None, help="List of npz files for multiple pred curves")
    p_plot.add_argument("--pred_labels", nargs="*", default=None, help="List of labels for pred curves")
    p_plot.set_defaults(func=plot_saved)
    return parser

def main():
    parser = build_cli()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
