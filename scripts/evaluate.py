#!/usr/bin/env python3
"""
用法：
uv run scripts/evaluate.py run --episode_id 0 --period 50 --out ./temp.npz
uv run scripts/evaluate.py plot --inp ./temp.npz --out ./temp.png
"""

import argparse
from collections.abc import Sequence
import os
import time

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import matplotlib.pyplot as plt
import numpy as np

from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


# ========== 工具函数 ==========
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


# ========== 子命令：run ==========
def run_infer_and_save(args):
    _ensure_dir(args.out)

    # 配置 & 模型
    cfg = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(cfg, args.checkpoint_dir)
    tokenizer = PaligemmaTokenizer()
    if hasattr(policy, "reset"):
        policy.reset()

    action_sequence_keys: Sequence[str] = ("action",)
    # 数据集 & episode
    dataset = lerobot_dataset.LeRobotDataset(args.repo_id, root=args.root,delta_timestamps={
            key: [t / args.fps for t in range(args.period)] for key in action_sequence_keys
        },)
    # print(dataset.__len__())
    # print(dataset[177256]["observation.state"])
    # print(dataset[177256]["index"])
    # print(dataset[179790]["observation.state"])
    # print(dataset[179790]["index"])
    # print(dataset.__getitem__(177256).keys())
    # print(dataset.__getitem__(179790)['action'])
    # exit(1)
    episode_steps = _select_episode_indices(dataset, args.episode_id)
    gt_actions_list, pred_actions_list = [], []
    infer_times_ms = []
    infer_states_list = []  # ⭐ 记录每次推理时使用的 state
    prev_main = None

    for t, idx in enumerate(episode_steps):
        step = dataset.__getitem__(idx)
        # gt_actions_list.append(np.asarray(step["action"]))

        # if t == 501:
        #     break

        if t % args.period == 0:
            # 文本
            prompt = step.get("task") or args.default_prompt
            tokenized, mask = tokenizer.tokenize(prompt)
            tokenized = _to_batch_np(tokenized, dtype=np.int32)
            mask = _to_batch_np(mask, dtype=bool)

            # 观测（这里的 state 就是“当次推理使用的 state”）
            cur_state = np.asarray(step["observation.state"])
            # Ensure obs["state"] is a numpy array
            state7 = np.asarray(step["observation.state"])
            if args.mode == "speed":
                # compute delta = current_main - prev_main (or zeros for first frame)
                if prev_main is None:
                    delta = np.zeros_like(state7)
                else:
                    try:
                        delta = state7 - prev_main
                    except Exception:
                        delta = np.zeros_like(state7)
                obs = np.concatenate([state7, delta], axis=-1)
                prev_main = state7.copy()
            else:
                obs = state7
            cur_state = obs
            infer_states_list.append(cur_state)  # ⭐ 保存
            print(cur_state)
            obs = {
                "images": {
                    "camera0": step["camera0"],
                    "camera1": step["camera1"],
                    "camera2": step["camera2"],
                    "camera3": step["camera3"],
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

            remain = len(episode_steps) - t
            block = np.asarray(result["actions"])[:min(args.period, remain)]
            gt_actions_list.extend(np.asarray(step["action"]))
            pred_actions_list.extend(block)
            print(f"current step {t}/{len(episode_steps)}")

    # 对齐并保存
    min_len = min(len(gt_actions_list), len(pred_actions_list))
    gt_actions = np.stack(gt_actions_list[:min_len])           # [T, A]
    pred_actions = np.stack(pred_actions_list[:min_len])       # [T, A]
    infer_times_ms = np.asarray(infer_times_ms, dtype=np.float32)
    infer_states = np.stack(infer_states_list, axis=0) if infer_states_list else np.zeros((0, gt_actions.shape[1]), dtype=gt_actions.dtype)

    np.savez_compressed(
        args.out,
        gt_actions=gt_actions,
        pred_actions=pred_actions,
        period=np.int32(args.period),
        episode_id=np.int32(args.episode_id),
        infer_times_ms=infer_times_ms,
        infer_states=infer_states,  # ⭐ 保存每次推理使用的 state
        repo_id=args.repo_id,
        root=args.root,
        checkpoint_dir=args.checkpoint_dir,
        config=args.config,
    )
    print(f"[RUN] saved npz -> {args.out} | gt={gt_actions.shape} pred={pred_actions.shape} infer_states={infer_states.shape}")

    del policy

    if args.plot_after_run:
        # 直接复用 plot 子命令
        class P: pass
        p = P()
        p.inp = args.out
        if args.out_png:
            p.out = args.out_png
        else:
            base = os.path.splitext(args.out)[0]
            p.out = base + "_action_compare_all.png"
        p.dpi = args.dpi
        plot_saved(p)


# ========== 子命令：plot ==========
def plot_saved(args):
    _ensure_dir(args.out)

    data = np.load(args.inp, allow_pickle=False)
    gt = data["gt_actions"]            # [T, A]
    pred = data["pred_actions"]        # [T, A]
    period = int(data["period"])
    infer_states = data["infer_states"]  # ⭐ [K, A]，每次推理使用的 state

    for i in range(gt.shape[0]):
        if gt[i].max() > 1e8:
            gt[i] = gt[i-1] if i > 0 else gt[i+1]

    T, A = gt.shape
    assert pred.shape == gt.shape, f"Shape mismatch: gt {gt.shape}, pred {pred.shape}"
    K = infer_states.shape[0]

    # 由 K 与 period 重建每次推理对应的时间索引（不再依赖 start_steps）
    highlight_idx_full = np.arange(0, K * period, period, dtype=np.int32)
    mask = highlight_idx_full < T
    highlight_idx = highlight_idx_full[mask]
    infer_states_vis = infer_states[: len(highlight_idx)]  # 与高亮步数对齐

    fig, axes = plt.subplots(A, 1, figsize=(10, 4 * A), sharex=True)
    axes = np.atleast_1d(axes)

    for i, ax in enumerate(axes):
        ax.plot(gt[:, i], label=f"GT action {i}", linestyle="--")
        ax.plot(pred[:, i], label=f"Pred action {i}")
        # 红点：每段期初的预测值
        if highlight_idx.size > 0:
            # ax.scatter(highlight_idx, pred[highlight_idx, i], label="First pred in period", zorder=5, s=20, color="red")
            # 绿色 x：每次推理时使用的 state
            ax.scatter(highlight_idx, infer_states_vis[:, i], label="State at infer", zorder=6, s=30, marker="x", color="green")
        ax.set_ylabel(f"Action dim {i}")
        ax.set_title(f"GT vs Predicted Actions (dim {i})")
        ax.legend(loc="best")

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"[PLOT] saved figure -> {args.out}")

    # 显示窗口（阻塞直到关闭）
    plt.show()

    # 关闭以释放内存
    plt.close(fig)


# ========== CLI ==========
def build_cli():
    parser = argparse.ArgumentParser(
        description="Run policy inference (save npz) or plot from saved npz, all-in-one script."
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # run
    p_run = subparsers.add_parser("run", help="Run inference and save results to .npz")
    p_run.add_argument("--config", default="pi05_agileX_test_effort")
    p_run.add_argument("--checkpoint_dir", default="/home/kleist/Documents/Model/cloud_server/1128_pi05_test/10000/")
    p_run.add_argument("--repo_id", default="lerobot/test")
    p_run.add_argument("--root", default="/home/test/jemotor/jedata/test_1112_trans/")
    p_run.add_argument("--episode_id", type=int, default=5)
    p_run.add_argument("--period", type=int, default=50)
    p_run.add_argument("--default_prompt", default="pick up the circular chip and place it on the yellow pot")
    p_run.add_argument("--out", default="./save2.npz")
    p_run.add_argument("--plot-after-run", action="store_true", help="After saving npz, immediately plot.")
    p_run.add_argument("--out-png", default="", help="If --plot-after-run, output PNG path (optional).")
    p_run.add_argument("--dpi", type=int, default=150)
    p_run.add_argument("--fps", type=int, default=30)
    p_run.add_argument("--mode", required=False, type=str, default="speed")
    p_run.set_defaults(func=run_infer_and_save)

    # plot
    p_plot = subparsers.add_parser("plot", help="Plot GT vs Pred from saved .npz")
    p_plot.add_argument("--inp", default="./save2.npz")
    p_plot.add_argument("--out", default="./save2.png")
    p_plot.add_argument("--dpi", type=int, default=150)
    p_plot.set_defaults(func=plot_saved)

    return parser


def main():
    parser = build_cli()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
