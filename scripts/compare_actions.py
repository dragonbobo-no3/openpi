#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法：
  python compare_actions.py --file_a a.npz --file_b b.npz --out diff.png --key pred_actions
"""
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Compare actions between two npz files and plot per-dim curves + error stats.")
    parser.add_argument('--file_a', required=True, help='First npz file (A)')
    parser.add_argument('--file_b', required=True, help='Second npz file (B)')
    parser.add_argument('--out', default='diff.png', help='Output plot file')
    parser.add_argument('--key', default='pred_actions', help='Key to compare (e.g., pred_actions or gt_actions)')
    args = parser.parse_args()

    if not os.path.exists(args.file_a) or not os.path.exists(args.file_b):
        print(f"File not found: {args.file_a} or {args.file_b}", file=sys.stderr)
        sys.exit(1)

    data_a = np.load(args.file_a, allow_pickle=True)
    data_b = np.load(args.file_b, allow_pickle=True)

    if args.key not in data_a or args.key not in data_b:
        print(f"Key '{args.key}' not found in one of the files.", file=sys.stderr)
        sys.exit(1)

    arr_a = data_a[args.key]
    arr_b = data_b[args.key]

    # 统一成 (T, D) 形状；如是 (T, 1, D) 则取[:, 0, :]
    if arr_a.ndim == 3:
        arr_a = arr_a[:, 0, :]
    if arr_b.ndim == 3:
        arr_b = arr_b[:, 0, :]
    if arr_a.ndim == 1:
        arr_a = arr_a[:, None]
    if arr_b.ndim == 1:
        arr_b = arr_b[:, None]

    if arr_a.shape != arr_b.shape:
        raise ValueError(f"Shape mismatch: {arr_a.shape} vs {arr_b.shape}")

    T, D = arr_a.shape

    # 误差
    diff = arr_a - arr_b                      # (T, D)
    l2_error = np.linalg.norm(diff, axis=1)   # (T,) 每帧 L2
    abs_error = np.abs(diff)                  # (T, D)

    # 逐维与总体统计
    mean_abs_per_dim = abs_error.mean(axis=0)         # (D,)
    max_abs_per_dim = abs_error.max(axis=0)           # (D,)
    overall_mean_abs = abs_error.mean()
    overall_max_abs = abs_error.max()
    max_t, max_d = np.unravel_index(abs_error.argmax(), abs_error.shape)

    # 打印统计
    print("===== Absolute Error Stats =====")
    print(f"Overall mean(|A-B|): {overall_mean_abs:.6f}")
    print(f"Overall max(|A-B|):  {overall_max_abs:.6f}  at (t={max_t}, dim={max_d})")
    print("Per-dimension mean(|A-B|):")
    for d in range(D):
        print(f"  dim {d:02d}: mean={mean_abs_per_dim[d]:.6f}, max={max_abs_per_dim[d]:.6f}")
    print("===== L2 Error Stats (per frame) =====")
    print(f"Mean L2 error: {l2_error.mean():.6f}")
    print(f"Max  L2 error: {l2_error.max():.6f}  at frame {l2_error.argmax()}")

    # 画图：前 D 行是「逐维 A 与 B 的曲线」，最后一行是 L2 误差
    fig_h = max(2.2 * (D + 1), 4.5)
    fig, axes = plt.subplots(D + 1, 1, figsize=(12, fig_h), sharex=True)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    # 逐维：A/B 同轴对比
    x = np.arange(T)
    for d in range(D):
        ax = axes[d]
        ax.plot(x, arr_a[:, d], label=f'A dim {d}')
        ax.plot(x, arr_b[:, d], linestyle='--', label=f'B dim {d}')
        # 在标题里简要标注该维误差统计
        ax.set_title(f'Dim {d}  (mean|Δ|={mean_abs_per_dim[d]:.4g}, max|Δ|={max_abs_per_dim[d]:.4g})')
        ax.set_ylabel('value')
        ax.legend(loc='upper right')

    # 最后一行：每帧 L2
    ax_last = axes[-1]
    ax_last.plot(x, l2_error, label='L2 error per frame')
    ax_last.set_title('Total L2 error')
    ax_last.set_ylabel('L2')
    ax_last.set_xlabel('Frame')
    ax_last.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"[PLOT] saved curves -> {args.out}")
    # 可按需保留/去掉
    plt.show()

if __name__ == "__main__":
    main()
