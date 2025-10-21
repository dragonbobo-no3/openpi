#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法：
python compare_actions.py --file_a a.npz --file_b b.npz --out diff.png
"""
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser(description="Compare actions between two npz files and plot error curve.")
    parser.add_argument('--file_a', required=True, help='First npz file (A)')
    parser.add_argument('--file_b', required=True, help='Second npz file (B)')
    parser.add_argument('--out', default='diff.png', help='Output plot file')
    parser.add_argument('--key', default='pred_actions', help='Key to compare (pred_actions or gt_actions)')
    args = parser.parse_args()

    data_a = np.load(args.file_a, allow_pickle=True)
    data_b = np.load(args.file_b, allow_pickle=True)
    arr_a = data_a[args.key]
    arr_b = data_b[args.key]

    # 只对前两维(T, action_dim)做对比，如果有三维则默认取[:,0,:]
    if arr_a.ndim == 3:
        arr_a = arr_a[:, 0, :]
    if arr_b.ndim == 3:
        arr_b = arr_b[:, 0, :]

    if arr_a.shape != arr_b.shape:
        raise ValueError(f"Shape mismatch: {arr_a.shape} vs {arr_b.shape}")

    diff = arr_a - arr_b
    l2_error = np.linalg.norm(diff, axis=1)  # 每帧L2误差
    abs_error = np.abs(diff)  # 每帧每维绝对误差

    print(f"Mean L2 error: {l2_error.mean():.4f}")
    print(f"Max L2 error: {l2_error.max():.4f}")
    print(f"Max L2 error idx: {l2_error.argmax()}")
    print(f"Mean abs error (per dim): {abs_error.mean(axis=0)}")

    # 所有action维度误差用subplot画在一张图
    T, D = arr_a.shape
    fig, axes = plt.subplots(D+1, 1, figsize=(10, 3*(D+1)), sharex=True)
    # 保证axes为list
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for d in range(D):
        ax = axes[d]
        ax.plot(abs_error[:, d], label=f'Abs error dim {d}')
        ax.set_ylabel('Abs error')
        ax.set_title(f'Action abs error (dim {d})')
        ax.legend()
    # 最后一张画总L2误差
    ax_last = axes[-1]
    ax_last.plot(l2_error, label='L2 error per frame', color='black')
    ax_last.set_ylabel('L2 error')
    ax_last.set_title('Total L2 error')
    ax_last.legend()
    ax_last.set_xlabel('Frame')
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"[PLOT] saved all dims diff curve -> {args.out}")
    plt.show()

if __name__ == "__main__":
    main()
