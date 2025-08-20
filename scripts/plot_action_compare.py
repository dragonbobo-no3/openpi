#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import matplotlib

# 无显示环境下自动用无界面后端，避免 Tk/Qt 报错
if not os.environ.get("DISPLAY", ""):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


def load_csv(path, delimiter=","):
    arr = np.loadtxt(path, delimiter=delimiter, dtype=np.float64)
    if arr.ndim == 1:  # 单行情况变成 (1, D)
        arr = arr[None, :]
    return arr


def main():
    ap = argparse.ArgumentParser(description="Plot columns from action & inference_sended_action CSVs")
    ap.add_argument("--pred_action", default="action.csv", help="path to action csv")
    ap.add_argument("--send_action", default="inference_sended_action.csv", help="path to inference_sended_action csv")
    ap.add_argument("--delimiter", default=",", help="CSV delimiter (default: ,)")
    ap.add_argument("--save", default="", help="save figure to file (e.g., out.png). Empty to show window.")
    args = ap.parse_args()

    action = load_csv(args.pred_action, args.delimiter)
    infer = load_csv(args.send_action, args.delimiter)

    if action.shape[1] != infer.shape[1]:
        raise ValueError(f"列数不一致：action {action.shape}, inference {infer.shape}")

    n_act, D = action.shape
    n_inf = infer.shape[0]

    # 画图：每列一个子图（两列排版）
    ncols = 2
    nrows = (D + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 1.8 * nrows), squeeze=False)
    axes = axes.ravel()

    x_act = np.arange(n_act)
    x_inf = np.arange(n_inf)

    for j in range(D):
        ax = axes[j]
        ax.plot(x_act, action[:, j], label="action", linewidth=1.0)
        ax.plot(x_inf, infer[:, j], label="inference_sended_action", linewidth=1.0, alpha=0.85)
        ax.set_title(f"Column {j}")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)

    # 去掉多余空子图
    for k in range(D, len(axes)):
        fig.delaxes(axes[k])

    fig.suptitle("Action vs Inference Sent Action", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"[OK] saved to {os.path.abspath(args.save)}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
