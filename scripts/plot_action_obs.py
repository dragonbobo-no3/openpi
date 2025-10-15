#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python3 scripts/plot_action_obs.py --action /home/test/jemotor/openpi/logs/action_0819_2_cameras.csv --obs /home/test/jemotor/openpi/logs/obs_0819_2_cameras.csv --k 30
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_csv(path):
    # 无表头、全数字、逗号分隔
    arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if arr.ndim == 1:  # 单行时变成 (1, D)
        arr = arr[None, :]
    return arr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", default="action.csv")
    parser.add_argument("--obs", default="obs.csv")
    parser.add_argument("--k", type=int, default=50, help="1 个 obs 对应 k 个 action")
    parser.add_argument("--save", default="", help="保存图片文件名（留空则弹窗显示）")
    args = parser.parse_args()

    action = load_csv(args.action)
    obs = load_csv(args.obs)

    if action.shape[1] != obs.shape[1]:
        raise ValueError(f"列数不一致: action {action.shape}, obs {obs.shape}")

    n_act, D = action.shape
    n_obs = obs.shape[0]

    # 如果长度与 k 不严格匹配，尝试从数据推断 k
    if n_act != args.k * n_obs:
        if n_obs > 0 and n_act % n_obs == 0:
            k = n_act // n_obs
            print(f"[WARN] n_act({n_act}) != k({args.k})*n_obs({n_obs}), 自动改用 k={k}")
        else:
            # 无法整除：退化为在每 50 步位置用散点画 obs，不做重复对齐
            k = args.k
            print(f"[WARN] 无法整除对齐（n_act={n_act}, n_obs={n_obs}, k={k}），将仅在每 {k} 步位置标出 obs。")
    else:
        k = args.k

    # 将 obs 扩展/对齐到 action 时间轴
    if n_act == k * n_obs:
        obs_aligned = np.repeat(obs, k, axis=0)   # (n_act, D)
        x_obs_mark = np.arange(n_obs) * k         # 标记点位置
    else:
        obs_aligned = None
        x_obs_mark = np.arange(n_obs) * k

    # 画图：每列一张子图
    ncols = D
    nrows = int(np.ceil(ncols / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 1.8 * nrows), squeeze=False)
    ax_list = axes.ravel()

    x_act = np.arange(n_act)

    for j in range(D):
        ax = ax_list[j]
        # action 曲线（高频）
        ax.plot(x_act, action[:, j], label="action", linewidth=1.0)

        # obs 对齐后的曲线或仅打标
        if obs_aligned is not None:
            ax.plot(x_act, obs_aligned[:, j], label="obs (×k重复)", linewidth=1.0, alpha=0.8)
            # 同时在每个 obs 原始采样点加一个标记便于观察
            ax.plot(x_obs_mark, obs[:, j], linestyle="none", marker="o", markersize=3, alpha=0.9, label="obs 点")
        else:
            # 只能在每 k 步标出 obs（可能越界，过滤一下）
            valid = x_obs_mark < n_act
            ax.plot(x_obs_mark[valid], obs[valid, j], linestyle="none", marker="o", markersize=3, alpha=0.9, label="obs 点")

        ax.set_title(f"Column {j}")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    # 多余子图去掉
    for j in range(D, len(ax_list)):
        fig.delaxes(ax_list[j])

    fig.suptitle(f"Action vs Obs (k={k})", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"[OK] saved to {os.path.abspath(args.save)}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
