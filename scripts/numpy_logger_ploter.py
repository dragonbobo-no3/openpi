#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_columns(spec: str, ncols: int) -> List[int]:
    """
    解析列选择：
      - "all"：所有列
      - "0,1,5"：指定列
      - "0:7"：半开区间 [0,7)
      - 混合写法："0:7,12,13:16"
    """
    if spec.lower() == "all":
        return list(range(ncols))
    cols = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, b = part.split(":")
            a = int(a) if a else 0
            b = int(b) if b else ncols
            cols.update(range(a, min(b, ncols)))
        else:
            idx = int(part)
            if 0 <= idx < ncols:
                cols.add(idx)
    return sorted(cols)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    if window > len(x):
        window = len(x)
    # 同步长度输出（居中对齐），两端用 NaN 填补，再前向填充
    y = pd.Series(x).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    return y


def main():
    ap = argparse.ArgumentParser(
        description="Visualize CSV rows produced by NumpyCSVLogger as time series."
    )
    ap.add_argument("--csv", required=True, help="Path to CSV file.")
    ap.add_argument("--delimiter", default=",", help="CSV delimiter (default: ',').")
    ap.add_argument("--skip-rows", type=int, default=0, help="Skip first N rows.")
    ap.add_argument(
        "--columns",
        default="all",
        help='Columns to plot. e.g. "all" | "0,1,5" | "0:7" | "0:7,12,13:16"',
    )
    ap.add_argument("--downsample", type=int, default=1, help="Stride downsample factor.")
    ap.add_argument("--ma", type=int, default=1, help="Moving average window size.")
    ap.add_argument(
        "--normalize",
        choices=["none", "zscore", "minmax"],
        default="none",
        help="Per-column normalization for visualization.",
    )
    ap.add_argument("--title", default="", help="Figure title.")
    ap.add_argument("--output", default="", help="Save to image (e.g., out.png). Leave empty to show window.")
    args = ap.parse_args()

    # 读取 CSV（假定没有表头）
    try:
        df = pd.read_csv(
            args.csv,
            header=None,
            delimiter=args.delimiter,
            skiprows=args.skip_rows,
            engine="python",
        )
    except Exception as e:
        print(f"[ERR] Failed to read CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # 仅保留数值列（全 NaN 的列会被丢弃）
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    if df.shape[1] == 0:
        print("[ERR] No numeric columns found.", file=sys.stderr)
        sys.exit(1)

    # 选择列
    col_idx = parse_columns(args.columns, df.shape[1])
    if not col_idx:
        print(f"[ERR] No valid columns selected from 0..{df.shape[1]-1}", file=sys.stderr)
        sys.exit(1)

    data = df.iloc[:, col_idx].to_numpy()

    # 降采样
    if args.downsample > 1:
        data = data[::args.downsample, :]

    # 平滑
    if args.ma > 1:
        for j in range(data.shape[1]):
            data[:, j] = moving_average(data[:, j], args.ma)

    # 归一化（仅影响显示）
    if args.normalize != "none":
        if args.normalize == "zscore":
            mean = np.nanmean(data, axis=0, keepdims=True)
            std = np.nanstd(data, axis=0, keepdims=True)
            std[std == 0] = 1.0
            data = (data - mean) / std
        elif args.normalize == "minmax":
            dmin = np.nanmin(data, axis=0, keepdims=True)
            dmax = np.nanmax(data, axis=0, keepdims=True)
            span = dmax - dmin
            span[span == 0] = 1.0
            data = (data - dmin) / span

    # 画图：一张图上画多条线（每列一条）
    x = np.arange(len(data))
    plt.figure(figsize=(10, 5))
    for j, c in enumerate(col_idx):
        plt.plot(x, data[:, j], label=f"col{c}")
    plt.xlabel("row index")
    plt.ylabel("value")
    if args.title:
        plt.title(args.title)
    # 列很多时就别显示 legend 了，太挤
    if len(col_idx) <= 20:
        plt.legend(ncol=2)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"[OK] Saved figure to {args.output}")
    else:
        # plt.show()
        plt.savefig("out.png", dpi=150)


if __name__ == "__main__":
    main()
