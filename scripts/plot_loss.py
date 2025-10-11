#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


LOSS_RE = re.compile(r'loss=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')

def parse_rows(fp):
    """
    解析 CSV 日志，返回 (steps, losses) 列表
    支持两列：step, info ；info 中包含 'loss=...'
    """
    steps, losses = [], []

    # 先尝试按表头解析
    fp.seek(0)
    reader = csv.DictReader(fp)
    used_dict_reader = False
    if reader.fieldnames and 'step' in reader.fieldnames and 'info' in reader.fieldnames:
        used_dict_reader = True
        for i, row in enumerate(reader, start=1):
            try:
                step = int(str(row['step']).strip())
                info = str(row['info'])
            except Exception:
                continue
            m = LOSS_RE.search(info)
            if m:
                steps.append(step)
                losses.append(float(m.group(1)))

    if not steps:
        # 回退为无表头的两列表
        fp.seek(0)
        reader2 = csv.reader(fp)
        for i, row in enumerate(reader2, start=1):
            # 跳过可能的表头
            if i == 1 and row and 'step' in row[0].lower():
                continue
            if len(row) < 2:
                continue
            try:
                step = int(str(row[0]).strip())
            except Exception:
                continue
            info = str(row[1])
            m = LOSS_RE.search(info)
            if m:
                steps.append(step)
                losses.append(float(m.group(1)))

    # 排序以防乱序
    pairs = sorted(zip(steps, losses), key=lambda x: x[0])
    steps, losses = [p[0] for p in pairs], [p[1] for p in pairs]
    return steps, losses


def main():
    parser = argparse.ArgumentParser(
        description="Plot loss vs step from training log CSV (columns: step, info with 'loss=...')."
    )
    parser.add_argument("file", type=Path, help="log CSV file path")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="output image path (e.g., loss.png). If omitted, show the plot interactively.")
    args = parser.parse_args()

    if not args.file.exists():
        raise SystemExit(f"File not found: {args.file}")

    with args.file.open("r", encoding="utf-8", newline="") as fp:
        steps, losses = parse_rows(fp)

    if not steps:
        raise SystemExit("No valid 'loss=' entries found in the log.")

    plt.figure(figsize=(7, 4))
    plt.plot(steps, losses)#, marker='o', linewidth=1)
    plt.title("Loss vs Step")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved plot to: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
