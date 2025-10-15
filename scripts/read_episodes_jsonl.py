#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
读取episodes.jsonl文件，统计前n行 length的累加和
用法：
  python read_episodes_jsonl.py /path/to/episodes.jsonl -n 2
  # 也支持 .gz 压缩：
  python sum_lengths.py /path/to/episodes.jsonl.gz -n 100
"""

import argparse
import json
import gzip
from pathlib import Path
from typing import TextIO

def open_maybe_gzip(path: Path) -> TextIO:
    """支持 .gz 与普通文本文件"""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="统计 JSONL 文件前 n 行的 length 累加和")
    ap.add_argument("path", type=Path, help="episodes.jsonl 文件路径（支持 .gz）")
    ap.add_argument("-n", "--num-lines", type=int, required=True, help="前 n 行")
    ap.add_argument("--key", default="length", help="要累加的字段名（默认 length）")
    args = ap.parse_args()

    if args.num_lines <= 0:
        print(0)
        return

    total = 0
    processed = 0

    with open_maybe_gzip(args.path) as f:
        for line_no, line in enumerate(f, start=1):
            if processed >= args.num_lines:
                break
            line = line.strip()
            if not line:
                # 空行也算在“前 n 行”里吗？若要严格计算每一行，可把 processed += 1 放到这里
                continue
            obj = json.loads(line)  # 期望每行都是合法 JSON
            val = obj.get(args.key)
            if val is None:
                # 没有该键就当 0 处理；如需报错则改为 raise KeyError
                val = 0
            total += int(val)
            processed += 1

    print(total)

if __name__ == "__main__":
    main()
