#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配合 check_data.py 使用
check_data.py 检测哪些数据有问题，最简单的方法就是直接使用其他数据集替换，
a 是被替换的 index，b 是替换来源的 index

示例：
python src/lerobot/replace_single_episode.py --root /home/kleist/Documents/Database/test_0928_100_v2 \
  --a 83 --b 99 --chunk chunk-000
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime

# 新增：pyarrow 用于修改 parquet 内的 episode_index 列
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as _e:
    pa = None
    pq = None


def copy_file(src: Path, dst: Path, dry: bool):
    if not src.exists():
        raise FileNotFoundError(f"源文件不存在: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"[DRY] copy {src} -> {dst}")
    else:
        shutil.copy2(src, dst)
        print(f"copied {src} -> {dst}")


def backup_file(p: Path, dataset_root: Path, backup_root: Path, dry: bool):
    if not p.exists():
        return
    rel = p.relative_to(dataset_root)
    bak = backup_root / rel
    bak.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"[DRY] backup {p} -> {bak}")
    else:
        shutil.copy2(p, bak)
        print(f"backup  {p} -> {bak}")


def load_jsonl(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"JSONL 不存在: {p}")
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl_atomic(p: Path, rows):
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)


def _deepcopy(obj):
    return json.loads(json.dumps(obj))


def _patch_stats_episode_index(row: dict, target_idx: int):
    """
    专门用于 episodes_stats.jsonl:
    - 修改顶层 row["episode_index"]
    - 同步修改 row["stats"]["episode_index"] 的 {min,max,mean,std} 等字段
    """
    row["episode_index"] = int(target_idx)

    stats = row.get("stats", {})
    epi = stats.get("episode_index")
    if not isinstance(epi, dict):
        return row

    def _set_list(field, value):
        if field not in epi:
            return
        v = epi[field]

        def _overwrite(x, val):
            if isinstance(x, list):
                if len(x) == 0:
                    return [val]
                if isinstance(x[0], list):
                    return [[val for _ in x[0]] for _ in x]
                else:
                    return [val for _ in x]
            return [val]

        epi[field] = _overwrite(v, value)

    _set_list("min", int(target_idx))
    _set_list("max", int(target_idx))
    _set_list("mean", float(target_idx))
    if "std" in epi:
        _set_list("std", 0.0)

    stats["episode_index"] = epi
    row["stats"] = stats
    return row


def replace_meta_episode(jsonl_path: Path, a: int, b: int,
                         backup_root: Path, dataset_root: Path, dry: bool):
    """
    在 jsonl 中找到 episode_index==a 的记录，用 episode_index==b 的记录替换；
    若 a 不存在，则在合适位置插入从 b 拷贝而来的记录（并将其 episode_index 改为 a）。
    对 episodes_stats.jsonl，会额外同步修改 stats.episode_index 的统计值。
    """
    rows = load_jsonl(jsonl_path)

    def _find_idx(target: int):
        for i, r in enumerate(rows):
            try:
                if int(r.get("episode_index")) == target:
                    return i
            except Exception:
                continue
        return None

    row_a_idx = _find_idx(a)
    row_b_idx = _find_idx(b)
    if row_b_idx is None:
        raise ValueError(f"{jsonl_path.name}: 找不到 episode_index={b}")

    new_row = _deepcopy(rows[row_b_idx])
    if jsonl_path.name == "episodes_stats.jsonl":
        new_row = _patch_stats_episode_index(new_row, a)
    else:
        new_row["episode_index"] = a

    if row_a_idx is None:
        insert_pos = 0
        for i, r in enumerate(rows):
            try:
                if int(r.get("episode_index")) > a:
                    break
            except Exception:
                pass
            insert_pos = i + 1
        new_rows = rows[:insert_pos] + [new_row] + rows[insert_pos:]
        action = "insert"
    else:
        new_rows = list(rows)
        new_rows[row_a_idx] = new_row
        action = "replace"

    if dry:
        print(f"[DRY] {jsonl_path.name}: {action} episode_index={a} ← {b}")
    else:
        backup_file(jsonl_path, dataset_root, backup_root, dry=False)
        save_jsonl_atomic(jsonl_path, new_rows)
        print(f"updated {jsonl_path.name}: episode {a} ← {b} ({action})")


# 新增：复制完 parquet 后，把其中的 episode_index 列改为 a
def patch_parquet_episode_index(parquet_path: Path, target_idx: int, dry: bool):
    if pa is None or pq is None:
        raise RuntimeError("需要 pyarrow 来修改 parquet 的列，请先安装：pip install pyarrow")

    if not parquet_path.exists():
        raise FileNotFoundError(f"parquet 文件不存在: {parquet_path}")

    # 先用 meta 拿行数，便于 dry-run 显示
    pf = pq.ParquetFile(parquet_path)
    nrows = pf.metadata.num_rows

    if dry:
        print(f"[DRY] patch parquet {parquet_path} : set episode_index={target_idx} for {nrows} rows")
        return

    table = pq.read_table(parquet_path)
    names = table.schema.names
    if "episode_index" not in names:
        raise KeyError(f"{parquet_path} 不含列 'episode_index'，无法修改")

    col_idx = names.index("episode_index")
    old_col = table.column(col_idx)
    target_type = old_col.type  # 保持原始整数类型（通常是 int64/int32）

    # 构造同长度常量列
    const_arr = pa.array([int(target_idx)] * table.num_rows, type=target_type)
    new_table = table.set_column(col_idx, "episode_index", const_arr)

    tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    pq.write_table(new_table, tmp_path)  # 如需指定压缩：compression="snappy"
    tmp_path.replace(parquet_path)
    print(f"patched episode_index in {parquet_path} -> {target_idx} (rows={nrows})")


def main():
    ap = argparse.ArgumentParser(description="把索引 a 的内容用索引 b 的内容替换（data/videos/meta 三处），并修正 parquet 内 episode_index")
    ap.add_argument("--root", required=True, help="数据集根目录（包含 data/, videos/, meta/）")
    ap.add_argument("--a", type=int, required=True, help="目标 episode 索引（被覆盖/插入）")
    ap.add_argument("--b", type=int, required=True, help="来源 episode 索引（作为模板）")
    ap.add_argument("--chunk", default="chunk-000", help="分块目录名，默认 chunk-000")
    ap.add_argument("--no-backup", action="store_true", help="不做备份（默认会在 .backup_replace_episode 下备份）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要做什么，不执行")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    chunk = args.chunk
    a, b = args.a, args.b

    data_dir = root / "data" / chunk
    videos_dir = root / "videos" / chunk
    meta_dir = root / "meta"

    ep_a = f"episode_{a:06d}"
    ep_b = f"episode_{b:06d}"

    backup_root = root / ".backup_replace_episode" / (
        datetime.now().strftime("%Y%m%d_%H%M%S") if not args.no_backup else "NO_BACKUP"
    )

    print(f"ROOT: {root}")
    print(f"操作: {a} <- {b}  （覆盖/插入 A 的内容为 B）")
    print(f"dry-run: {args.dry_run}, backup: {not args.no_backup}")

    # 1) parquet：先备份 A 的旧文件，再复制 B → A，最后修 A 的 parquet 内部 episode_index
    src_parquet = data_dir / f"{ep_b}.parquet"
    dst_parquet = data_dir / f"{ep_a}.parquet"
    if not args.no_backup:
        backup_file(dst_parquet, root, backup_root, dry=args.dry_run)
    copy_file(src_parquet, dst_parquet, dry=args.dry_run)
    # 关键：修正 parquet 内部的 episode_index 列
    patch_parquet_episode_index(dst_parquet, a, dry=args.dry_run)

    # 2) videos
    if videos_dir.exists():
        for cam_dir in sorted(p for p in videos_dir.iterdir() if p.is_dir()):
            src_mp4 = cam_dir / f"{ep_b}.mp4"
            dst_mp4 = cam_dir / f"{ep_a}.mp4"
            if src_mp4.exists():
                if not args.no_backup:
                    backup_file(dst_mp4, root, backup_root, dry=args.dry_run)
                copy_file(src_mp4, dst_mp4, dry=args.dry_run)
            else:
                print(f"[WARN] 源视频缺失（跳过）: {src_mp4}")
    else:
        print(f"[WARN] 未找到视频目录: {videos_dir}")

    # 3) meta（注意：episodes_stats.jsonl 需要同步改两处 episode_index）
    for name in ("episodes.jsonl", "episodes_stats.jsonl"):
        path = meta_dir / name
        replace_meta_episode(path, a, b, backup_root, root, dry=args.dry_run)

    print("完成。建议检查：")
    print(f"  ls {data_dir}/{ep_a}.parquet")
    print(f"  parquet-tools head -n 1 {data_dir}/{ep_a}.parquet | grep episode_index")
    print(f"  ls {videos_dir}/**/{ep_a}.mp4")
    print(f"  grep '\"episode_index\": {a}' {meta_dir}/episodes*.jsonl")


if __name__ == "__main__":
    main()
