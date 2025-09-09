# csv_logger.py
import os
import csv
import atexit
from datetime import datetime
from threading import Lock

# 全局状态（本进程内复用）
_CSV_LOCK = Lock()
_CSV_FILE = None          # type: ignore
_CSV_WRITER = None        # type: ignore
_CSV_PATH = None

def _ensure_csv_open(log_dir: str = "logs",
                     filename_prefix: str = "train_log",
                     date_fmt: str = "%Y%m%d"):
    """若还未打开日志文件，则按当前日期创建一个唯一的 CSV，并写入表头。"""
    global _CSV_FILE, _CSV_WRITER, _CSV_PATH
    if _CSV_WRITER is not None:
        return

    os.makedirs(log_dir, exist_ok=True)
    date_str = datetime.now().strftime(date_fmt)
    base = os.path.join(log_dir, f"{filename_prefix}_{date_str}.csv")
    path = base
    if os.path.exists(path):
        i = 1
        while os.path.exists(f"{base[:-4]}-{i}.csv"):
            i += 1
        path = f"{base[:-4]}-{i}.csv"

    _CSV_FILE = open(path, "a", newline="", encoding="utf-8")
    _CSV_WRITER = csv.writer(_CSV_FILE)
    _CSV_WRITER.writerow(["step", "info"])  # 表头
    _CSV_FILE.flush()
    _CSV_PATH = path

def _close_csv():
    global _CSV_FILE, _CSV_WRITER, _CSV_PATH
    if _CSV_FILE is not None:
        try:
            _CSV_FILE.flush()
            _CSV_FILE.close()
        finally:
            _CSV_FILE = None
            _CSV_WRITER = None
            _CSV_PATH = None

atexit.register(_close_csv)

def log_to_csv(info_str: str, step: int):
    """
    将一条日志写入 CSV。
    - 首次调用：按日期创建 logs/train_log_YYYYMMDD.csv（若存在则加 -1/-2…）。
    - 后续调用：追加写入。每行格式为: step, info（完整字符串）。
    """
    global _CSV_WRITER, _CSV_FILE
    with _CSV_LOCK:
        _ensure_csv_open()
        _CSV_WRITER.writerow([int(step), info_str])
        _CSV_FILE.flush()
