import os
import csv
import threading
import time
from typing import Any
import numpy as np

class NumpyCSVLogger:
    """
    用法:
        logger = NumpyCSVLogger("out.csv", mode="w")
        logger.log(np.array([1, 2, 3]), np.random.randn(2, 2))
        logger.close()
    """
    def __init__(self, path: str, mode: str = "a", delimiter: str = ",",
                 thread_safe: bool = True, add_timestamp_to_filename: bool = False):
        """
        path: CSV 文件路径
        mode: 'a' 追加, 'w' 覆盖
        delimiter: 分隔符
        thread_safe: 是否加锁，适合多线程写入
    # 若 add_timestamp_to_filename 为 True，会在文件名中插入时间戳（见下）；
    # 不再支持在每行中自动插入时间戳（若需要，请显式把时间作为第一个参数传入 log()）。
        """
        self.path = path
        # 如果要求在文件名中加入时间戳，则把时间戳插入到扩展名前面
        if add_timestamp_to_filename:
            try:
                from datetime import datetime
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                base, ext = os.path.splitext(self.path)
                # 如果没有扩展名，直接追加
                if ext:
                    self.path = f"{base}_{ts}{ext}"
                else:
                    self.path = f"{self.path}_{ts}"
            except Exception:
                # 保持原名（不应阻塞日志创建）
                pass
        # open using the (possibly modified) self.path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._fh = open(self.path, mode, newline="")          # newline="" 交给 csv 控制行尾
        self._writer = csv.writer(self._fh, delimiter=delimiter, lineterminator="\n")  # 明确行尾为 \n
        self._lock = threading.Lock() if thread_safe else None

    def log(self, *arrays: Any) -> None:
        """
        接受一个或多个 numpy 参数；会被扁平化后依次写成一行。
        例如: log(np.array([1,2]), np.array([[3,4],[5,6]]))
        行内容: 1,2,3,4,5,6\n
        """
        row = []
        for a in arrays:
            a = np.asarray(a)                     # 支持标量/列表/numpy
            flat = a.ravel()                      # 扁平化为一维
            row.extend(flat.tolist())             # 转为 Python 标量，便于 csv 写出
        # 如果调用者想记录时间戳，请显式把时间作为第一个参数传入 log()
        if self._lock:
            with self._lock:
                self._writer.writerow(row)
                self._fh.flush()
        else:
            self._writer.writerow(row)
            self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    # 支持 with 语法
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        self.close()
