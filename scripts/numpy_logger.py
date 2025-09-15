import os
import csv
import threading
from typing import Any
import numpy as np

class NumpyCSVLogger:
    """
    用法:
        logger = NumpyCSVLogger("out.csv", mode="w")
        logger.log(np.array([1, 2, 3]), np.random.randn(2, 2))
        logger.close()
    """
    def __init__(self, path: str, mode: str = "a", delimiter: str = ",", thread_safe: bool = True):
        """
        path: CSV 文件路径
        mode: 'a' 追加, 'w' 覆盖
        delimiter: 分隔符
        thread_safe: 是否加锁，适合多线程写入
        """
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, mode, newline="")          # newline="" 交给 csv 控制行尾
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
