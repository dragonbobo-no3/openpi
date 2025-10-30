#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import collections
import os, json, threading, time, re
from collections import deque
from typing import List, Tuple, Optional, Any, Dict
from datetime import datetime
from queue import SimpleQueue, Empty
import pynput

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray, Header

from cv_bridge import CvBridge
import cv2
import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training import config as _config


# ====================== 小工具 ======================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name.strip('/'))


def stamp_to_ns(stamp) -> int:
    # 假设 stamp 有 sec / nanosec，且可能为 0
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _dtype_and_channels(encoding: str):
    """根据 encoding 返回 (numpy.dtype, channels)。常见编码覆盖。"""
    enc = encoding.lower()
    if enc in ("mono8", "8uc1"):
        return np.uint8, 1
    if enc in ("mono16", "16uc1"):
        return np.uint16, 1
    if enc in ("32fc1",):
        return np.float32, 1
    if enc in ("rgb8", "bgr8"):
        return np.uint8, 3
    if enc in ("rgba8", "bgra8"):
        return np.uint8, 4
    # 其他编码可按需扩展
    raise ValueError(f"Unsupported image encoding: {encoding}")


def _image_to_numpy(msg: Image) -> np.ndarray:
    """将 ROS2 Image 解到 HWC numpy。保留原始通道顺序（rgb8 就是 RGB，bgr8 就是 BGR）。"""
    dtype, ch = _dtype_and_channels(msg.encoding)
    # 从 data 构造一维 buffer
    buf = np.frombuffer(msg.data, dtype=dtype)
    # 每行 stride 的元素数（step 是字节数）
    row_elems = msg.step // np.dtype(dtype).itemsize
    # 先按 (H, row_elems) reshape，再切到有效列宽（width*ch）
    arr = buf.reshape(msg.height, row_elems)
    arr = arr[:, : msg.width * ch]
    if ch == 1:
        return arr.reshape(msg.height, msg.width, 1)  # 统一输出 HWC
    else:
        return arr.reshape(msg.height, msg.width, ch)


def _pack_depth_u16_to_rgb8(depth_u16: np.ndarray,
                            order: str = "HI_LO",
                            b_fill: int = 0) -> np.ndarray:
    """
    depth_u16: (H,W,1) uint16 -> (H,W,3) uint8
    order: "HI_LO" 用高 8 位到 R、低 8 位到 G；"LO_HI" 相反
    """
    if depth_u16.dtype != np.uint16:
        raise ValueError("depth_u16 must be uint16")
    depth_u16 = depth_u16[..., 0]  # (H,W)
    hi = ((depth_u16 >> 8) & 0xFF).astype(np.uint8)
    lo = (depth_u16 & 0xFF).astype(np.uint8)
    if order.upper() == "HI_LO":
        r, g = hi, lo
    elif order.upper() == "LO_HI":
        r, g = lo, hi
    else:
        raise ValueError("order must be 'HI_LO' or 'LO_HI'")
    b = np.full_like(r, np.uint8(b_fill))
    return np.stack([r, g, b], axis=-1)  # (H,W,3)


def transform_ros2msg_2_np(
        picks: List[Tuple[int, Any]],
        state_idx: int,
        color_idx: List[int],
        depth_idx: List[int],
        pack_depth_2_rgb8: bool,
) -> Dict:
    """
    将 ros2 消息转换为模型需要的形式：
      - state: JointState.position -> np.ndarray (N,)
      - color images: Image -> np.ndarray HWC，dtype 由 encoding 决定（rgb8/bgr8/rgba8/...）
      - depth images: Image -> uint16 (H,W,1)，或按开关打包为 rgb8 (H,W,3)
    说明：该函数假定 picks[i] = (timestamp_or_idx, ros2_msg)
    """
    obs_dict = {"state": None, "images": {}}

    # -------- state --------
    js = picks[state_idx][1]
    if not isinstance(js, JointState):
        raise TypeError(f"state_idx points to {type(js)}, expected JointState")
    if js.position is None or len(js.position) == 0:
        raise ValueError("JointState.position is empty")
    obs_dict["state"] = np.asarray(js.position, dtype=np.float32)

    # -------- color images --------
    for k, idx in enumerate(color_idx):
        msg = picks[idx][1]
        if not isinstance(msg, Image):
            raise TypeError(f"color_idx[{k}] points to {type(msg)}, expected sensor_msgs.msg.Image")
        img = _image_to_numpy(msg)  # HWC, dtype~=encoding
        obs_dict["images"][f"camera{k}"] = img

    # -------- depth images --------
    for k, idx in enumerate(depth_idx):
        msg = picks[idx][1]
        if not isinstance(msg, Image):
            raise TypeError(f"depth_idx[{k}] points to {type(msg)}, expected sensor_msgs.msg.Image")
        depth_arr = _image_to_numpy(msg)  # HWC
        # 仅当是 16 位单通道深度时，才按开关进行打包；否则原样返回
        enc = msg.encoding.lower()
        if enc in ("mono16", "16uc1") and depth_arr.shape[-1] == 1:
            if pack_depth_2_rgb8:
                depth_rgb = _pack_depth_u16_to_rgb8(depth_arr)
                obs_dict["images"][f"camera{k}_depth"] = depth_rgb
            else:
                obs_dict["images"][f"camera{k}_depth"] = depth_arr  # (H,W,1) uint16
        else:
            # 如为 32FC1 或其他深度格式，直接保留（H,W,1）
            obs_dict["images"][f"camera{k}_depth"] = depth_arr

    return obs_dict


# ====================== 高效多流对齐器 ======================
class MultiStreamAligner:
    """
    无锁摄取 → 工作线程独占本地窗口：
      - 回调 put_nowait 到 ingest 队列
      - step():
        1) drain 各 ingest → 本地 times / msgs（保持单调；乱序帧丢弃）
        2) t_ref = min(latest_i)
        3) 对每路推进“只前进”游标到 <= target 的最大位置，并在它与后一位择近；误差<=tol_i
        4) 成功返回 (t_ref, picks)，并剪枝到选中位置
      - 支持 max_window_ns 以限制窗口大小
    日志增强：
      - 统计入队/乱序丢弃/窗口裁剪/对齐成功率
      - 记录失败原因（空流/超容差）及细节
      - 限频打印（避免刷屏）
    """

    def __init__(
            self,
            num_streams: int,
            tolerances_ns: List[int],
            offsets_ns: Optional[List[int]] = None,
            max_window_ns: Optional[int] = None,
            *,
            logger: Optional[Any] = None,
            name: str = "aligner",
            debug: bool = False,
            log_every: int = 200,  # 每尝试 N 次打印一条汇总
            log_period_s: float = 5.0,  # 或每隔 T 秒打印一条汇总（两者取其一满足）
            ref_indices: Optional[List[int]] = None,  # <-- 新增
            non_consuming_indices: Optional[List[int]] = None,  # <-- 新增
    ):
        assert num_streams == len(tolerances_ns), "tolerances_ns length must match num_streams"
        self.N = int(num_streams)
        self.tolerances_ns = list(map(int, tolerances_ns))
        self.offsets_ns = list(map(int, offsets_ns)) if offsets_ns else [0] * self.N
        self.max_window_ns = int(max_window_ns) if max_window_ns else None

        # 参考流（用于计算 t_ref），默认用所有流；否则只用传入的索引
        if ref_indices is None:
            self.ref_indices = list(range(self.N))
        else:
            # 过滤非法索引并去重、排序
            self.ref_indices = sorted({int(i) for i in ref_indices if 0 <= int(i) < self.N})
            assert self.ref_indices, "ref_indices cannot be empty"

        # 非消耗流：对齐成功后不删除所选帧（让低频流可复用上次样本）
        non_consuming_set = set()
        if non_consuming_indices:
            non_consuming_set = {int(i) for i in non_consuming_indices if 0 <= int(i) < self.N}
        self.consume_mask = [False if i in non_consuming_set else True for i in range(self.N)]

        self.ingest: List[SimpleQueue] = [SimpleQueue() for _ in range(self.N)]
        self.times: List[List[int]] = [[] for _ in range(self.N)]
        self.msgs: List[List[Any]] = [[] for _ in range(self.N)]
        self.cursor: List[int] = [-1] * self.N

        # ---- 日志 & 统计 ----
        self.debug = bool(debug)
        self.log_every = int(max(1, log_every))
        self.log_period_s = float(max(0.5, log_period_s))
        self._last_log_t = time.monotonic()
        self.log = logger if logger is not None else logging.getLogger(f"{__name__}.{name}")

        # 统计计数
        self.stats: Dict[str, Any] = {
            "enq": [0] * self.N,  # 入队计数
            "drain": [0] * self.N,  # drain 总取出计数
            "append": [0] * self.N,  # 追加到窗口的计数（乱序会被丢）
            "drop_ooo": [0] * self.N,  # 乱序丢弃计数
            "prune": [0] * self.N,  # 窗口裁剪丢弃计数
            "consumed": [0] * self.N,  # <-- 新增：对齐成功后剪枝删除的数量
            "step_attempt": 0,  # 尝试对齐次数
            "step_success": 0,  # 成功对齐次数
            "last_fail": "",  # 最近失败原因
            "last_fail_detail": None,  # 最近失败细节
        }

        self._dbg("initialized: N=%d, max_window_ns=%s, tolerances=%s, offsets=%s",
                  self.N, str(self.max_window_ns), self.tolerances_ns, self.offsets_ns)

    # ---------- 日志工具 ----------
    def _dbg(self, msg: str, *args):
        if self.debug and hasattr(self.log, "debug"):
            try:
                self.log.debug(msg % args if args else msg)
            except Exception:
                pass

    def _info(self, msg: str, *args):
        if hasattr(self.log, "info"):
            try:
                self.log.info(msg % args if args else msg)
            except Exception:
                pass

    def _warn(self, msg: str, *args):
        if hasattr(self.log, "warn"):
            try:
                self.log.warn(msg % args if args else msg)
            except Exception:
                if hasattr(self.log, "warning"):
                    try:
                        self.log.warning(msg % args if args else msg)
                    except Exception:
                        pass

    def _err(self, msg: str, *args):
        if hasattr(self.log, "error"):
            try:
                self.log.error(msg % args if args else msg)
            except Exception:
                pass

    def _maybe_log_summary(self):
        """限频打印汇总统计。"""
        t = time.monotonic()
        need = (self.stats["step_attempt"] % self.log_every == 0) or ((t - self._last_log_t) >= self.log_period_s)
        if not need:
            return
        self._last_log_t = t
        hit = (self.stats["step_success"] / self.stats["step_attempt"]) if self.stats["step_attempt"] else 0.0
        # 每路窗口长度
        lens = [len(ti) for ti in self.times]
        self._info(
            "[align] attempts=%d, success=%d, hit=%.1f%%, window_len=%s, "
            "drop_ooo=%s, prune=%s, consumed=%s, last_fail=%s %s",
            self.stats["step_attempt"], self.stats["step_success"], 100.0 * hit,
            lens, self.stats["drop_ooo"], self.stats["prune"], self.stats["consumed"],
            self.stats["last_fail"],
            f"detail={self.stats['last_fail_detail']}" if self.stats["last_fail_detail"] else "",
        )

    # ---------- 接口 ----------
    def put_nowait(self, i: int, t_ns: int, msg: Any):
        """无阻塞入队。极端异常直接丢帧。"""
        try:
            self.ingest[i].put_nowait((t_ns, msg))
            self.stats["enq"][i] += 1
            # 低成本调试：偶尔打点
            if self.debug and (self.stats["enq"][i] % (self.log_every // 2 or 1) == 0):
                self._dbg("enqueued stream[%d] total=%d last_ts=%d", i, self.stats["enq"][i], t_ns)
        except Exception as e:
            self._warn("enqueue failed stream[%d]: %s", i, e)

    def _drain_one(self, i: int):
        qi = self.ingest[i]
        ti = self.times[i]
        mi = self.msgs[i]
        last = ti[-1] if ti else -1

        drained = 0
        appended = 0
        drop_ooo = 0

        while True:
            try:
                t_ns, msg = qi.get_nowait()
                drained += 1
            except Empty:
                break
            if t_ns >= last:
                ti.append(t_ns)
                mi.append(msg)
                last = t_ns
                appended += 1
            else:
                # 乱序：丢弃
                drop_ooo += 1

        self.stats["drain"][i] += drained
        self.stats["append"][i] += appended
        self.stats["drop_ooo"][i] += drop_ooo

        if self.debug and drained:
            self._dbg("drain stream[%d]: drained=%d, appended=%d, drop_ooo=%d, window_len=%d, last_ts=%d",
                      i, drained, appended, drop_ooo, len(ti), last)

        # 限定窗口长度以控内存
        if self.max_window_ns and len(ti) >= 2:
            cutoff = last - self.max_window_ns
            drop_k, n = 0, len(ti)
            while drop_k < n and ti[drop_k] < cutoff:
                drop_k += 1
            if drop_k > 0:
                del ti[:drop_k]
                del mi[:drop_k]
                self.cursor[i] -= drop_k
                if self.cursor[i] < -1:
                    self.cursor[i] = -1
                self.stats["prune"][i] += drop_k
                self._dbg("prune stream[%d]: pruned=%d, new_window_len=%d, cursor=%d",
                          i, drop_k, len(ti), self.cursor[i])

    def _drain(self):
        for i in range(self.N):
            self._drain_one(i)

    @staticmethod
    def _advance_and_pick(times: List[int], cur: int, target: int) -> Tuple[int, int]:
        n = len(times)
        # 推进到最后一个 <= target
        while cur + 1 < n and times[cur + 1] <= target:
            cur += 1
        # 在 cur 与 cur+1 中择近
        pick = 0 if cur < 0 else cur
        if cur + 1 < n:
            left_ts, right_ts = times[pick], times[cur + 1]
            if abs(right_ts - target) < abs(left_ts - target):
                pick = cur + 1
        return pick, cur

    def step(self) -> Optional[Tuple[int, List[Tuple[int, Any]]]]:
        """尝试做一次对齐；成功返回 (t_ref, picks)，否则返回 None。"""
        self._drain()
        self.stats["step_attempt"] += 1

        # 若某路还没数据
        # latest = []
        for i in range(self.N):
            if not self.times[i]:
                self.stats["last_fail"] = f"waiting_stream_{i}"
                self.stats["last_fail_detail"] = {"stream": i, "reason": "empty"}
                self._maybe_log_summary()
                return None
            # latest.append(self.times[i][-1])

        # 2) 计算 t_ref：只考虑“参考流”（相机流）中的 latest
        latest_refs = [self.times[i][-1] for i in self.ref_indices]
        t_ref = min(latest_refs)

        picks_idx: List[int] = [-1] * self.N
        picks: List[Tuple[int, Any]] = [None] * self.N  # type: ignore

        # 对各路与 target 的误差
        deltas = [None] * self.N  # type: ignore

        for i in range(self.N):
            target = t_ref + self.offsets_ns[i]
            pick_i, cur_after = self._advance_and_pick(self.times[i], self.cursor[i], target)
            err = abs(self.times[i][pick_i] - target)
            deltas[i] = int(err)
            if err > self.tolerances_ns[i]:
                # 容差失败
                self.stats["last_fail"] = "tolerance_exceeded"
                self.stats["last_fail_detail"] = {
                    "stream": i,
                    "target_ns": int(target),
                    "picked_ts_ns": int(self.times[i][pick_i]),
                    "error_ns": int(err),
                    "tolerance_ns": int(self.tolerances_ns[i]),
                }
                self._dbg("align fail: stream[%d] err_ns=%d > tol_ns=%d (target=%d, pick_ts=%d)",
                          i, err, self.tolerances_ns[i], target, self.times[i][pick_i])
                self._maybe_log_summary()
                return None
            picks_idx[i] = pick_i
            picks[i] = (self.times[i][pick_i], self.msgs[i][pick_i])
            self.cursor[i] = cur_after

        # 成功：剪枝到选择点之后
        for i in range(self.N):
            k = picks_idx[i]
            if k < 0:
                continue
            if self.consume_mask[i]:
                # 消耗：删除到选中位置（包含）
                del self.times[i][:k + 1]
                del self.msgs[i][:k + 1]
                self.cursor[i] -= (k + 1)
                if self.cursor[i] < -1:
                    self.cursor[i] = -1
                self.stats["consumed"][i] += (k + 1)
            else:
                # 非消耗：不删，允许后续重用同一帧
                # 仅依靠 _drain_one 的时间窗口裁剪来控内存
                pass

        self.stats["step_success"] += 1
        self.stats["last_fail"] = ""
        self.stats["last_fail_detail"] = None

        if self.debug:
            self._dbg("align OK: t_ref=%d, deltas_ns=%s, windows=%s",
                      t_ref, deltas, [len(ti) for ti in self.times])

        self._maybe_log_summary()
        return t_ref, picks

    # ---------- 公共统计接口 ----------
    def metrics(self) -> Dict[str, Any]:
        """返回一个可读的统计快照（不会清零）。"""
        hit = (self.stats["step_success"] / self.stats["step_attempt"]) if self.stats["step_attempt"] else 0.0
        return {
            "streams": self.N,
            "window_len": [len(ti) for ti in self.times],
            "enqueued": list(self.stats["enq"]),
            "drained": list(self.stats["drain"]),
            "appended": list(self.stats["append"]),
            "dropped_out_of_order": list(self.stats["drop_ooo"]),
            "pruned_by_window": list(self.stats["prune"]),
            "consumed": list(self.stats["consumed"]),
            "attempts": int(self.stats["step_attempt"]),
            "success": int(self.stats["step_success"]),
            "hit_ratio": hit,
            "last_fail": self.stats["last_fail"],
            "last_fail_detail": self.stats["last_fail_detail"],
        }

    def reset_metrics(self):
        """清空统计计数（窗口内容不变）。"""
        for k in ("enq", "drain", "append", "drop_ooo", "prune"):
            self.stats[k] = [0] * self.N
        self.stats["step_attempt"] = 0
        self.stats["step_success"] = 0
        self.stats["last_fail"] = ""
        self.stats["last_fail_detail"] = None


# ====================== ROS2 节点 ======================

class Manager(Node):
    """
    订阅多路彩色/深度/关节/触觉；保存线程使用 min-latest + 单调游标 对齐；锁外转换/写盘。
    修复点：
      - 0 时间戳/无时间戳 → 回退本地时钟
      - 过滤空话题
      - 图像/深度 BEST_EFFORT；关节/触觉 RELIABLE
    """

    def __init__(self):
        super().__init__('manager')

        # ---------- 参数 ----------
        # 话题
        self.declare_parameter('color_topics', [])
        self.declare_parameter('depth_topics', [])
        self.declare_parameter('color_topics_csv',
                               '/camera_01/color/image_raw,/camera_03/color/image_raw,/camera_04/color/image_raw,')
        self.declare_parameter('depth_topics_csv',
                               '/camera_01/depth/image_raw,/camera_03/depth/image_raw,/camera_04/depth/image_raw,')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('tactile_topic', '/tactile_data')
        self.declare_parameter('checkpoint_dir', '/home/kleist/Documents/Model/cloud_server/1017_pi05_test/60000/')

        # 频率与容差（ms）
        self.declare_parameter('rate_hz', 30.0)
        self.declare_parameter('image_tolerance_ms', 15.0)  # ≤ 半帧
        self.declare_parameter('joint_tolerance_ms', 15.0)
        self.declare_parameter('tactile_tolerance_ms', 60.0)

        # 窗口与目录
        self.declare_parameter('queue_seconds', 2.0)
        self.declare_parameter('save_dir', os.path.expanduser('~/ros2_logs/sensor_logger'))
        self.declare_parameter('session_name', '')
        self.declare_parameter('save_depth', False)

        # 其他
        self.declare_parameter('use_ros_time', True)
        self.declare_parameter('color_jpeg_quality', 95)
        self.declare_parameter('do_calculate_hz', True)
        self.declare_parameter('stats_window_s', 5.0)  # 统计窗口长度(秒)
        self.declare_parameter('stats_log_period_s', 2.0)  # 日志输出周期(秒)
        self.declare_parameter('episode_idx', 0)
        self.declare_parameter('mode', 0)

        # 偏置（ms）
        self.declare_parameter('color_offsets_ms', [])
        self.declare_parameter('depth_offsets_ms', [])
        self.declare_parameter('joint_offset_ms', 0.0)
        self.declare_parameter('tactile_offset_ms', 0.0)

        p = self.get_parameter

        # 读取话题参数
        color_topics = list(p('color_topics').value or [])
        depth_topics = list(p('depth_topics').value or [])
        if not color_topics:
            csv = p('color_topics_csv').value or ''
            if csv.strip():
                color_topics = [s.strip() for s in csv.split(',') if s.strip()]
        if not depth_topics:
            csv = p('depth_topics_csv').value or ''
            if csv.strip():
                depth_topics = [s.strip() for s in csv.split(',') if s.strip()]

        # 过滤空字符串话题
        color_topics = [t.strip() for t in color_topics if t and t.strip()]
        depth_topics = [t.strip() for t in depth_topics if t and t.strip()]

        self.color_topics: List[str] = color_topics
        self.depth_topics: List[str] = depth_topics
        self.joint_topic: str = p('joint_state_topic').value
        self.tactile_topic: str = p('tactile_topic').value

        # 频率/容差
        self.rate_hz: float = float(p('rate_hz').value)
        image_tol_ns = int(float(p('image_tolerance_ms').value) * 1e6)
        joint_tol_ns = int(float(p('joint_tolerance_ms').value) * 1e6)
        tactile_tol_ns = int(float(p('tactile_tolerance_ms').value) * 1e6)

        # 窗口/目录
        self.queue_seconds: float = float(p('queue_seconds').value)
        self.save_dir: str = p('save_dir').value
        session_name: str = p('session_name').value
        self.save_depth: bool = bool(p('save_depth').value)
        self.use_ros_time: bool = bool(p('use_ros_time').value)
        self.jpeg_quality: int = int(p('color_jpeg_quality').value)

        # 偏置
        color_offsets_ms = list(p('color_offsets_ms').value or [])
        depth_offsets_ms = list(p('depth_offsets_ms').value or [])
        joint_offset_ms = float(p('joint_offset_ms').value or 0.0)
        tactile_offset_ms = float(p('tactile_offset_ms').value or 0.0)

        # 统计频率
        self.stats_window_s: float = float(p('stats_window_s').value)
        self.stats_log_period_s: float = float(p('stats_log_period_s').value)
        self.do_calculate_hz: bool = bool(p('do_calculate_hz').value)
        self._attempt_win = 0
        self._success_win = 0
        self._save_times = deque()  # 保存成功的时间戳(秒, perf_counter)
        self._last_rate_log_t = time.perf_counter()  # 上次打印统计的时间

        self.episode_idx: int = int(p('episode_idx').value)
        self.mode: int = int(p('mode').value)
        self.checkpoint_dir: str = p('checkpoint_dir').value

        if not session_name:
            session_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(self.save_dir, sanitize(session_name))
        ensure_dir(self.session_dir)

        # 相机目录
        self.cam_count = len(self.color_topics)
        self.cam_dirs = []
        for i in range(self.cam_count):
            cam_name = sanitize(self.color_topics[i])
            base = os.path.join(self.session_dir, f'cam_{i:02d}_{cam_name}')
            ensure_dir(base)
            ensure_dir(os.path.join(base, 'color'))
            if self.save_depth and (i < len(self.depth_topics) and self.depth_topics[i]):
                ensure_dir(os.path.join(base, 'depth'))
            self.cam_dirs.append(base)

        self.meta_dir = os.path.join(self.session_dir, 'meta')
        ensure_dir(self.meta_dir)

        self.get_logger().info(f"Session: {self.session_dir}")
        self.get_logger().info(f"Color topics: {self.color_topics}")
        self.get_logger().info(f"Depth  topics: {self.depth_topics}")
        self.get_logger().info(f"Joint: {self.joint_topic}, Tactile: {self.tactile_topic}")
        self.get_logger().info(
            f"rate={self.rate_hz}Hz, tol(img/joint/tactile)=[{image_tol_ns / 1e6:.1f},{joint_tol_ns / 1e6:.1f},{tactile_tol_ns / 1e6:.1f}]ms"
        )

        # ---------- 对齐器 ----------
        C = len(self.color_topics)
        D = len(self.depth_topics) if self.save_depth else 0
        self._idx_color = list(range(C))
        self._idx_depth = list(range(C, C + D))
        self._idx_joint = C + D
        self._idx_tact = C + D + 1
        num_streams = C + D + 2

        tolerances_ns: List[int] = []
        tolerances_ns += [image_tol_ns] * C
        tolerances_ns += [image_tol_ns] * D
        tolerances_ns += [joint_tol_ns, tactile_tol_ns]

        offsets_ns: List[int] = []
        if color_offsets_ms and len(color_offsets_ms) == C:
            offsets_ns += [int(ms * 1e6) for ms in color_offsets_ms]
        else:
            offsets_ns += [0] * C
        if D > 0:
            if depth_offsets_ms and len(depth_offsets_ms) == D:
                offsets_ns += [int(ms * 1e6) for ms in depth_offsets_ms]
            else:
                offsets_ns += [0] * D
        offsets_ns += [int(joint_offset_ms * 1e6), int(tactile_offset_ms * 1e6)]

        max_window_ns = int(self.queue_seconds * 1e9)

        camera_refs = self._idx_color + self._idx_depth  # 只用相机做 t_ref
        non_consuming = [self._idx_tact]
        self.aligner = MultiStreamAligner(
            num_streams=num_streams,
            tolerances_ns=tolerances_ns,
            offsets_ns=offsets_ns,
            max_window_ns=max_window_ns,
            logger=self.get_logger(),
            ref_indices=camera_refs,
            non_consuming_indices=non_consuming,
        )

        # ---------- 订阅（区分 QoS） ----------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        for i, topic in enumerate(self.color_topics):
            self.create_subscription(Image, topic, self._mk_color_cb(i), reliable_qos)

        for j in range(D):
            topic = self.depth_topics[j]
            if topic and self.save_depth:
                self.create_subscription(Image, topic, self._mk_depth_cb(j), reliable_qos)

        # 关节/触觉用 RELIABLE（若上游是 BEST_EFFORT，可改成 sensor_qos）
        self.create_subscription(JointState, self.joint_topic, self._joint_cb, reliable_qos)
        self.create_subscription(Float32MultiArray, self.tactile_topic, self._tactile_cb, reliable_qos)

        self.pub_joint_cmd = self.create_publisher(JointState, '/joint_cmd', reliable_qos)

        # 工具
        self.bridge = CvBridge()

        # === 录制开关与键盘监听 ===
        self._record_enabled = False  # 是否允许调用 _save_once（默认暂停，按 Ctrl 开始）
        self._kb_listener = None
        self._last_pause_log = 0.0  # 限流输出“暂停中”的日志

        # 可选：提示
        self.get_logger().info("Keyboard control: right Alt = start/resume, right Ctrl = stop")

        # 优先尝试 pynput 全局键盘监听（无需焦点；在无 X/Wayland 的纯控制台可能不可用）
        try:
            from pynput import keyboard as _kb
            self._pynput_kb = _kb
            self._kb_listener = _kb.Listener(on_press=self._on_key_press)
            self._kb_listener.daemon = True
            self._kb_listener.start()
        except Exception as e:
            # 不能监听就仅提示（录制永远保持默认 False，或你可手动在代码里改为 True）
            self.get_logger().warn(
                f"Keyboard control unavailable (install pynput or ensure GUI session). "
                f"Recording remains paused until Ctrl event can be captured. detail={e}"
            )

        # 保存线程
        self.sample_idx = 0
        self._stop_evt = threading.Event()
        if self.mode == 0:
            self.get_logger().error("Select a mode!!!")
        elif self.mode == 1:
            self.worker = threading.Thread(target=self._save_loop, name='logger-aligner', daemon=False)
            self.worker.start()
        elif self.mode == 2:
            config = _config.get_config("pi05_agileX")
            self.policy = _policy_config.create_trained_policy(config, self.checkpoint_dir)
            self.worker = threading.Thread(target=self._inference_loop, name='logger-aligner', daemon=False)
            self.worker.start()
        self.actions_queue = collections.deque(maxlen=100)

    def _on_key_press(self, key):
        """全局键盘：Ctrl = start/resume，Right Arrow = stop"""
        try:
            kb = getattr(self, "_pynput_kb", None)
            if kb is None:
                return

            # Ctrl：开始/继续
            if key == kb.Key.alt_r:
                if not self._record_enabled:
                    self._record_enabled = True
                    self.get_logger().info("Recording ENABLED by right Alt")
                else:
                    # 已经在录就保持不变（幂等）
                    pass

            # 右方向键：停止
            elif key == kb.Key.ctrl_r:
                if self._record_enabled:
                    self._record_enabled = False
                    self.episode_idx += 1
                    self.get_logger().info("Recording DISABLED by right Ctrl")
        except Exception as e:
            self.get_logger().warn(f"Keyboard handler error: {e}")

    # ---------- 时间戳回退：0/无时间戳 → 本地时钟 ----------
    def _ns_from_header_or_clock(self, header) -> int:
        try:
            s = int(header.stamp.sec)
            ns = int(header.stamp.nanosec)
            if s != 0 or ns != 0:
                return s * 1_000_000_000 + ns
        except Exception:
            pass
        return self.get_clock().now().nanoseconds

    # ---------- 回调：无锁摄取 ----------
    def _mk_color_cb(self, i: int):
        def _cb(msg: Image):
            if self.use_ros_time:
                t_ns = self._ns_from_header_or_clock(msg.header)
            else:
                t_ns = self.get_clock().now().nanoseconds
            self.aligner.put_nowait(self._idx_color[i], t_ns, msg)
            # self.get_logger().info(f"receive image:{i} at {time.perf_counter()}")

        return _cb

    def _mk_depth_cb(self, j: int):
        def _cb(msg: Image):
            if self.use_ros_time:
                t_ns = self._ns_from_header_or_clock(msg.header)
            else:
                t_ns = self.get_clock().now().nanoseconds
            self.aligner.put_nowait(self._idx_depth[j], t_ns, msg)
            # self.get_logger().info(f"receive depth:{j} at {time.perf_counter()}")

        return _cb

    def _joint_cb(self, msg: JointState):
        # Joint 用 header（若 0 则回退）
        t_ns = self._ns_from_header_or_clock(msg.header)
        self.aligner.put_nowait(self._idx_joint, t_ns, msg)

    def _tactile_cb(self, msg: Float32MultiArray):
        # 触觉通常无 header，用接收时钟
        t_ns = self.get_clock().now().nanoseconds
        self.aligner.put_nowait(self._idx_tact, t_ns, msg)

    def _save_loop(self):
        period = 1.0 / max(1e-6, self.rate_hz)
        next_t = time.perf_counter()
        print(rclpy.ok())
        while rclpy.ok() and not self._stop_evt.is_set():
            now = time.perf_counter()
            if now < next_t:
                time.sleep(max(0.0, next_t - now))
            next_t += period
            try:
                # 若暂停录制：做一次对齐推进以“排水”，但不保存，避免堆积
                if not self._record_enabled:
                    _ = self.aligner.step()
                    # 限流打印（每 2 秒一次）
                    if (now - self._last_pause_log) > 2.0:
                        self.get_logger().debug("Paused: press Ctrl to start/resume, Right Arrow to stop.")
                        self._last_pause_log = now
                    continue

                # === 新增：记录一次尝试（可选） ===
                if self.do_calculate_hz:
                    self._attempt_win += 1

                out = self.aligner.step()
                if out is None:
                    continue

                # === 新增：记录一次成功（可选） ===
                if self.do_calculate_hz:
                    self._success_win += 1

                t_ref, picks = out
                self._save_once(t_ref, picks)

                # === 新增：保存成功后更新统计（可选） ===
                if self.do_calculate_hz:
                    self._on_saved_stats()
            except Exception as e:
                self.get_logger().error(f"save loop error: {e}")

    def _inference_loop(self):
        period = 1.0 / max(1e-6, self.rate_hz)
        next_t = time.perf_counter()
        print(rclpy.ok())
        while rclpy.ok() and not self._stop_evt.is_set():
            now = time.perf_counter()
            if now < next_t:
                time.sleep(max(0.0, next_t - now))
            next_t += period
            try:
                if self.actions_queue:
                    self._publish_once()
                    continue
                out = self.aligner.step()
                if out is None:
                    continue
                t_ref, picks = out
                obs = transform_ros2msg_2_np(picks, self._idx_joint, self._idx_color, self._idx_depth, False)
                print(obs.keys())
                exit(1)
                result = self.policy.infer(obs)
                for action in result['actions']:
                    self.actions_queue.append(action)
                self._publish_once()
            except Exception as e:
                self.get_logger().error(f"save loop error: {e}")

    def _publish_once(self):
        action_to_send = self.actions_queue.popleft()
        js = JointState()
        js.header = Header(stamp=self._now(), frame_id='')
        js.name = [f'joint{i + 1}' for i in range(7)]
        js.position = action_to_send[0:7]
        self.pub_joint_cmd.publish(js)
        return

        # === 新增：统计函数 ===

    def _on_saved_stats(self):
        now = time.perf_counter()
        self._save_times.append(now)

        # 滑动窗口：只保留最近 stats_window_s 秒的时间戳
        cutoff = now - self.stats_window_s
        while self._save_times and self._save_times[0] < cutoff:
            self._save_times.popleft()

        # 计算窗口 FPS 和瞬时 FPS
        win_fps = 0.0
        inst_fps = 0.0
        if len(self._save_times) >= 2:
            dt_win = self._save_times[-1] - self._save_times[0]
            if dt_win > 0:
                win_fps = (len(self._save_times) - 1) / dt_win
            dt_inst = self._save_times[-1] - self._save_times[-2]
            if dt_inst > 0:
                inst_fps = 1.0 / dt_inst

        # 到时间就打印一次统计，并清空尝试/成功的窗口计数
        if (now - self._last_rate_log_t) >= self.stats_log_period_s:
            hit_ratio = (self._success_win / self._attempt_win) if self._attempt_win > 0 else 0.0
            self.get_logger().info(
                f"[save stats] inst_fps={inst_fps:.2f}, "
                f"win({self.stats_window_s:.1f}s)_fps={win_fps:.2f}, "
                f"align_hit_ratio={hit_ratio:.1%} "
                f"(attempts={self._attempt_win}, success={self._success_win})"
            )
            self._attempt_win = 0
            self._success_win = 0
            self._last_rate_log_t = now

    def _save_once(self, t_ref: int, picks: List[Tuple[int, Any]]):
        # self.get_logger().info(f"save once at {time.perf_counter()}")
        return
        idx = self.sample_idx
        self.sample_idx += 1

        C = len(self._idx_color)
        D = len(self._idx_depth)
        color_picks = picks[0:C]
        depth_picks = picks[C:C + D] if D > 0 else []
        joint_ts, js_msg = picks[C + D]
        tact_ts, tact_msg = picks[C + D + 1]

        meta = {
            'index': idx,
            't_ref_ns': int(t_ref),
            'color': [],
            'depth': [],
            'joint': {'stamp_ns': int(joint_ts)},
            'tactile': {'stamp_ns': int(tact_ts)},
        }

        # --- 保存彩色 ---
        for cam_i, (t_ns, msg) in enumerate(color_picks):
            # cv_bridge 转换（锁外）
            if getattr(msg, "encoding", "") != 'bgr8':
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                img = self.bridge.imgmsg_to_cv2(msg)
            cam_dir = self.cam_dirs[cam_i]
            fn = f'color_{idx:06d}.jpg'
            fp = os.path.join(cam_dir, 'color', fn)
            cv2.imwrite(fp, img, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)])
            meta['color'].append({
                'cam_index': cam_i,
                'topic': self.color_topics[cam_i],
                'stamp_ns': int(t_ns),
                'file': os.path.relpath(fp, self.session_dir),
            })

        # --- 保存深度 ---
        if self.save_depth and depth_picks:
            for dep_i, (t_ns, msg) in enumerate(depth_picks):
                cv_img = self.bridge.imgmsg_to_cv2(msg)
                scale_info = None
                if cv_img.dtype in (np.float32, np.float64):
                    cv_mm = np.clip(cv_img * 1000.0, 0, 65535).astype(np.uint16)
                    out = cv_mm
                    scale_info = 'float_meter_to_uint16_mm'
                elif cv_img.dtype == np.uint16:
                    out = cv_img
                else:
                    out = cv_img.astype(np.uint16)

                cam_dir = self.cam_dirs[dep_i] if dep_i < len(self.cam_dirs) else os.path.join(self.session_dir,
                                                                                               f'cam_dep_{dep_i:02d}')
                ensure_dir(os.path.join(cam_dir, 'depth'))
                fn = f'depth_{idx:06d}.png'
                fp = os.path.join(cam_dir, 'depth', fn)
                cv2.imwrite(fp, out)
                meta['depth'].append({
                    'depth_index': dep_i,
                    'topic': self.depth_topics[dep_i],
                    'stamp_ns': int(t_ns),
                    'file': os.path.relpath(fp, self.session_dir),
                    'note': scale_info,
                })

        # --- JointState ---
        js: JointState = js_msg
        meta['joint'].update({
            'name': list(js.name),
            'position': [float(x) for x in js.position],
            'velocity': [float(x) for x in js.velocity],
            'effort': [float(x) for x in js.effort],
        })

        # --- 触觉 ---
        tm: Float32MultiArray = tact_msg
        meta['tactile'].update({
            'data': [float(x) for x in tm.data]
        })

        # --- 写 meta ---
        meta_fp = os.path.join(self.meta_dir, f'meta_{idx:06d}.json')
        with open(meta_fp, 'w') as f:
            json.dump(meta, f, indent=2)

        # 周期日志
        if idx % int(max(1, self.rate_hz)) == 0:
            self.get_logger().info(f"saved idx={idx} @t_ref={t_ref} ns")

    # ---------- 关闭 ----------
    def destroy_node(self):
        # 停止键盘监听
        try:
            if self._kb_listener is not None:
                self._kb_listener.stop()
        except Exception:
            pass

        self._stop_evt.set()
        try:
            if self.worker.is_alive():
                self.worker.join(timeout=3.0)
        except Exception:
            pass
        return super().destroy_node()


# ====================== main：多线程执行器 ======================

def main(args=None):
    rclpy.init(args=args)
    node = Manager()
    try:
        from rclpy.executors import MultiThreadedExecutor
        executor = MultiThreadedExecutor(num_threads=os.cpu_count() or 4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
