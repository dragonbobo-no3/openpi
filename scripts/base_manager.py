#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, re, logging, shutil
from collections import deque
from typing import List, Tuple, Optional, Any, Dict
from datetime import datetime
from queue import SimpleQueue, Empty

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from .ros2_qos import reliable_qos, sensor_qos

from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray
from common.msg import OculusInitJointState

# ====================== 一阶低通滤波 ======================

class TorqueFilter:
    def __init__(self, fs: float, tau_sec: float):
        dt = 1.0 / fs
        self.alpha = dt / (tau_sec + dt)
        self.y = None

    def update(self, x: float) -> float:
        if self.y is None:
            self.y = x
        else:
            self.y = self.y + self.alpha * (x - self.y)
        return self.y

# ====================== 小工具 ======================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name.strip('/'))


# ====================== 多流对齐器 ======================
class MultiStreamAligner:
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
            log_every: int = 200,
            log_period_s: float = 5.0,
            ref_indices: Optional[List[int]] = None,
            non_consuming_indices: Optional[List[int]] = None,
            optional_indices: Optional[List[int]] = None,
    ):
        assert num_streams == len(tolerances_ns), "tolerances_ns length must match num_streams"
        self.N = int(num_streams)
        self.tolerances_ns = list(map(int, tolerances_ns))
        self.offsets_ns = list(map(int, offsets_ns)) if offsets_ns else [0] * self.N
        self.max_window_ns = int(max_window_ns) if max_window_ns else None

        if ref_indices is None:
            self.ref_indices = list(range(self.N))
        else:
            self.ref_indices = sorted({int(i) for i in ref_indices if 0 <= int(i) < self.N})
            assert self.ref_indices, "ref_indices cannot be empty"

        self.optional_set = set(int(i) for i in (optional_indices or []) if 0 <= int(i) < self.N)
        non_consuming_set = set(int(i) for i in (non_consuming_indices or []) if 0 <= int(i) < self.N)
        self.consume_mask = [False if i in non_consuming_set else True for i in range(self.N)]

        self.ingest: List[SimpleQueue] = [SimpleQueue() for _ in range(self.N)]
        self.times: List[List[int]] = [[] for _ in range(self.N)]
        self.msgs: List[List[Any]] = [[] for _ in range(self.N)]
        self.cursor: List[int] = [-1] * self.N

        self.debug = bool(debug)
        self.log_every = int(max(1, log_every))
        self.log_period_s = float(max(0.5, log_period_s))
        self._last_log_t = time.monotonic()
        self.log = logger if logger is not None else logging.getLogger(f"{__name__}.{name}")

        self.stats: Dict[str, Any] = {
            "enq": [0] * self.N,
            "drain": [0] * self.N,
            "append": [0] * self.N,
            "drop_ooo": [0] * self.N,
            "prune": [0] * self.N,
            "consumed": [0] * self.N,
            "step_attempt": 0,
            "step_success": 0,
            "last_fail": "",
            "last_fail_detail": None,
        }

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

    def _maybe_log_summary(self):
        return
        t = time.monotonic()
        need = (self.stats["step_attempt"] % self.log_every == 0) or ((t - self._last_log_t) >= self.log_period_s)
        if not need: return
        self._last_log_t = t
        hit = (self.stats["step_success"] / self.stats["step_attempt"]) if self.stats["step_attempt"] else 0.0
        lens = [len(ti) for ti in self.times]
        self._info(
            "[align] attempts=%d, success=%d, hit=%.1f%%, window_len=%s, drop_ooo=%s, prune=%s, consumed=%s, last_fail=%s %s",
            self.stats["step_attempt"], self.stats["step_success"], 100.0 * hit,
            lens, self.stats["drop_ooo"], self.stats["prune"], self.stats["consumed"],
            self.stats["last_fail"],
            f"detail={self.stats['last_fail_detail']}" if self.stats["last_fail_detail"] else "",
        )

    def put_nowait(self, i: int, t_ns: int, msg: Any):
        try:
            self.ingest[i].put_nowait((t_ns, msg))
            self.stats["enq"][i] += 1
        except Exception as e:
            self._warn("enqueue failed stream[%d]: %s", i, e)

    def _drain_one(self, i: int):
        qi = self.ingest[i]
        ti = self.times[i]
        mi = self.msgs[i]
        last = ti[-1] if ti else -1
        drained = appended = drop_ooo = 0
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
                drop_ooo += 1
        self.stats["drain"][i] += drained
        self.stats["append"][i] += appended
        self.stats["drop_ooo"][i] += drop_ooo

        if self.max_window_ns and len(ti) >= 2:
            cutoff = last - self.max_window_ns
            drop_k, n = 0, len(ti)
            while drop_k < n and ti[drop_k] < cutoff: drop_k += 1
            if drop_k > 0:
                del ti[:drop_k]
                del mi[:drop_k]
                self.cursor[i] -= drop_k
                if self.cursor[i] < -1: self.cursor[i] = -1
                self.stats["prune"][i] += drop_k

    def _drain(self):
        for i in range(self.N):
            self._drain_one(i)

    @staticmethod
    def _advance_and_pick(times: List[int], cur: int, target: int) -> Tuple[int, int]:
        n = len(times)
        while cur + 1 < n and times[cur + 1] <= target:
            cur += 1
        pick = 0 if cur < 0 else cur
        if cur + 1 < n:
            left_ts, right_ts = times[pick], times[cur + 1]
            if abs(right_ts - target) < abs(left_ts - target):
                pick = cur + 1
        return pick, cur

    def step(self) -> Optional[Tuple[int, List[Optional[Tuple[int, Any]]]]]:
        self._drain()
        self.stats["step_attempt"] += 1

        ref_latest = []
        for i in self.ref_indices:
            if not self.times[i]:
                self.stats["last_fail"] = f"waiting_ref_stream_{i}"
                self.stats["last_fail_detail"] = {"stream": i, "reason": "empty_ref"}
                self._maybe_log_summary()
                return None
            ref_latest.append(self.times[i][-1])
        t_ref = min(ref_latest)

        picks_idx: List[int] = [-1] * self.N
        picks: List[Optional[Tuple[int, Any]]] = [None] * self.N

        for i in range(self.N):
            if (i in self.optional_set) and (not self.times[i]):
                continue
            if not self.times[i]:
                self.stats["last_fail"] = f"waiting_stream_{i}"
                self.stats["last_fail_detail"] = {"stream": i, "reason": "empty"}
                self._maybe_log_summary()
                return None

            target = t_ref + self.offsets_ns[i]
            pick_i, cur_after = self._advance_and_pick(self.times[i], self.cursor[i], target)
            err = abs(self.times[i][pick_i] - target)

            if err > self.tolerances_ns[i]:
                self.stats["last_fail"] = "tolerance_exceeded"
                self.stats["last_fail_detail"] = {
                    "stream": i,
                    "target_ns": int(target),
                    "picked_ts_ns": int(self.times[i][pick_i]),
                    "error_ns": int(err),
                    "tolerance_ns": int(self.tolerances_ns[i]),
                }
                self._maybe_log_summary()
                return None

            picks_idx[i] = pick_i
            picks[i] = (self.times[i][pick_i], self.msgs[i][pick_i])
            self.cursor[i] = cur_after

        for i in range(self.N):
            k = picks_idx[i]
            if k < 0: continue
            if self.consume_mask[i]:
                del self.times[i][:k + 1]
                del self.msgs[i][:k + 1]
                self.cursor[i] -= (k + 1)
                if self.cursor[i] < -1: self.cursor[i] = -1
                self.stats["consumed"][i] += (k + 1)

        self.stats["step_success"] += 1
        self.stats["last_fail"] = ""
        self.stats["last_fail_detail"] = None

        self._maybe_log_summary()
        return t_ref, picks

    # ---------- 公共统计接口 ----------
    def metrics(self) -> Dict[str, Any]:
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
        for k in ("enq", "drain", "append", "drop_ooo", "prune"):
            self.stats[k] = [0] * self.N
        self.stats["step_attempt"] = 0
        self.stats["step_success"] = 0
        self.stats["last_fail"] = ""
        self.stats["last_fail_detail"] = None


# ====================== BaseManager（抽象基类） ======================

class BaseManager(Node):
    """
    抽象基类：参数、订阅、对齐器、保存/元数据逻辑与按键处理均在此；
    但**不自动启动**按键监听或任何线程，由子类决定。
    """

    def __init__(self, node_name: str = 'manager'):
        super().__init__(node_name)

        # ---------- 参数 ----------
        # 话题
        self.declare_parameter('color_topics', [])
        self.declare_parameter('depth_topics', [])
        self.declare_parameter('color_topics_csv',
                               '/camera_03/color/image_raw,/camera_04/color/image_raw,/camera_05/color/image_raw')
        self.declare_parameter('depth_topics_csv',
                               '')

        # joint/tactile：多路 + 兼容单路
        self.declare_parameter('joint_state_topics', [])
        self.declare_parameter('joint_state_topics_csv', '/joint_states_double_arm')
        self.declare_parameter('joint_cmd_topics', [])
        self.declare_parameter('joint_cmd_topics_csv', '/joint_cmd_double_arm')
        self.declare_parameter('tactile_topics', [])
        self.declare_parameter('tactile_topics_csv', '')
        self.declare_parameter('joint_state_topic', '')  # legacy
        self.declare_parameter('joint_cmd_topic', '')  # legacy
        self.declare_parameter('tactile_topic', '')  # legacy

        # 频率与容差（ms）
        self.declare_parameter('rate_hz', 30.0)
        self.declare_parameter('image_tolerance_ms', 22.0)
        self.declare_parameter('joint_tolerance_ms', 10.0)
        self.declare_parameter('tactile_tolerance_ms', 50.0)

        # 窗口与目录
        self.declare_parameter('queue_seconds', 2.0)
        self.declare_parameter('save_dir', os.path.expanduser('~/jemotor/log/'))
        self.declare_parameter('session_name', '')
        self.declare_parameter('save_depth', True)
        self.declare_parameter('overwrite', False)

        # 其他
        self.declare_parameter('use_ros_time', True)
        self.declare_parameter('color_jpeg_quality', 95)
        self.declare_parameter('do_calculate_hz', True)
        self.declare_parameter('stats_window_s', 5.0)
        self.declare_parameter('stats_log_period_s', 2.0)
        self.declare_parameter('episode_idx', 0)

        # 力矩低通滤波配置（默认 7 维）
        self.declare_parameter('effort_filter_enable', True)    # 是否启用力矩滤波
        self.declare_parameter('effort_filter_fs', 50.0)         # <=0 时使用 rate_hz
        self.declare_parameter('effort_filter_tau_sec', 0.10)   # 一阶低通时间常数
        self.declare_parameter('effort_filter_num_channels', 7) # 默认 7 维力矩

        # 偏置（ms）
        self.declare_parameter('color_offsets_ms', [])
        self.declare_parameter('depth_offsets_ms', [])
        self.declare_parameter('joint_offset_ms', 0.0)
        self.declare_parameter('tactile_offset_ms', 0.0)

        p = self.get_parameter

        # 读取话题参数（list/csv 兼容）
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
        color_topics = [t.strip() for t in color_topics if t and t.strip()]
        depth_topics = [t.strip() for t in depth_topics if t and t.strip()]
        self.color_topics: List[str] = color_topics
        self.depth_topics: List[str] = depth_topics

        joint_state_topics = list(p('joint_state_topics').value or [])
        if not joint_state_topics:
            csv = p('joint_state_topics_csv').value or ''
            if csv.strip():
                joint_state_topics = [s.strip() for s in csv.split(',') if s.strip()]
        if not joint_state_topics:
            legacy = p('joint_state_topic').value
            if legacy and str(legacy).strip():
                joint_state_topics = [str(legacy).strip()]
        self.joint_state_topics: List[str] = list(
            dict.fromkeys([t for t in joint_state_topics if t and t.strip()])
        )

        joint_cmd_topics = list(p('joint_cmd_topics').value or [])
        if not joint_cmd_topics:
            csv = p('joint_cmd_topics_csv').value or ''
            if csv.strip():
                joint_cmd_topics = [s.strip() for s in csv.split(',') if s.strip()]
        if not joint_cmd_topics:
            legacy = p('joint_cmd_topic').value
            if legacy and str(legacy).strip():
                joint_cmd_topics = [str(legacy).strip()]
        self.joint_cmd_topics: List[str] = list(dict.fromkeys([t for t in joint_cmd_topics if t and t.strip()]))

        tactile_topics = list(p('tactile_topics').value or [])
        if not tactile_topics:
            csv = p('tactile_topics_csv').value or ''
            if csv.strip():
                tactile_topics = [s.strip() for s in csv.split(',') if s.strip()]
        if not tactile_topics:
            legacy = p('tactile_topic').value
            if legacy and str(legacy).strip():
                tactile_topics = [str(legacy).strip()]
        self.tactile_topics: List[str] = [t for t in tactile_topics if t and t.strip()]

        # 频率/容差
        self.rate_hz: float = float(p('rate_hz').value)
        image_tol_ns = int(float(p('image_tolerance_ms').value) * 1e6)

        def _as_list_ns(param_name: str, n: int) -> List[int]:
            raw = p(param_name).value
            if isinstance(raw, (list, tuple)):
                seq = list(raw)
                if len(seq) == 0:
                    return [0] * n
                if len(seq) == 1 and n > 1:
                    return [int(float(seq[0]) * 1e6)] * n
                assert len(seq) == n, f"{param_name} length must == {n} (got {len(seq)})"
                return [int(float(x) * 1e6) for x in seq]
            else:
                return [int(float(raw) * 1e6)] * n

        # 窗口/目录
        self.queue_seconds: float = float(p('queue_seconds').value)
        self.save_dir: str = p('save_dir').value
        session_name: str = p('session_name').value
        self.save_depth: bool = bool(p('save_depth').value)
        self.overwrite: bool = bool(p('overwrite').value)
        self.use_ros_time: bool = bool(p('use_ros_time').value)
        self.jpeg_quality: int = int(p('color_jpeg_quality').value)

        # 偏置 -> ns 列表
        def _as_list_offset_ns(param_name: str, n: int) -> List[int]:
            raw = p(param_name).value
            if isinstance(raw, (list, tuple)):
                seq = list(raw)
                if len(seq) == 0:
                    return [0] * n
                if len(seq) == 1 and n > 1:
                    return [int(float(seq[0]) * 1e6)] * n
                assert len(seq) == n, f"{param_name} length must == {n} (got {len(seq)})"
                return [int(float(x) * 1e6) for x in seq]
            else:
                return [int(float(raw) * 1e6)] * n

        # 统计
        self.stats_window_s: float = float(p('stats_window_s').value)
        self.stats_log_period_s: float = float(p('stats_log_period_s').value)
        self.do_calculate_hz: bool = bool(p('do_calculate_hz').value)
        self._attempt_win = 0
        self._success_win = 0
        self._save_times = deque()
        self._last_rate_log_t = time.perf_counter()

        # ===== 力矩滤波器初始化（默认 7 维） =====
        self.effort_filter_enable: bool = bool(p('effort_filter_enable').value)
        self.effort_filter_tau_sec: float = float(p('effort_filter_tau_sec').value)
        eff_fs = float(p('effort_filter_fs').value)
        if eff_fs <= 0.0:
            # 如果没有专门指定，就先用对齐输出频率；如果你知道 JointState 是 100Hz，可以在参数里设成 100
            eff_fs = self.rate_hz
        self.effort_filter_fs: float = eff_fs

        self.effort_filter_num_channels: int = int(p('effort_filter_num_channels').value or 7)
        if self.effort_filter_num_channels <= 0:
            self.effort_filter_num_channels = 7

        if self.effort_filter_enable:
            # 默认创建 7 个滤波器，对应 7 维力矩
            self.effort_filters = [
                TorqueFilter(fs=self.effort_filter_fs, tau_sec=self.effort_filter_tau_sec)
                for _ in range(self.effort_filter_num_channels)
            ]
        else:
            self.effort_filters = []

        # 缓存原始 effort，用于 RecorderManager._save_once 里同时保存原始 + 滤波后
        # key: (joint_topic_index k, stamp_ns) -> List[float]
        self._effort_raw_cache = {}
        
        self.episode_idx: int = int(p('episode_idx').value)
        self.frame_idx = 0

        if not session_name:
            session_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(self.save_dir, sanitize(session_name))
        # 确保 save_dir 存在
        ensure_dir(self.save_dir)

        # 如果开启 overwrite，则清空 save_dir 下的所有内容（谨慎）
        if self.overwrite:
            try:
                if os.path.exists(self.save_dir):
                    for name in os.listdir(self.save_dir):
                        path = os.path.join(self.save_dir, name)
                        try:
                            if os.path.islink(path) or os.path.isfile(path):
                                os.remove(path)
                            elif os.path.isdir(path):
                                shutil.rmtree(path)
                        except Exception as e:
                            self.get_logger().warn(f"Failed to remove '{path}' while clearing save_dir: {e}")
                    self.get_logger().info(f"Overwrite enabled: cleared save_dir {self.save_dir}")
            except Exception as e:
                self.get_logger().warn(f"Failed to clear save_dir {self.save_dir}: {e}")

        self.get_logger().info(f"Save dir: {self.save_dir}")
        self.get_logger().info(f"Color topics: {self.color_topics}")
        self.get_logger().info(f"Depth  topics: {self.depth_topics}")
        self.get_logger().info(f"Joint state topics: {self.joint_state_topics}")
        self.get_logger().info(f"Joint cmd topics:   {self.joint_cmd_topics}")
        self.get_logger().info(f"Tactile topics:{self.tactile_topics}")

        # ---------- 对齐器 ----------
        C = len(self.color_topics)
        D = len(self.depth_topics) if self.save_depth else 0
        J_state = len(self.joint_state_topics) if self.joint_state_topics else 0
        J_cmd = len(self.joint_cmd_topics) if self.joint_cmd_topics else 0
        J = J_state + J_cmd
        T = len(self.tactile_topics) if self.tactile_topics else 0

        self._idx_color = list(range(C))
        self._idx_depth = list(range(C, C + D))
        self._idx_joint_state = list(range(C + D, C + D + J_state))
        self._idx_joint_cmd = list(range(C + D + J_state, C + D + J_state + J_cmd))
        self._idx_joint = self._idx_joint_state + self._idx_joint_cmd
        self._idx_tact = list(range(C + D + J, C + D + J + T))
        num_streams = C + D + J + T

        tolerances_ns: List[int] = []
        tolerances_ns += [image_tol_ns] * C
        tolerances_ns += [image_tol_ns] * D
        tolerances_ns += _as_list_ns('joint_tolerance_ms', J) if J else []
        tolerances_ns += _as_list_ns('tactile_tolerance_ms', T) if T else []

        color_offsets_ms = list(p('color_offsets_ms').value or [])
        depth_offsets_ms = list(p('depth_offsets_ms').value or [])
        offsets_ns: List[int] = []
        offsets_ns += (
            [int(ms * 1e6) for ms in color_offsets_ms] if color_offsets_ms and len(color_offsets_ms) == C else [0] * C)
        offsets_ns += ([int(ms * 1e6) for ms in depth_offsets_ms] if D and depth_offsets_ms and len(
            depth_offsets_ms) == D else [0] * D)
        offsets_ns += _as_list_offset_ns('joint_offset_ms', J) if J else []
        offsets_ns += _as_list_offset_ns('tactile_offset_ms', T) if T else []

        max_window_ns = int(self.queue_seconds * 1e9)
        camera_refs = self._idx_color + self._idx_depth
        non_consuming = list(self._idx_tact)  # tactile 非消耗
        optional_indices = list(self._idx_tact)  # tactile 可选

        ref_indices = list(camera_refs)
        if not ref_indices:
            if self._idx_joint:
                ref_indices = list(self._idx_joint)
            elif self._idx_tact:
                ref_indices = list(self._idx_tact)
            else:
                ref_indices = list(range(num_streams))

        self.aligner = MultiStreamAligner(
            num_streams=num_streams,
            tolerances_ns=tolerances_ns,
            offsets_ns=offsets_ns,
            max_window_ns=max_window_ns,
            logger=self.get_logger(),
            ref_indices=ref_indices,
            non_consuming_indices=non_consuming,
            optional_indices=optional_indices,
        )

        self.get_logger().info(
            f"rate={self.rate_hz}Hz, streams: C={C}, D={D}, J_state={J_state}, J_cmd={J_cmd}, J_total={J}, T={T}, total={num_streams}"
        )

        # ---------- 订阅 ----------
        cg_color = ReentrantCallbackGroup()
        cg_depth = ReentrantCallbackGroup()
        for i, topic in enumerate(self.color_topics):
            self.create_subscription(Image, topic, self._mk_color_cb(i), reliable_qos, callback_group=cg_color)
        for j in range(D):
            topic = self.depth_topics[j]
            if topic and self.save_depth:
                self.create_subscription(Image, topic, self._mk_depth_cb(j), reliable_qos, callback_group=cg_depth)

        cg_joint = ReentrantCallbackGroup()
        cg_tact = ReentrantCallbackGroup()
        for k, topic in enumerate(self.joint_state_topics):
            stream_idx = self._idx_joint_state[k]
            if "double_arm" in topic:
                self.create_subscription(
                    OculusInitJointState, topic, self._mk_oculus_init_joint_cb(stream_idx), reliable_qos,
                    callback_group=cg_joint
                )
            else:
                self.create_subscription(
                    JointState, topic, self._mk_joint_cb(stream_idx), reliable_qos, callback_group=cg_joint
                )
        for k, topic in enumerate(self.joint_cmd_topics):
            stream_idx = self._idx_joint_cmd[k]
            self.create_subscription(
                OculusInitJointState, topic, self._mk_oculus_init_joint_cb(stream_idx), reliable_qos,
                callback_group=cg_joint
            )
        for k, topic in enumerate(self.tactile_topics):
            self.create_subscription(Float32MultiArray, topic, self._mk_tactile_cb(k), reliable_qos,
                                     callback_group=cg_tact)

    # ---------- 回调 ----------
    def _ns_from_header_or_clock(self, header) -> int:
        try:
            s = int(header.stamp.sec)
            ns = int(header.stamp.nanosec)
            if s != 0 or ns != 0:
                return s * 1_000_000_000 + ns
        except Exception:
            pass
        return self.get_clock().now().nanoseconds

    def _mk_color_cb(self, i: int):
        def _cb(msg: Image):
            t_ns = self._ns_from_header_or_clock(
                msg.header) if self.use_ros_time else self.get_clock().now().nanoseconds
            self.aligner.put_nowait(self._idx_color[i], t_ns, msg)

        return _cb

    def _mk_depth_cb(self, j: int):
        def _cb(msg: Image):
            t_ns = self._ns_from_header_or_clock(
                msg.header) if self.use_ros_time else self.get_clock().now().nanoseconds
            self.aligner.put_nowait(self._idx_depth[j], t_ns, msg)

        return _cb

    def _mk_joint_cb(self, stream_idx: int):
        def _cb(msg: JointState):
            t_ns = self._ns_from_header_or_clock(msg.header)
            self.aligner.put_nowait(stream_idx, t_ns, msg)

        return _cb

    # def _mk_joint_cb(self, k: int):
    #     def _cb(msg: JointState):
    #         # 统一算出时间戳（ns）
    #         t_ns = self._ns_from_header_or_clock(msg.header)

    #         # ===== 力矩低通滤波（高频） =====
    #         try:
    #             if self.effort_filter_enable and getattr(msg, "effort", None) is not None:
    #                 # 原始力矩列表（拷一份，避免原地改的时候丢失原始）
    #                 raw_efforts = [float(x) for x in msg.effort]
    #                 n_raw = len(raw_efforts)

    #                 # 防御：如果实际维度和配置不一致，处理前 n 个
    #                 n_ch = min(self.effort_filter_num_channels, n_raw)

    #                 filtered_efforts = []
    #                 for i_dim, val in enumerate(raw_efforts):
    #                     if i_dim < n_ch and i_dim < len(self.effort_filters):
    #                         filtered = float(self.effort_filters[i_dim].update(val))
    #                         filtered_efforts.append(filtered)
    #                     else:
    #                         # 超出配置范围的维度，直接原样通过
    #                         filtered_efforts.append(val)

    #                 # 按你说的方案：
    #                 # 使用 msg.effort 一个字段，打包成 [原始7维, 滤波后7维]
    #                 # 假设 n_raw == 7，如果不是，前 n_ch 维对应滤波，其余部分原样复制两遍也可以按需改
    #                 msg.effort = raw_efforts + filtered_efforts
    #         except Exception as e:
    #             # 任何滤波错误都不能影响对齐逻辑
    #             self._warn("effort filter in joint_cb error: %s", e)

    #         # 无论是否启用滤波，消息都照常送给对齐器
    #         self.aligner.put_nowait(self._idx_joint[k], t_ns, msg)

    #     return _cb

    def _mk_oculus_init_joint_cb(self, stream_idx: int):
        def _cb(msg: OculusInitJointState):
            t_ns = self._ns_from_header_or_clock(msg.header)
            self.aligner.put_nowait(stream_idx, t_ns, msg)

        return _cb

    # def _mk_oculus_init_joint_cb(self, k: int):
    #     last_log_t = time.perf_counter()
    #     last_count = 0
    #     count = 0
    #     topic = f"joint[{k}]"

    #     def _cb(msg: OculusInitJointState):
    #         nonlocal last_log_t, last_count, count
    #         count += 1
    #         if self.do_calculate_hz:
    #             now = time.perf_counter()
    #             log_period_s = max(0.2, float(self.stats_log_period_s))
    #             if now - last_log_t >= log_period_s:
    #                 dt = now - last_log_t
    #                 delta = count - last_count
    #                 hz = (delta / dt) if dt > 0 else 0.0
    #                 self.get_logger().info(
    #                     f"[oculus_init_joint_cb] idx={k} topic={topic} rate={hz:.2f}Hz"
    #                 )
    #                 last_log_t = now
    #                 last_count = count
    #         t_ns = self._ns_from_header_or_clock(msg.header)
    #         self.aligner.put_nowait(self._idx_joint[k], t_ns, msg)

    #     return _cb

    def _mk_tactile_cb(self, k: int):
        def _cb(msg: Float32MultiArray):
            t_ns = self.get_clock().now().nanoseconds
            self.aligner.put_nowait(self._idx_tact[k], t_ns, msg)

        return _cb

    # ---------- 关闭 ----------
    def destroy_node(self):
        return super().destroy_node()

def main(args=None):
    """
    Simple entrypoint to run BaseManager for quick testing.
    Usage:
      python3 -m common_utils.base_manager
    """
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    try:
        rclpy.init(args=args)
    except Exception:
        # rclpy.init may already be called by a launcher; ignore init errors
        pass

    node = None
    try:
        node = BaseManager()  # uses default node name 'manager'
        node.get_logger().info("BaseManager started. Ctrl-C to exit.")
        rclpy.spin(node)
        # executor = MultiThreadedExecutor(num_threads=1)
        # executor.add_node(node)
        # executor.spin()
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("Keyboard interrupt, shutting down.")
    except Exception as e:
        if node is not None:
            node.get_logger().error(f"Unhandled error in main: {e}")
        else:
            print(f"Unhandled error in main: {e}")
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
