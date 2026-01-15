#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import threading
import time
import traceback
import json
from typing import List, Tuple, Any, Dict, Optional
from collections import deque
import numpy as np
from dataclasses import dataclass

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

COMMON_UTILS_ROOT = "~/ros2_ws/src/common"
if COMMON_UTILS_ROOT not in sys.path:
    sys.path.insert(0, COMMON_UTILS_ROOT)

from common_utils.base_manager import BaseManager
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

from sensor_msgs.msg import JointState, Image
from common.msg import OculusInitJointState
from std_msgs.msg import Header
from numpy_logger import NumpyCSVLogger

from rclpy.time import Time


# ---------------------- 工具：图像解码 ----------------------
def _dtype_and_channels(encoding: str):
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
    raise ValueError(f"Unsupported image encoding: {encoding}")


def _image_to_numpy(msg: Image) -> np.ndarray:
    dtype, ch = _dtype_and_channels(msg.encoding)
    buf = np.frombuffer(msg.data, dtype=dtype)
    elem_size = np.dtype(dtype).itemsize
    row_elems = msg.step // elem_size
    need = msg.width * ch
    if row_elems < need:
        raise ValueError(
            f"Invalid image stride: row_elems({row_elems}) < width*channels({need}). "
            f"encoding={msg.encoding}, step={msg.step}, dtype={dtype}"
        )
    arr2d = buf.reshape(msg.height, row_elems)[:, :need]
    if ch == 1:
        out = arr2d.reshape(1, msg.height, msg.width)
    else:
        out = arr2d.reshape(msg.height, msg.width, ch).transpose(2, 0, 1)
    return out


def _pack_depth_u16_to_rgb8(depth_u16: np.ndarray, order: str = "HI_LO", b_fill: int = 0) -> np.ndarray:
    if depth_u16.dtype != np.uint16:
        raise ValueError("depth_u16 must be uint16")
    depth_u16 = depth_u16[..., 0]
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
        picks: List[Optional[Tuple[int, Any]]],
        state_idx,
        color_idx: List[int],
        depth_idx: List[int],
        pack_depth_2_rgb8: bool,
) -> Dict:
    obs_dict: Dict[str, Any] = {"state": None, "images": {}}

    # state
    state_indices = state_idx if isinstance(state_idx, (list, tuple)) else [state_idx]
    parts: List[np.ndarray] = []
    for idx in state_indices:
        if not isinstance(idx, int) or idx < 0 or idx >= len(picks):
            continue
        item = picks[idx]
        if item is None:
            continue
        _, js = item
        if isinstance(js, JointState):
            if not js.position:
                continue
            parts.append(np.asarray(js.position, dtype=np.float32))
            continue
        if isinstance(js, OculusInitJointState):
            added = False
            if getattr(js, "left_valid", False) and js.left.position:
                parts.append(np.asarray(js.left.position, dtype=np.float32))
                added = True
            if getattr(js, "right_valid", False) and js.right.position:
                parts.append(np.asarray(js.right.position, dtype=np.float32))
                added = True
            if added:
                continue
            continue
        raise TypeError(f"state_idx {idx} points to {type(js)}, expected JointState/OculusInitJointState")
    if not parts:
        raise ValueError("No valid JointState.position found")
    obs_dict["state"] = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]

    # color
    for k, idx in enumerate(color_idx):
        if not isinstance(idx, int) or idx < 0 or idx >= len(picks):
            continue
        item = picks[idx]
        if item is None:
            continue
        _, msg = item
        if not isinstance(msg, Image):
            raise TypeError(f"color_idx[{k}] -> {type(msg)}, expect Image")
        obs_dict["images"][f"camera{k}"] = _image_to_numpy(msg)

    # depth
    for k, idx in enumerate(depth_idx):
        if not isinstance(idx, int) or idx < 0 or idx >= len(picks):
            continue
        item = picks[idx]
        if item is None:
            continue
        _, msg = item
        if not isinstance(msg, Image):
            raise TypeError(f"depth_idx[{k}] -> {type(msg)}, expect Image")
        depth_chw = _image_to_numpy(msg)
        enc = str(getattr(msg, "encoding", "")).lower()
        if enc in ("mono16", "16uc1") and depth_chw.ndim == 3 and depth_chw.shape[0] == 1:
            if pack_depth_2_rgb8:
                hw_u16 = depth_chw[0].astype(np.uint16, copy=False)
                depth_rgb_hwc = _pack_depth_u16_to_rgb8(hw_u16)
                obs_dict["images"][f"camera{k}_depth"] = depth_rgb_hwc.transpose(2, 0, 1)
            else:
                obs_dict["images"][f"camera{k}_depth"] = depth_chw
        else:
            obs_dict["images"][f"camera{k}_depth"] = depth_chw
    return obs_dict


def cubic_transition(old_actions, new_actions):
    """
    三次插值平滑衔接，返回平滑后的动作序列。
    old_actions: list[np.ndarray] or (M, D) array，尚未执行的旧动作（从“下一帧”开始）
    new_actions: np.ndarray shape=(N, D)，新推理动作（从对应对齐帧开始）
    返回: list[np.ndarray]，长度为 max(min(M,N), N) —— 先对前 min(M,N) 帧平滑，之后直接接新动作剩余部分
    """
    # 统一为 (M, D) / (N, D) 的 float32 数组
    old_arr = np.asarray(old_actions, dtype=np.float32)
    if old_arr.ndim == 1:
        old_arr = old_arr[None, :]
    new_arr = np.asarray(new_actions, dtype=np.float32)
    if new_arr.ndim == 1:
        new_arr = new_arr[None, :]

    # 维度对齐（零填充到相同 D）
    D = max(old_arr.shape[1], new_arr.shape[1])

    def _pad_dim(a, D):
        if a.shape[1] == D:
            return a
        return np.pad(a, ((0, 0), (0, D - a.shape[1])), mode="constant", constant_values=0.0)

    old_arr = _pad_dim(old_arr, D)
    new_arr = _pad_dim(new_arr, D)

    n_old = old_arr.shape[0]
    n_new = new_arr.shape[0]
    n_interp = min(n_old, n_new)

    if n_interp == 0:
        # 没有重叠可平滑，直接采用新动作
        return [row for row in new_arr]

    if n_interp == 1:
        # 只有 1 帧可平滑：取中点（避免硬切）
        t = 0.5
        h = 3 * t * t - 2 * t * t * t  # 0.5
        blended0 = (1.0 - h) * old_arr[0] + h * new_arr[0]
        out = [blended0]
    else:
        # 端点包含：t=0 -> old，t=1 -> new
        t = np.linspace(0.0, 1.0, n_interp, dtype=np.float32)  # [0,1]
        h = (3 * t ** 2 - 2 * t ** 3)[:, None]  # (n_interp,1)
        blended = (1.0 - h) * old_arr[:n_interp] + h * new_arr[:n_interp]
        out = [row for row in blended]

    # 接上新动作剩余部分
    if n_new > n_interp:
        out.extend([row for row in new_arr[n_interp:]])

    # 返回 list[np.ndarray]（每帧 shape=(D,)）
    return [np.asarray(row, dtype=np.float32) for row in out]


# ======================= 动作帧（带时间戳） =======================
@dataclass
class ActionFrame:
    ts: float  # 该帧计划时间戳（秒，float）
    a: np.ndarray  # (dof,) float32


def load_jsonl(path: str, logger) -> Tuple[List[Dict], List[Dict]]:
    """
    Read a JSONL file and return (right_list, left_list).

    Each line should be a JSON object with a 'joints' list. Each joint entry is expected to
    contain fields like 'topic', 'stamp_ns', 'name', 'position', 'velocity', 'effort'.
    """
    right_list: List[Dict] = []
    left_list: List[Dict] = []
    if not os.path.exists(path):
        logger.error(f"File does not exist: {path}")
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                js = json.loads(line)
            except Exception as e:
                logger.warning(f"Skipping invalid JSON at line {line_no+1}: {e}")
                continue
            joints = js.get('joints', [])
            for j in joints:
                topic = j.get('topic', '')
                entry = {
                    'stamp_ns': j.get('stamp_ns', 0),
                    'name': j.get('name', []),
                    'position': j.get('position', []),
                    'velocity': j.get('velocity', []),
                    'effort': j.get('effort', [])
                }
                if isinstance(topic, str) and (topic.endswith('right') or topic == '/joint_states_right'):
                    right_list.append(entry)
                elif isinstance(topic, str) and (topic.endswith('left') or topic == '/joint_states_left'):
                    left_list.append(entry)
    logger.info(f"Loaded {len(right_list)} right entries and {len(left_list)} left entries from {path}")
    return right_list, left_list

def make_jointstate_msg(self, entry: dict) -> JointState:
    msg = JointState()
    # set stamp if available
    stamp_ns = int(entry.get('stamp_ns', 0))
    try:
        # Time requires non-negative integer nanoseconds
        if stamp_ns > 0:
            msg.header.stamp = Time(nanoseconds=stamp_ns).to_msg()
    except Exception:
        # ignore stamp if invalid
        pass
    msg.name = entry.get('name', [])
    msg.position = [float(x) for x in entry.get('position', [])]
    msg.velocity = [float(x) for x in entry.get('velocity', [])]
    msg.effort = [float(x) for x in entry.get('effort', [])]
    return msg
    
# ======================= 主要节点 =======================
class InferenceManager(BaseManager):
    """
    - 发布线程：固定 30Hz，从 _future_actions 里 popleft()，并把 _frames_since_update += 1
    - 推理线程：仅当 _frames_since_update >= replan_threshold_frames 或 队列为空 时，取最新观测 -> 推理，
      产出 new_plan（含时间戳；第0帧 ts=观测时间），
      然后用时间戳对齐 old_tail[0].ts 与 new_plan[k].ts，融合前 fw 帧，更新队列，最后 _frames_since_update 置0。
    """

    def __init__(self):
        super().__init__(node_name='inference_manager')

        # ---------- 参数 ----------
        self.declare_parameter('checkpoint_dir', '/home/kleist/Documents/Model/cloud_server/1206_pi05_test/30000/')  # noqa: Q000
        self.declare_parameter('policy_name', 'pi05_agileX')

        self.declare_parameter('publish_rate_hz', 30)
        self.declare_parameter('horizon', 50)
        self.declare_parameter('replan_threshold_frames', 20)
        self.declare_parameter('ema', 0.70)
        self.ema_ignore_dims = [6] # None or [0,3,6]
        
        self.declare_parameter('cmd_joint_topic', '/joint_cmd_right')
        self.declare_parameter('cmd_joint_names',
                               ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7'])
        self.declare_parameter('skip_if_no_subscriber', False)
        self.declare_parameter('dump_logs', True)

        p = self.get_parameter
        self.checkpoint_dir: str = str(p('checkpoint_dir').value)
        self.policy_name: str = str(p('policy_name').value)
        self.publish_rate_hz: float = float(p('publish_rate_hz').value)
        self.horizon: int = int(p('horizon').value)
        self.replan_threshold_frames: int = int(p('replan_threshold_frames').value)
        self.ema: float = float(p('ema').value)

        self.cmd_joint_topic: str = str(p('cmd_joint_topic').value)
        self.cmd_joint_names: List[str] = list(p('cmd_joint_names').value)
        self.skip_if_no_sub: bool = bool(p('skip_if_no_subscriber').value)
        self.dump_logs: bool = bool(p('dump_logs').value)   

        # ---------- 发布者 ----------
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.pub_joint_cmd = self.create_publisher(JointState, self.cmd_joint_topic, reliable_qos)

        # ---------- 策略 ----------
        try:
            cfg = _config.get_config(self.policy_name)
            self.policy = _policy_config.create_trained_policy(cfg, self.checkpoint_dir)
            self.get_logger().info(f"Policy loaded: {self.policy_name} from {self.checkpoint_dir}")
        except Exception as e:
            self.get_logger().error(f"Failed to load policy: {e}\n{traceback.format_exc()}")
            raise
        self.horizon = cfg.model.action_horizon
        # ---------- 共享状态 ----------
        self._dt = 1.0 / max(1.0, self.publish_rate_hz)
        self._future_actions: deque[ActionFrame] = deque(maxlen=self.horizon)  # 未来队列（含 ts）
        self._future_lock = threading.Lock()
        self._frames_since_update: int = 999999  # 启动即触发一次推理
        self._last_action: Optional[np.ndarray] = None
        self._safe_action = np.zeros((len(self.cmd_joint_names),), dtype=np.float32)

        # 发布时钟
        self._base_time: Optional[float] = None
        self._next_tick: Optional[float] = None

        # ---------- 日志 ----------
        if self.dump_logs:
            self.logger_action = NumpyCSVLogger(
                "logs/action_0819_2_cameras.csv",
                mode="w",
                add_timestamp_to_filename=True,
            )
            self.logger_obs = NumpyCSVLogger(
                "logs/obs_0819_2_cameras.csv",
                mode="w",
                add_timestamp_to_filename=True,
            )
            self.logger_action_sended = NumpyCSVLogger(
                "logs/action_sended_0819_2_cameras.csv",
                mode="w",
                add_timestamp_to_filename=True,
            )

        # # prepare storage for optional JSONL-loaded joint entries
        # jsonl_path = '/home/test/jemotor/openpi/manager_node_temp/meta.jsonl'  # 修改为实际路径或通过参数传入
        # self.right_list: List[Dict] = []
        # self.left_list: List[Dict] = []
        # if jsonl_path:
        #     try:
        #         r, l = load_jsonl(jsonl_path, self.get_logger())
        #         self.right_list = r
        #         self.left_list = l
        #     except Exception as e:
        #         self.get_logger().error(f"Failed to load joint states jsonl: {e}")

        # 线程管理
        self._stop_evt = threading.Event()
        self._threads: List[threading.Thread] = []

        self._start_threads()

    # ---------- 线程 ----------
    def _start_threads(self):
        now = time.monotonic()
        self._base_time = now
        self._next_tick = self._ceil_to_next_tick(now)

        th_pub = threading.Thread(target=self._publish_loop, name='publish-loop', daemon=True)
        th_inf = threading.Thread(target=self._inference_loop, name='inference-loop', daemon=True)
        th_pub.start()
        th_inf.start()
        self._threads.extend([th_pub, th_inf])
        self.get_logger().info("[threads] publish-loop & inference-loop started")

    # ---------- 发布线程：30Hz 逐帧发布 ----------
    def _publish_loop(self):
        assert self._base_time is not None and self._next_tick is not None
        while rclpy.ok() and not self._stop_evt.is_set():
            now = time.monotonic()
            sleep_s = self._next_tick - now
            if sleep_s > 0:
                time.sleep(min(sleep_s, 0.002))
                continue
            else:
                # 我们已经落后，跳过丢失的 ticks，重置 next_t
                self._next_tick = now

            try:
                action = self._pop_next_action()
                action = np.asarray(action, dtype=float).reshape(-1)
                if action.shape[0] != 7:
                    raise ValueError(f"action must be shape (7,), got {action.shape}")

                # 你可以在类里定义：self.ema_ignore_dims = None 或 [] 或 [0,3,6]
                ignore_dims = getattr(self, "ema_ignore_dims", None)

                if self.ema > 0.0 and self._last_action is not None:
                    action = self._apply_ema_with_ignore(
                        action=action,
                        last_action=self._last_action,
                        ema=float(self.ema),
                        ignore_dims=ignore_dims,
                    )

                self._publish_joint(action)
                self._last_action = action

                if self.dump_logs:
                    self.logger_action_sended.log(action)

                self._frames_since_update += 1

            except Exception as e:
                self.get_logger().error(f"publish error: {e}\n{traceback.format_exc()}")

            self._next_tick += self._dt

    def _publish_loop_replay(self):
        # Scheduled replay publish loop: mirror behavior of _publish_loop
        assert self._base_time is not None and self._next_tick is not None
        # ensure an index counter exists
        if not hasattr(self, 'index'):
            self.index = 0

        while rclpy.ok() and not self._stop_evt.is_set():
            now = time.monotonic()
            sleep_s = self._next_tick - now
            if sleep_s > 0:
                time.sleep(min(sleep_s, 0.002))
                continue
            else:
                # we're behind; reset next tick to avoid burst catch-up
                self._next_tick = now

            try:
                # Publish right-side entry if present
                if len(self.right_list) > 0:
                    idx_r = self.index % len(self.right_list)
                    entry_r = self.right_list[idx_r]
                    # entry can be a dict (from jsonl) or an action array; handle both
                    if isinstance(entry_r, dict):
                        try:
                            msg_r = make_jointstate_msg(self, entry_r)
                            self.pub_joint_cmd.publish(msg_r)
                        except Exception:
                            # failed to convert/publish dict entry as JointState
                            self.get_logger().error(f"Failed to publish right_list dict entry at index {idx_r}\n{traceback.format_exc()}")
                    else:
                        self._publish_joint(entry_r)

                # (optional) left list handling left commented out to match prior behaviour
                # if len(self.left_list) > 0:
                #     idx_l = self.index % len(self.left_list)
                #     msg_l = make_jointstate_msg(self, self.left_list[idx_l])
                #     self.pub_left.publish(msg_l)

                # advance counters
                self.index += 1
                self._frames_since_update += 1

                # Determine whether we've published the last frame for both lists
                done_r = (len(self.right_list) == 0) or (self.index >= len(self.right_list))
                done_l = (len(self.left_list) == 0) or (self.index >= len(self.left_list))
                if done_r and done_l:
                    self.get_logger().info("Finished replay (non-loop). Shutting down node.")
                    # signal stop and break
                    self._stop_evt.set()
                    break

            except Exception as e:
                self.get_logger().error(f"publish replay error: {e}\n{traceback.format_exc()}")

            # schedule next tick
            self._next_tick += self._dt

    def _pop_next_action(self) -> np.ndarray:
        with self._future_lock:
            if self._future_actions:
                fr = self._future_actions.popleft()
                a = fr.a
            else:
                # self.get_logger().error("no future actions")
                a = self._last_action if self._last_action is not None else self._safe_action
        return self._fit_action_dim(a)

    # ---------- 推理线程：按阈值触发 ----------
    def _inference_loop(self):
        while rclpy.ok() and not self._stop_evt.is_set():
            try:
                need_replan = (self._frames_since_update >= self.replan_threshold_frames)
                # need_replan = False
                with self._future_lock:
                    queue_empty = (len(self._future_actions) == 0)

                if not (need_replan or queue_empty):
                    time.sleep(0.002)
                    continue

                # 拿最新对齐观测
                out = self.aligner.step()
                if out is None:
                    time.sleep(0.001)
                    continue
                t_ref, picks = out
                obs = self._build_obs(picks)

                # 规范化观测时间戳（秒）
                obs_ts = float(t_ref * 1e-9)
                # 推理
                t0 = time.monotonic()
                result = self.policy.infer(obs)
                t1 = time.monotonic()

                if self.dump_logs:
                    # 直接把 state / action 写入；不在此处自动加入行级时间戳
                    self.logger_obs.log(obs['state'])
                    for row in result['actions']:
                        self.logger_action.log(row)

                actions = result.get('actions', None)
                if actions is None:
                    self.get_logger().warn("Policy returned no 'actions'")
                    time.sleep(0.01)
                    continue

                act = np.asarray(actions, dtype=np.float32)
                if act.ndim == 1:
                    act = act[None, :]
                if act.shape[0] != self.horizon:
                    act = self._pad_or_clip(act, self.horizon)

                # 构造带时间戳的新计划（第0帧 ts = obs_ts，等间隔 dt）
                new_plan = self._make_new_plan_frames(act, obs_ts)

                # 按时间戳对齐并融合 -> 更新未来队列
                self._fuse_and_update_queue_by_ts(new_plan)

                # 复位计数
                self._frames_since_update = 0

                self.get_logger().debug(
                    f"[infer] latency={(t1 - t0) * 1000:.1f}ms, obs_ts={obs_ts:.6f}, new plan H={len(new_plan)}"
                )

            except Exception as e:
                self.get_logger().error(f"inference loop error: {e}\n{traceback.format_exc()}")

    # ---------- 推理线程：按阈值触发 ----------
    def _inference_loop_replay(self):
        start_index = 0  # starting index for replay data
        while rclpy.ok() and not self._stop_evt.is_set():
            try:
                need_replan = (self._frames_since_update >= self.replan_threshold_frames)
                # need_replan = False
                with self._future_lock:
                    queue_empty = (len(self._future_actions) == 0)

                if not (need_replan or queue_empty):
                    time.sleep(0.002)
                    continue

                # 拿最新对齐观测
                out = self.aligner.step()
                if out is None:
                    time.sleep(0.001)
                    continue
                t_ref, picks = out
                obs = self._build_obs(picks)

                # 规范化观测时间戳（秒）
                obs_ts = float(t_ref * 1e-9)
                # 推理
                t0 = time.monotonic()
                result = self.policy.infer(obs)
                t1 = time.monotonic()

                if self.dump_logs:
                    # 直接把 state / action 写入；不在此处自动加入行级时间戳
                    self.logger_obs.log(obs['state'])
                    for row in result['actions']:
                        self.logger_action.log(row)

                # Use positions from right_list as replayed actions
                end_index = min(start_index + self.horizon, len(self.right_list))
                chunk = self.right_list[start_index:end_index]
                start_index = end_index  # advance index for next replay
                # extract position lists
                actions = []
                for entry in chunk:
                    if isinstance(entry, dict):
                        actions.append(entry.get('position', []))
                    else:
                        actions.append(entry)
                if not actions:
                    self.get_logger().warn("No replay actions available in chunk")
                    time.sleep(0.01)
                    continue
                
                act = np.asarray(actions, dtype=np.float32)
                if act.ndim == 1:
                    act = act[None, :]
                if act.shape[0] != self.horizon:
                    act = self._pad_or_clip(act, self.horizon)

                # 构造带时间戳的新计划（第0帧 ts = obs_ts，等间隔 dt）
                new_plan = self._make_new_plan_frames(act, obs_ts)

                # 按时间戳对齐并融合 -> 更新未来队列
                self._fuse_and_update_queue_by_ts(new_plan)

                # 复位计数
                self._frames_since_update = 0

                self.get_logger().debug(
                    f"[infer] latency={(t1 - t0) * 1000:.1f}ms, obs_ts={obs_ts:.6f}, new plan H={len(new_plan)}"
                )

            except Exception as e:
                self.get_logger().error(f"inference loop error: {e}\n{traceback.format_exc()}")


    # ---------- 融合逻辑（基于时间戳对齐） ----------
    def _fuse_and_update_queue_by_ts(self, new_plan: List[ActionFrame]):
        """
        1) old_tail = 未来队列（发布线程在推理期间可能已经弹掉多帧）
        2) 以 old_tail[0].ts 在 new_plan 中用时间戳对齐出 index k（最近/不早于者）
        3) 前 fw 帧线性融合：old_tail[i] <-> new_plan[k+i]；其后直接采用 new_plan[k+fw:]
        4) 为保证发布节拍连续，输出队列的 ts 从 old_tail[0].ts 起按 dt 重新铺设
        """
        self.get_logger().info(f"Fusing new plan with {len(new_plan)} frames into future queue {len(self._future_actions)} frames")
        with self._future_lock:
            if len(self._future_actions) == 0:
                # 队列空或不需要融合：直接覆盖（但仍按 old_ts 铺设；此时用 new_plan[0].ts）
                fused_actions = [fr.a.astype(np.float32, copy=False) for fr in new_plan]
                start_ts = new_plan[0].ts
            else:
                old_tail = list(self._future_actions)  # List[ActionFrame]
                old0_ts = old_tail[0].ts

                # 在 new_plan 中找到与 old0_ts 对齐的起点 k
                new_ts = np.asarray([fr.ts for fr in new_plan], dtype=np.float64)
                # 首先找不早于 old0_ts 的位置
                k = int(np.searchsorted(new_ts, old0_ts, side='left'))
                # 取最近（如果左边更近，就往前挪一位）
                if k > 0 and (k >= len(new_ts) or abs(new_ts[k] - old0_ts) > abs(new_ts[k - 1] - old0_ts)):
                    k -= 1
                k = max(0, min(k, len(new_plan) - 1))

                fw = int(min(len(old_tail), len(new_plan) - k))
                if fw > 0:
                    old_arr = np.stack([fr.a for fr in old_tail], axis=0).astype(np.float32, copy=False)
                    new_arr = np.stack([fr.a for fr in new_plan[k:]], axis=0).astype(np.float32, copy=False)
                    fused_actions = cubic_transition(old_arr, new_arr)  # -> list[np.ndarray]
                else:
                    fused_actions = [new_plan[k + i].a.astype(np.float32, copy=False) for i in range(len(new_plan) - k)]

                # 输出时间轴从 old0_ts 开始，以保持与发布节拍连续
                start_ts = old0_ts

            # 维度对齐 & 裁剪到 horizon，并重建 ActionFrame（按 dt 铺设 ts）
            fused_actions = [self._fit_action_dim(a) for a in fused_actions]
            if len(fused_actions) > self.horizon:
                fused_actions = fused_actions[:self.horizon]

            self._future_actions.clear()
            for i, a in enumerate(fused_actions):
                self._future_actions.append(ActionFrame(ts=start_ts + i * self._dt, a=a))

    # ---------- 计划构建（带时间戳） ----------
    def _make_new_plan_frames(self, act: np.ndarray, start_ts: float) -> List[ActionFrame]:
        """
        act: (H, dof) float32
        start_ts: 第0帧时间戳（秒）= 观测时间
        """
        frames: List[ActionFrame] = []
        for i in range(act.shape[0]):
            frames.append(ActionFrame(ts=start_ts + i * self._dt, a=act[i]))
        return frames

    # ---------- 发布 ----------
    def _publish_joint(self, action: np.ndarray):
        # 无订阅者时可跳过
        try:
            if self.skip_if_no_sub and hasattr(self.pub_joint_cmd, "get_subscription_count"):
                if self.pub_joint_cmd.get_subscription_count() == 0:
                    return
        except Exception:
            pass

        action_list = list(map(float, self._fit_action_dim(action)))
        js = JointState()
        js.header = Header()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(self.cmd_joint_names)
        js.position = action_list
        self.pub_joint_cmd.publish(js)

    # ---------- 观测 ----------
    def _build_obs(self, picks):
        try:
            return transform_ros2msg_2_np(picks, self._idx_joint, self._idx_color, self._idx_depth, False)
        except Exception as e:
            raise RuntimeError(f"_build_obs failed: {e}")

    # ---------- 小工具 ----------
    def _fit_action_dim(self, a: np.ndarray) -> np.ndarray:
        dof = len(self.cmd_joint_names)
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        if a.shape[0] == dof:
            return a
        out = np.zeros((dof,), dtype=np.float32)
        n = min(dof, a.shape[0])
        out[:n] = a[:n]
        return out

    @staticmethod
    def _pad_or_clip(arr: np.ndarray, H: int) -> np.ndarray:
        if arr.shape[0] == H:
            return arr
        if arr.shape[0] > H:
            return arr[:H]
        pad = np.repeat(arr[-1][None, :], H - arr.shape[0], axis=0)
        return np.concatenate([arr, pad], axis=0)

    def _ceil_to_next_tick(self, t: float) -> float:
        if self._base_time is None:
            self._base_time = t
        k = int(np.ceil((t - self._base_time) / self._dt))
        return self._base_time + k * self._dt

    @staticmethod
    def _normalize_ts_to_seconds(t_ref) -> float:
        if hasattr(t_ref, "nanoseconds"):
            return float(t_ref.nanoseconds) * 1e-9
        if hasattr(t_ref, "sec") and hasattr(t_ref, "nanosec"):
            return float(t_ref.sec) + float(t_ref.nanosec) * 1e-9
        if isinstance(t_ref, (int, np.integer)):
            v = int(t_ref)
            return (v * 1e-9) if v > 1_000_000_000_000 else float(v)
        return float(t_ref)

    def _apply_ema_with_ignore(action: np.ndarray,
                            last_action: np.ndarray,
                            ema: float,
                            ignore_dims=None) -> np.ndarray:
        """
        action/last_action: shape (7,)
        ignore_dims: None / [] / iterable of indices (0..6). Those dims will NOT be EMA-smoothed.
        """
        if ema <= 0.0 or last_action is None:
            return action

        # None 或空 => 不忽略任何维度（对所有维度做 EMA）
        if not ignore_dims:
            return (1.0 - ema) * action + ema * last_action

        # 生成 mask：True 表示做 EMA；False 表示忽略（直接用当前 action）
        mask = np.ones(action.shape, dtype=bool)

        # 允许传入如 [0,3,6]，也允许传入 np array/list
        idx = np.asarray(list(ignore_dims), dtype=int)

        # 支持负索引（Python 风格），并校验范围
        idx = np.where(idx < 0, idx + action.shape[0], idx)
        if np.any((idx < 0) | (idx >= action.shape[0])):
            raise ValueError(f"ema_ignore_dims out of range: {ignore_dims}, action_dim={action.shape[0]}")

        mask[idx] = False

        out = action.copy()
        out[mask] = (1.0 - ema) * action[mask] + ema * last_action[mask]
        # out[~mask] 已经是 action 原值
        return out

    # ---------- 关闭 ----------
    def destroy_node(self):
        self._stop_evt.set()
        for th in self._threads:
            try:
                if th.is_alive():
                    th.join(timeout=3.0)
                    if th.is_alive():
                        th.join()
            except Exception:
                pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = InferenceManager()
    try:
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
