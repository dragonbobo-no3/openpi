#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import threading
import time
import traceback
from typing import List, Sequence, Tuple, Any, Dict, Optional

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from base_manager import BaseManager

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Header


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
    """将 ROS2 Image 解到 CHW numpy。保留原始通道顺序（rgb8→RGB，bgr8→BGR 等）。"""
    dtype, ch = _dtype_and_channels(msg.encoding)

    # 从 data 构造一维 buffer
    buf = np.frombuffer(msg.data, dtype=dtype)

    # 每行 stride 的元素数（step 是字节数）
    elem_size = np.dtype(dtype).itemsize
    row_elems = msg.step // elem_size
    need = msg.width * ch
    if row_elems < need:
        raise ValueError(
            f"Invalid image stride: row_elems({row_elems}) < width*channels({need}). "
            f"encoding={msg.encoding}, step={msg.step}, dtype={dtype}"
        )

    # 先按 (H, row_elems) reshape，再切到有效列宽（width*ch）
    arr2d = buf.reshape(msg.height, row_elems)[:, :need]

    if ch == 1:
        # 单通道：返回 (1, H, W)
        out = arr2d.reshape(1, msg.height, msg.width)
    else:
        # 多通道：先得到 HWC，再转为 CHW
        out = arr2d.reshape(msg.height, msg.width, ch).transpose(2, 0, 1)

    # 需要连续内存可取消注释（例如喂给某些需要 contiguous 的库）
    # out = np.ascontiguousarray(out)

    return out


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
        picks: List[Optional[Tuple[int, Any]]],
        state_idx,                     # 可为 int 或 List[int]
        color_idx: List[int],
        depth_idx: List[int],
        pack_depth_2_rgb8: bool,
) -> Dict:
    """
    将 ros2 消息转换为模型需要的形式：
      - state: JointState.position -> np.ndarray (N,)
      - color images: Image -> np.ndarray CHW，dtype 由 encoding 决定（rgb8/bgr8/rgba8/...）
      - depth images: Image -> uint16 CHW (1,H,W)，或按开关打包为 rgb8 CHW (3,H,W)
    说明：该函数假定 picks[i] = (timestamp_or_idx, ros2_msg)
    """
    obs_dict: Dict[str, Any] = {"state": None, "images": {}}

    # -------- state --------
    # 允许传入单 index 或 多 index（多 index 时将 position 串接）
    state_indices = state_idx if isinstance(state_idx, (list, tuple)) else [state_idx]
    state_parts: List[np.ndarray] = []
    for idx in state_indices:
        if not isinstance(idx, int) or idx < 0 or idx >= len(picks):
            continue
        item = picks[idx]
        if item is None:
            continue
        _, js = item
        if not isinstance(js, JointState):
            raise TypeError(f"state_idx {idx} points to {type(js)}, expected JointState")
        if js.position is None or len(js.position) == 0:
            continue
        state_parts.append(np.asarray(js.position, dtype=np.float32))
    if not state_parts:
        raise ValueError("No valid JointState.position found from provided state_idx/indices")
    obs_dict["state"] = np.concatenate(state_parts, axis=0) if len(state_parts) > 1 else state_parts[0]

    # -------- color images (CHW) --------
    for k, idx in enumerate(color_idx):
        if not isinstance(idx, int) or idx < 0 or idx >= len(picks):
            continue
        item = picks[idx]
        if item is None:
            continue
        _, msg = item
        if not isinstance(msg, Image):
            raise TypeError(f"color_idx[{k}] points to {type(msg)}, expected sensor_msgs.msg.Image")
        img_chw = _image_to_numpy(msg)  # CHW
        obs_dict["images"][f"camera{k}"] = img_chw

    # -------- depth images (CHW) --------
    for k, idx in enumerate(depth_idx):
        if not isinstance(idx, int) or idx < 0 or idx >= len(picks):
            continue
        item = picks[idx]
        if item is None:
            continue
        _, msg = item
        if not isinstance(msg, Image):
            raise TypeError(f"depth_idx[{k}] points to {type(msg)}, expected sensor_msgs.msg.Image")

        depth_chw = _image_to_numpy(msg)  # CHW
        enc = str(getattr(msg, "encoding", "")).lower()

        # 仅当是 16 位单通道深度时，才按开关进行打包
        if enc in ("mono16", "16uc1") and depth_chw.ndim == 3 and depth_chw.shape[0] == 1:
            if pack_depth_2_rgb8:
                # _pack_depth_u16_to_rgb8 期望 2D uint16，先取出 (H,W)
                hw_u16 = depth_chw[0]
                if hw_u16.dtype != np.uint16:
                    hw_u16 = hw_u16.astype(np.uint16, copy=False)
                depth_rgb_hwc = _pack_depth_u16_to_rgb8(hw_u16)  # (H,W,3) uint8
                depth_rgb_chw = depth_rgb_hwc.transpose(2, 0, 1)  # -> (3,H,W)
                obs_dict["images"][f"camera{k}_depth"] = depth_rgb_chw
            else:
                obs_dict["images"][f"camera{k}_depth"] = depth_chw  # (1,H,W) uint16
        else:
            # 其他格式（如 32FC1 或非单通道）按 CHW 原样输出
            obs_dict["images"][f"camera{k}_depth"] = depth_chw

    return obs_dict


class InferenceManager(BaseManager):
    """
    推理管理器：基于 BaseManager 提供的订阅与对齐功能，在固定频率：
      1) 调用对齐器拿一帧观测
      2) 送入策略得到 action 序列
      3) 发布到 /joint_cmd（可配置）
    """

    def __init__(self):
        super().__init__(node_name='inference_manager')

        # ===== 子类新增参数 =====
        self.declare_parameter('checkpoint_dir', '/home/test/jemotor/jemodel/pi05/1029_pi05_test/25000/')
        self.declare_parameter('policy_name', 'pi05_agileX_depth')
        self.declare_parameter('inference_rate_hz', 30)  # 默认跟随 BaseManager 的 rate_hz
        self.declare_parameter('cmd_joint_topic', '/joint_cmd_right')
        self.declare_parameter('cmd_joint_names',
                               ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7'])
        self.declare_parameter('skip_if_no_subscriber', False)

        p = self.get_parameter
        self.checkpoint_dir: str = str(p('checkpoint_dir').value)
        self.policy_name: str = str(p('policy_name').value)
        self.infer_rate_hz: float = float(p('inference_rate_hz').value)
        self.cmd_joint_topic: str = str(p('cmd_joint_topic').value)
        self.cmd_joint_names: List[str] = list(p('cmd_joint_names').value)
        self.skip_if_no_sub: bool = bool(p('skip_if_no_subscriber').value)

        # ===== Publisher / 队列 / 策略先就绪，再启线程 =====
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.pub_joint_cmd = self.create_publisher(JointState, self.cmd_joint_topic, reliable_qos)

        # actions 队列：策略可能一次吐出多步
        from collections import deque
        self.actions_queue = deque(maxlen=256)

        # 加载策略
        try:
            cfg = _config.get_config(self.policy_name)
            self.policy = _policy_config.create_trained_policy(cfg, self.checkpoint_dir)
            self.get_logger().info(f"Policy loaded: {self.policy_name} from {self.checkpoint_dir}")
        except Exception as e:
            self.get_logger().error(f"Failed to load policy: {e}\n{traceback.format_exc()}")
            raise

        # 线程管理
        self._stop_evt = threading.Event()
        self._threads: List[threading.Thread] = []

        # 启动推理线程（只启一次）
        self.start_inference_thread()

    # ---------- 线程控制 ----------
    def start_inference_thread(self):
        th = threading.Thread(target=self._inference_loop, name='inference-loop', daemon=False)
        th.start()
        self._threads.append(th)
        self.get_logger().info("[threads] inference-loop started")

    # ---------- 主循环 ----------
    def _inference_loop(self):
        hz = max(1e-3, self.infer_rate_hz)
        period = 1.0 / hz
        next_t = time.monotonic()

        while rclpy.ok() and not self._stop_evt.is_set():
            now = time.monotonic()
            sleep_time = next_t - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 我们已经落后，跳过丢失的 ticks，重置 next_t
                next_t = now
            next_t += period

            try:
                # 优先把队列里已有的动作发出去，降低控制延迟
                if self.actions_queue:
                    self._publish_once()
                    continue

                out = self.aligner.step()
                if out is None:
                    # self.get_logger().warn(f"No align results")
                    continue

                t_ref, picks = out
                obs = self._build_obs(picks)  # 封装后的观测构建

                result = self.policy.infer(obs)
                actions = result.get('actions', None)
                if actions is None:
                    self.get_logger().warn("Policy returned no 'actions' field.")
                    continue

                # 统一为二维 (T, dof)
                try:
                    import numpy as np
                    act = np.asarray(actions)
                    if act.ndim == 1:
                        act = act[None, :]
                    for row in act:
                        self.actions_queue.append(row.tolist())
                except Exception:
                    # 退化处理
                    if isinstance(actions, Sequence) and actions and isinstance(actions[0], Sequence):
                        for row in actions:
                            self.actions_queue.append(list(map(float, row)))
                    else:
                        self.actions_queue.append(list(map(float, actions)))

                self._publish_once()

            except Exception as e:
                self.get_logger().error(f"inference loop error: {e}\n{traceback.format_exc()}")

    # ---------- 发布一帧 ----------
    def _publish_once(self):
        if not self.actions_queue:
            self.get_logger().warn("No actions queue.")
            return
        # 如果没有订阅者且允许跳过，直接返回
        try:
            if self.skip_if_no_sub and hasattr(self.pub_joint_cmd, "get_subscription_count"):
                if self.pub_joint_cmd.get_subscription_count() == 0:
                    return
        except Exception:
            pass

        action = self.actions_queue.popleft()

        # 维度对齐与类型转换
        dof = len(self.cmd_joint_names)
        if len(action) < dof:
            self.get_logger().warn(f"Action dof({len(action)}) < names({dof}), padding zeros.")
            action = list(action) + [0.0] * (dof - len(action))
        elif len(action) > dof:
            self.get_logger().warn(f"Action dof({len(action)}) > names({dof}), truncating.")
            action = list(action[:dof])
        else:
            action = list(map(float, action))

        js = JointState()
        js.header = Header()
        js.header.stamp = self.get_clock().now().to_msg()  # 正确的 ROS2 时间戳
        js.name = list(self.cmd_joint_names)
        js.position = action  # 若需发送速度/力矩，可加 js.velocity / js.effort
        self.pub_joint_cmd.publish(js)

    # ---------- 观测构建 ----------
    def _build_obs(self, picks):
        """
        把对齐后的 ROS2 消息转为策略需要的 numpy/tensor 字典。
        默认调用你的工具函数；你也可以改成项目内的封装。Select a mode!!
        """
        try:
            # 如果你已有该函数，直接导入并使用
            obs = transform_ros2msg_2_np(picks, self._idx_joint, self._idx_color, self._idx_depth, False)
            return obs
        except Exception as e:
            # 给出清晰的错误，便于定位
            raise RuntimeError(f"_build_obs failed, please implement it for your project: {e}")

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
