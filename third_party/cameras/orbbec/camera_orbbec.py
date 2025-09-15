import logging
import threading
import time
from threading import Event, Lock, Thread
from typing import Any, Optional, Union, List, Dict, Tuple

import cv2
import numpy as np
import pyorbbecsdk as ob

from .configuration_orbbec import OrbbecCameraConfig
from .. import ColorMode
from ..errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..utils import capture_timestamp_utc
from ..OButils import i420_to_bgr, nv12_to_bgr, nv21_to_bgr
from ..camera import Camera


def frame_to_rgb_image(frame: ob.VideoFrame) -> Optional[np.ndarray]:
    """将 OB.VideoFrame 转为 RGB ndarray，返回 None 表示不支持的格式或解码失败。"""
    if frame is None:
        return None

    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    buf = frame.get_data()  # memoryview / bytes
    raw = np.frombuffer(buf, dtype=np.uint8).copy()

    try:
        if fmt == ob.OBFormat.RGB:
            # 已经是 RGB，直接 reshape
            rgb = raw.reshape(height, width, 3)
            return np.ascontiguousarray(rgb)

        elif fmt == ob.OBFormat.BGR:
            bgr = raw.reshape(height, width, 3)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return np.ascontiguousarray(rgb)

        elif fmt == ob.OBFormat.YUYV:
            # OpenCV 期望 (H, W, 2)
            yuyv = raw.reshape(height, width, 2)
            rgb = cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUYV)
            return np.ascontiguousarray(rgb)

        elif fmt == ob.OBFormat.UYVY:
            uyvy = raw.reshape(height, width, 2)
            rgb = cv2.cvtColor(uyvy, cv2.COLOR_YUV2RGB_UYVY)
            return np.ascontiguousarray(rgb)

        elif fmt == ob.OBFormat.MJPG:
            # imdecode 得到 BGR，需要再转 RGB
            bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return np.ascontiguousarray(rgb)

        elif fmt == ob.OBFormat.I420:
            # I420 (YUV420 planar) 在 OpenCV 中 reshape 为 (H*3/2, W)
            yuv = raw.reshape(height * 3 // 2, width)
            rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
            return np.ascontiguousarray(rgb)

        elif fmt == ob.OBFormat.NV12:
            yuv = raw.reshape(height * 3 // 2, width)
            rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV12)
            return np.ascontiguousarray(rgb)

        elif fmt == ob.OBFormat.NV21:
            yuv = raw.reshape(height * 3 // 2, width)
            rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV21)
            return np.ascontiguousarray(rgb)

        else:
            logging.info(f"Unsupported color format: {fmt}")
            return None

    except Exception as e:
        logging.warning(f"Failed to convert frame to RGB: {e}")
        return None


class TemporalFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result


SERIAL_NUMBER_INDEX = 1

MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm


class OrbbecCamera(Camera):
    def __init__(
            self,
            config: OrbbecCameraConfig,
    ):
        super().__init__(config)

        self.fps = config.fps
        self.width = config.width
        self.height = config.height
        self.use_depth = config.use_depth
        self.index_or_path = config.index_or_path
        self.warmup_s = config.warmup_s
        self.camera_pipeline = None
        self.is_connected_used = False
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: np.ndarray | None = None
        self.new_frame_event: Event = Event()
        # self.logs = {}
        # self.temporal_filter = TemporalFilter(config.TemporalFilter_alpha)
        self.device = config.device_list.get_device_by_serial_number(self.index_or_path)

    @property
    def is_connected(self) -> bool:
        """Checks if the camera is currently connected and opened."""
        return self.is_connected_used

    @staticmethod
    def find_cameras() -> List[Dict[str, Any]]:
        logging.error("this method is not implemented")
        return None

    def _get_stream_config(self, pipeline: ob.Pipeline):
        """
        Gets the stream configuration for the pipeline.

        Args:
            pipeline (Pipeline): The pipeline object.

        Returns:
            Config: The stream configuration.
        """
        config = ob.Config()
        try:
            # Get the list of color stream profiles
            profile_list = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
            assert profile_list is not None

            # Iterate through the color stream profiles
            for i in range(len(profile_list)):
                color_profile = profile_list[i]

                # Check if the color format is RGB
                if color_profile.get_format() != ob.OBFormat.RGB:
                    continue

                # Get the list of hardware aligned depth-to-color profiles
                hw_d2c_profile_list = pipeline.get_d2c_depth_profile_list(color_profile, ob.OBAlignMode.HW_MODE)
                if len(hw_d2c_profile_list) == 0:
                    continue

                # Get the first hardware aligned depth-to-color profile
                hw_d2c_profile = hw_d2c_profile_list[0]
                print("hw_d2c_profile: ", hw_d2c_profile)

                # Enable the depth and color streams
                config.enable_stream(hw_d2c_profile)
                config.enable_stream(color_profile)

                # Set the alignment mode to hardware alignment
                config.set_align_mode(ob.OBAlignMode.HW_MODE)
                return config
        except Exception as e:
            print(e)
            return None
        return None

    def _get_stream_config_v2(self, pipeline: ob.Pipeline):
        """
        以软件对齐（SW_MODE）方式开启 640x480 的彩色与深度流。
        不使用 get_d2c_depth_profile_list(...)，避免触发硬件 D2C 管线。
        """
        cfg = ob.Config()
        want_w, want_h, want_fps = int(self.width), int(self.height), int(self.fps)

        try:
            # ---------- 选择 COLOR (RGB) 640x480 ----------
            color_list = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
            color_profile = None

            # 先找分辨率+fps 都匹配的
            for i in range(len(color_list)):
                p = color_list[i]
                if (p.get_format() == ob.OBFormat.RGB and
                        p.get_width() == want_w and p.get_height() == want_h and
                        int(round(p.get_fps())) == want_fps):
                    color_profile = p
                    break

            # 次选：分辨率匹配，fps 任意
            if color_profile is None:
                for i in range(len(color_list)):
                    p = color_list[i]
                    if (p.get_format() == ob.OBFormat.RGB and
                            p.get_width() == want_w and p.get_height() == want_h):
                        color_profile = p
                        break

            if color_profile is None:
                logging.error("No RGB COLOR profile %dx%d found (want fps=%d)", want_w, want_h, want_fps)
                return None

            # ---------- 选择 DEPTH 640x480 ----------
            depth_list = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
            depth_profile = None

            # 先找分辨率+fps 都匹配的
            for i in range(len(depth_list)):
                dp = depth_list[i]
                if (dp.get_width() == want_w and dp.get_height() == want_h and
                        int(round(dp.get_fps())) == want_fps):
                    depth_profile = dp
                    break

            # 次选：分辨率匹配，fps 任意
            if depth_profile is None:
                for i in range(len(depth_list)):
                    dp = depth_list[i]
                    if dp.get_width() == want_w and dp.get_height() == want_h:
                        depth_profile = dp
                        break

            if depth_profile is None:
                logging.error("No DEPTH profile %dx%d found (want fps=%d)", want_w, want_h, want_fps)
                return None

            # ---------- 启动两路流（不走硬件 D2C 对齐） ----------
            cfg.enable_stream(color_profile)
            cfg.enable_stream(depth_profile)

            # 使用软件对齐（如需关闭对齐，直接注释掉下一行）
            cfg.set_align_mode(ob.OBAlignMode.SW_MODE)

            # 保存引用，防止 Python GC 提前释放导致底层悬挂指针
            self._bound_profiles = (color_profile, depth_profile)

            logging.info(
                f"Selected COLOR {color_profile.get_width()}x{color_profile.get_height()}@{int(round(color_profile.get_fps()))}fps (RGB), "
                f"DEPTH {depth_profile.get_width()}x{depth_profile.get_height()}@{int(round(depth_profile.get_fps()))}fps (SW align)")

            return cfg

        except Exception:
            logging.exception("prepare stream config (SW align) failed")
            return None

    def connect(self, warmup: bool = True):
        if self.is_connected_used:
            raise DeviceAlreadyConnectedError("OrbbecCamera is readyConnected")

        logging.info("\033[32mHello! Orbbec!\033[0m")

        self.camera_pipeline = ob.Pipeline(self.device)

        ob_config = self._get_stream_config_v2(self.camera_pipeline)
        if ob_config is None:
            logging.error("Camera connection failed")
            return
        self.camera_pipeline.start(ob_config)

        self.is_connected_used = True
        if warmup:
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                self.read()
                time.sleep(0.1)
        logging.info(f"Camera {self.index_or_path} connected!")

    def _post_process_depth_frame(self, depth_frame):
        if not depth_frame:
            logging.error("No depth frame received")
            return None

        h, w = depth_frame.get_height(), depth_frame.get_width()
        buf = depth_frame.get_data()  # 原始字节
        depth_u16 = np.frombuffer(buf, dtype=np.uint16, count=h * w).reshape(h, w, 1).copy()  # 无损 + 自有拷贝
        # 可选：确认是C连续
        assert depth_u16.flags['C_CONTIGUOUS']

        return depth_u16

    def _post_process_depth_frame_v2(self,
                                     depth_frame,
                                     *,
                                     pack_to_rgb_u8: bool = True,
                                     order: str = "HI_LO",  # "HI_LO": R=高8位, G=低8位；"LO_HI": R=低8位, G=高8位
                                     b_fill: int = 0):  # 第三通道填充值(0~255)，仅为占位/标记
        """
        读取深度帧为 uint16，并可选打包为 (H,W,3) 的 RGB uint8：
          - pack_to_rgb_u8=True  -> 返回 (H,W,3) uint8, 其中两通道存放 uint16 的高/低8位，可逆
          - pack_to_rgb_u8=False -> 返回 (H,W,1) uint16（原样）
        """
        if not depth_frame:
            logging.error("No depth frame received")
            return None

        h, w = depth_frame.get_height(), depth_frame.get_width()

        # 原始字节 -> uint16 深度图（单位通常为毫米）
        buf = depth_frame.get_data()
        depth_u16 = np.frombuffer(buf, dtype=np.uint16, count=h * w).reshape(h, w).copy()
        assert depth_u16.flags['C_CONTIGUOUS'], "depth_u16 should be C-contiguous after copy()"

        if not pack_to_rgb_u8:
            return depth_u16[..., None]  # (H,W,1) uint16

        # 显式位运算(端序无关)拆分高/低8位
        hi = ((depth_u16 >> 8) & 0xFF).astype(np.uint8)
        lo = (depth_u16 & 0xFF).astype(np.uint8)

        if order.upper() == "HI_LO":
            r, g = hi, lo
        elif order.upper() == "LO_HI":
            r, g = lo, hi
        else:
            raise ValueError("order must be 'HI_LO' or 'LO_HI'")

        # 第三通道占位（可做校验/标记：0 或 255 或 r^g 等）
        b = np.full_like(r, np.uint8(b_fill))

        depth_rgb_u8 = np.stack((r, g, b), axis=-1)  # (H,W,3) uint8
        assert depth_rgb_u8.flags['C_CONTIGUOUS']
        return depth_rgb_u8

    def read(self, color_mode: ColorMode | None = None) -> Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """
        Returns:
            - use_depth == False: 返回 color_image (H, W, 3) 的 np.ndarray
            - use_depth == True : 返回 (color_image, depth_map)
            - 发生异常/丢帧: 返回 None
        """
        if not self.is_connected_used:
            raise DeviceNotConnectedError(f"{self.index_or_path} is not connected.")

        start_time = time.perf_counter()
        try:
            frames = self.camera_pipeline.wait_for_frames(100)
        except Exception as e:
            logging.warning("wait_for_frames error: %s", e);
            return None
        if frames is None:
            logging.warning(f"{self.index_or_path} No frames received")
            return None

        # 获取彩色帧与（可选）深度帧
        try:
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame() if self.use_depth else None
        except Exception as e:
            logging.warning("get_*_frame error: %s", e);
            return None

        if not color_frame or (self.use_depth and (not depth_frame)):
            # logging.error(f"Camera {self.index_or_path} receives no frames, color:{not not color_frame}, depth:{not not depth_frame}")
            return None
        # logging.error(f"Camera {self.index_or_path} receives all frames, color:{not not color_frame}, depth:{not not depth_frame}")

        try:
            # 转成 numpy 彩色图
            color_image = frame_to_rgb_image(color_frame)
            if color_image is None:
                logging.info("failed to convert color frame to image")
                return None

            # 仅在启用深度时才做后处理与返回
            if self.use_depth:
                depth_map = self._post_process_depth_frame_v2(depth_frame)  # 期望返回 np.ndarray
                if depth_map is None:
                    logging.info("failed to post-process depth frame")
                    return None
                result: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]] = (color_image, depth_map)
            else:
                result: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]] = color_image

        except Exception as e:
            logging.warning("post_process_depth_frame error: %s", e)

        # 同步原有日志与成员
        # self.logs["delta_timestamp_s"] = time.perf_counter() - start_time
        # self.logs["timestamp_utc"] = capture_timestamp_utc()

        return result

    def _read_loop(self):
        """
        Background loop:
        - self.read() -> color 或 (color, depth)
        - 赋值到 self.lastest_color_image / self.latest_depth_image
        - 通知有新帧
        """
        while not self.stop_event.is_set():
            # try:
            result = self.read()  # None | np.ndarray | (np.ndarray, np.ndarray)
            if result is None:
                time.sleep(0.001)
                continue

            with self.frame_lock:
                # 你要求的字段
                self.latest_frame = result
                # self.latest_color_frame = color_image
                # self.latest_depth_frame = depth_map  # 可能是 None
                self.new_frame_event.set()

        # except DeviceNotConnectedError:
        #     break
        # except Exception:
        #     # 使用 exc_info=True 会自动附加异常信息和堆栈跟踪
        #     camera_id = self.index_or_path if self.index_or_path is not None else "Unknown"
        #     logging.error(f"Unhandled exception in background thread for camera {camera_id}", exc_info=True)
        #     # logging.error(f"Error reading frame in background thread for: {e}")

    def _start_read_thread(self) -> None:
        """Starts or restarts the background read thread if it's not running."""
        if self.thread is not None and self.thread.is_alive():
            if self.stop_event is not None:
                self.stop_event.set()
            self.thread.join(timeout=2.0)

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self.index_or_path}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 200) -> Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """
        Reads the latest available frame asynchronously.

        This method retrieves the most recent frame captured by the background
        read thread. It does not block waiting for the camera hardware directly,
        but may wait up to timeout_ms for the background thread to provide a frame.

        Args:
            timeout_ms (float): Maximum time in milliseconds to wait for a frame
                to become available. Defaults to 200ms (0.2 seconds).

        Returns:
            - use_depth == False: 返回 color_image (H, W, 3) 的 np.ndarray
            - use_depth == True : 返回 (color_image, depth_map)
            - 发生异常/丢帧: 返回 None

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame becomes available within the specified timeout.
            RuntimeError: If an unexpected error occurs.
        """
        if not self.is_connected_used:
            raise DeviceNotConnectedError(f"{self.index_or_path} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self.index_or_path} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self.index_or_path}.")

        return frame

    def disconnect(self):
        if not self.is_connected_used:
            raise DeviceNotConnectedError(
                f"Orbbec camera ({self.index_or_path}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is not None and self.thread.is_alive():
            self._stop_read_thread()

        self.camera_pipeline.stop()
        self.camera_pipeline = None

        self.is_connected_used = False
        logging.info(f"{self.index_or_path} disconnected.")

    def test_read(self):
        if self.thread is None:
            self.stop_event = threading.Event()
            self.thread = Thread(target=self.test_loop, args=())
            self.thread.daemon = True
            self.thread.start()
