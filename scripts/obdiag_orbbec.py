#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
import logging.handlers
import subprocess
import sys
import time

import numpy as np
import pyorbbecsdk as ob

# 顶部 imports 下方加入：                                        # >>> NEW
OB_FORMAT_NAMES = {                                             # >>> NEW
    getattr(ob.OBFormat, n): n for n in dir(ob.OBFormat) if not n.startswith("_")
}                                                               # >>> NEW

def fmt_name(v):                                                # >>> NEW
    return OB_FORMAT_NAMES.get(v, str(v))                       # >>> NEW

# -------------------- 日志 --------------------
def setup_logging(level=logging.INFO, logfile=None):
    fmt = "%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
    if logfile:
        fh = logging.handlers.RotatingFileHandler(logfile, maxBytes=5 * 1024 * 1024, backupCount=2)
        fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logging.getLogger().addHandler(fh)


def sh(cmd: list[str], timeout=3) -> str:
    try:
        out = subprocess.check_output(cmd, timeout=timeout, stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="ignore").strip()
    except Exception as e:
        return f"[exec fail] {cmd}: {e}"


def sys_brief():
    log = logging.getLogger("sys")
    log.info("uname: %s", sh(["uname", "-a"]))
    log.info("python: %s", sys.version.replace("\n", " "))
    log.info("lsusb -t:\n%s", sh(["bash", "-lc", "lsusb -t | sed -n '1,120p'"]))
    log.info("usbcore autosuspend: %s", sh(["bash", "-lc", "cat /sys/module/usbcore/parameters/autosuspend || true"]))
    log.info("usbcore autoruntime_pm: %s",
             sh(["bash", "-lc", "cat /sys/module/usbcore/parameters/autoruntime_pm || true"]))


# -------------------- 适配层：Context / DeviceList --------------------
def ctx_get_devlist(ctx):
    """
    返回“设备集合”对象，可能是：
      - ob.DeviceList（有 get_count()/get_device()）
      - Python list/tuple（元素为 ob.Device）
    """
    log = logging.getLogger("compat")

    # 1) 你的版本有 query_devices()
    if hasattr(ctx, "query_devices"):
        try:
            dl = ctx.query_devices()
            if dl is not None:
                log.info("Context.query_devices() ok, type=%s", type(dl).__name__)
                return dl
        except Exception as e:
            log.warning("ctx.query_devices() failed: %s", e)

    # 2) 老名字 query_device_list()
    if hasattr(ctx, "query_device_list"):
        try:
            dl = ctx.query_device_list()
            if dl is not None:
                log.info("Context.query_device_list() ok, type=%s", type(dl).__name__)
                return dl
        except Exception as e:
            log.warning("ctx.query_device_list() failed: %s", e)

    # 3) 另一老名字 get_device_list()
    if hasattr(ctx, "get_device_list"):
        try:
            dl = ctx.get_device_list()
            if dl is not None:
                log.info("Context.get_device_list() ok, type=%s", type(dl).__name__)
                return dl
        except Exception as e:
            log.warning("ctx.get_device_list() failed: %s", e)

    raise RuntimeError(f"Context 无法枚举设备；可用属性：{dir(ctx)}")


def devlist_count(devs):
    # 支持 list/tuple
    try:
        return len(devs)
    except Exception:
        pass
    # 支持对象方法
    for name in ("get_count", "get_device_count", "device_count", "size", "getLength"):
        if hasattr(devs, name):
            try:
                return int(getattr(devs, name)())
            except Exception:
                pass
    raise RuntimeError(f"DeviceList 无法获取数量；dir={dir(devs)}")


def devlist_get(devs, i: int):
    # 支持 list/tuple
    try:
        return devs[i]
    except Exception:
        pass
    # 支持对象方法
    for name in ("get_device", "getDevice", "get_by_index", "device_at", "get"):
        if hasattr(devs, name):
            return getattr(devs, name)(i)
    raise RuntimeError(f"DeviceList 无法按索引获取设备；dir={dir(devs)}")


def devlist_get_by_serial(devs, serial: str):
    # 直接 API
    if hasattr(devs, "get_device_by_serial_number"):
        try:
            return devs.get_device_by_serial_number(serial)
        except Exception:
            pass
    # 手动遍历
    n = devlist_count(devs)
    for i in range(n):
        d = devlist_get(devs, i)
        try:
            info = d.get_device_info()
            sn = getattr(info, "get_serial_number", lambda: None)()
            if sn == serial:
                return d
        except Exception:
            continue
    return None


# -------------------- 适配层：StreamProfileList --------------------
def list_count(lst):
    try:
        return len(lst)
    except Exception:
        pass
    for name in ("get_count", "size", "getLength"):
        if hasattr(lst, name):
            try:
                return int(getattr(lst, name)())
            except Exception:
                pass
    raise RuntimeError(f"无法获取列表长度；dir={dir(lst)}")


def list_get(lst, i: int):
    try:
        return lst[i]
    except Exception:
        pass
    for name in ("get", "get_profile", "getProfile", "at"):
        if hasattr(lst, name):
            return getattr(lst, name)(i)
    raise RuntimeError(f"无法索引列表；dir={dir(lst)}")

def dump_all_profiles(dev):                                                     # >>> NEW
    log = logging.getLogger("dump")
    pipe = ob.Pipeline(dev)
    try:
        color_list = pipe.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        depth_list = pipe.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
    except Exception as e:
        log.error("get_stream_profile_list error: %s", e)
        return 1

    def _iter(lst):
        try:
            n = len(lst)
            for i in range(n):
                yield lst[i]
        except Exception:
            # 兼容旧接口
            try:
                n = lst.get_count()
                for i in range(n):
                    yield lst.get(i) if hasattr(lst, "get") else lst.get_profile(i)
            except Exception as e:
                log.error("iterate profiles failed: %s", e)
                return

    log.info("---- COLOR profiles ----")
    for p in _iter(color_list) or []:
        try:
            log.info("  %dx%d @ %sfps  fmt=%s",
                     p.get_width(), p.get_height(), int(round(p.get_fps())), fmt_name(p.get_format()))
        except Exception:
            pass

    log.info("---- DEPTH profiles ----")
    for p in _iter(depth_list) or []:
        try:
            log.info("  %dx%d @ %sfps  fmt=%s",
                     p.get_width(), p.get_height(), int(round(p.get_fps())), fmt_name(getattr(p, "get_format", lambda: "?")()))
        except Exception:
            pass
    return 0


# -------------------- Profile 选择 --------------------
def choose_profiles(pipeline: ob.Pipeline, w: int, h: int, fps: int, use_depth: bool, align: str, force_color: str | None = None):
    cfg = ob.Config()

    # ---------- COLOR ----------
    color_list = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)

    # 决定候选格式优先级：若指定 --force-color 就只试它
    if force_color:
        try_order = [getattr(ob.OBFormat, force_color)]
    else:
        # 一些 Gemini 固件的 RGB 直出不稳定；按以下顺序更稳：MJPG -> YUYV -> UYVY -> RGB -> NV12 -> NV21 -> I420
        prefer = ["MJPG", "YUYV", "UYVY", "RGB", "NV12", "NV21", "I420"]
        try_order = [getattr(ob.OBFormat, n) for n in prefer if hasattr(ob.OBFormat, n)]

    color = None
    # 先在目标分辨率+fps 下找
    for fmt in try_order:
        for i in range(list_count(color_list)):
            p = list_get(color_list, i)
            try:
                if p.get_format() == fmt and p.get_width()==w and p.get_height()==h and int(round(p.get_fps()))==fps:
                    color = p; break
            except Exception:
                continue
        if color: break

    # 次选：目标分辨率，fps 任意
    if color is None:
        for fmt in try_order:
            for i in range(list_count(color_list)):
                p = list_get(color_list, i)
                try:
                    if p.get_format()==fmt and p.get_width()==w and p.get_height()==h:
                        color = p; break
                except Exception:
                    continue
            if color: break

    if color is None:
        logging.error("No COLOR profile %dx%d found (tried fmts=%s)", w, h, [fmt_name(x) for x in try_order])
        return None

    cfg.enable_stream(color)

    # ---------- DEPTH ----------
    depth = None
    if use_depth:
        depth_list = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
        # 与之前相同的选择逻辑...
        nd = list_count(depth_list)
        for i in range(nd):
            dp = list_get(depth_list, i)
            try:
                if (dp.get_width()==w and dp.get_height()==h and int(round(dp.get_fps()))==fps):
                    depth = dp; break
            except Exception:
                continue
        if depth is None:
            for i in range(nd):
                dp = list_get(depth_list, i)
                try:
                    if dp.get_width()==w and dp.get_height()==h:
                        depth = dp; break
                except Exception:
                    continue
        if depth is None:
            logging.error("No DEPTH profile %dx%d found", w, h)
            return None
        cfg.enable_stream(depth)

    # ---------- 对齐 ----------
    a = align.lower()
    if a == "hw":
        cfg.set_align_mode(ob.OBAlignMode.HW_MODE)
    elif a == "sw":
        cfg.set_align_mode(ob.OBAlignMode.SW_MODE)
    # a == none: 不设

    logging.info("Selected COLOR %dx%d@%dfps(%s), DEPTH=%s",
                 color.get_width(), color.get_height(), int(round(color.get_fps())), fmt_name(color.get_format()),
                 f"{depth.get_width()}x{depth.get_height()}@{int(round(depth.get_fps()))}" if depth else "OFF")

    return cfg

# -------------------- 诊断主流程 --------------------
def main():
    ap = argparse.ArgumentParser(description="Orbbec camera diagnostic")
    ap.add_argument("--serial", required=False, help="device serial; leave empty to only list")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--timeout-ms", type=int, default=800)
    ap.add_argument("--use-depth", action="store_true")
    ap.add_argument("--align", choices=["hw", "sw", "none"], default="sw")
    ap.add_argument("--logfile", default="/tmp/obdiag.log")
    ap.add_argument("--dump-profiles", action="store_true", help="List all color/depth profiles and exit")  # >>> NEW
    ap.add_argument("--force-color", type=str, default=None, help="Force color OBFormat, e.g. MJPG,YUYV,RGB")  # >>> NEW
    ap.add_argument("--sdk-log", action="store_true", help="Enable SDK console log at INFO level")  # >>> NEW

    args = ap.parse_args()

    setup_logging(logging.INFO, args.logfile)
    sys_brief()

    ctx = ob.Context()
    if args.sdk_log:  # >>> NEW
        try:
            ctx.set_logger_to_console(True)
            # 可选：也可以写入文件：ctx.set_logger_to_file(True, "/tmp/obdiag_sdk.log", True)
            if hasattr(ob, "OBLogSeverity"):
                ctx.set_logger_level(ob.OBLogSeverity.INFO)
        except Exception as e:
            logging.getLogger("compat").warning("enable SDK log failed: %s", e)

    # 某些版本要显式打开设备热插拔/网络枚举；加上不影响
    if hasattr(ctx, "enable_net_device_enumeration"):
        try:
            ctx.enable_net_device_enumeration(True)
        except Exception:
            pass

    devs = ctx_get_devlist(ctx)
    n = devlist_count(devs)
    logging.info("Found %d Orbbec devices.", n)
    for i in range(n):
        d = devlist_get(devs, i)
        try:
            info = d.get_device_info()
            name = getattr(info, "get_name", lambda: "?")()
            sn = getattr(info, "get_serial_number", lambda: "?")()
            fw = getattr(info, "get_firmware_version", lambda: "?")()
            usb = getattr(info, "get_usb_type", lambda: "?")()
            logging.info("  #%d: name=%s  serial=%s  fw=%s  usb=%s", i, name, sn, fw, usb)
        except Exception as e:
            logging.warning("  #%d: <info error: %s>", i, e)

    if not args.serial:
        logging.info("No --serial provided; listing only. Exiting.")
        return 0

    dev = devlist_get_by_serial(devs, args.serial)
    if dev is None:
        logging.error("Device with serial %s not found.", args.serial)
        return 2

    pipe = ob.Pipeline(dev)
    cfg = choose_profiles(pipe, args.width, args.height, args.fps, args.use_depth, args.align)
    if cfg is None:
        return 3

    pipe.start(cfg)
    logging.info("Pipeline started. Warming up 2s...")
    t_warm = time.monotonic()
    warm_ok = 0
    while time.monotonic() - t_warm < 2.0:
        try:
            fr = pipe.wait_for_frames(1000)
            if fr:
                warm_ok += 1
        except Exception:
            pass
    logging.info("Warm frames: %d", warm_ok)

    T = args.seconds
    deadline = time.monotonic() + T
    delivered = 0
    miss_color = 0
    miss_depth = 0
    timeouts = 0
    slow_wait = 0
    wait_ms_max = 0.0

    prev_ts = None
    inter_ms_list = []
    wait_ms_list = []

    while time.monotonic() < deadline:
        t0 = time.perf_counter()
        try:
            fr = pipe.wait_for_frames(args.timeout_ms)
        except Exception as e:
            logging.warning("wait_for_frames error: %s", e)
            timeouts += 1
            continue

        if not fr:
            timeouts += 1
            logging.warning("frames=None (timeout %d ms)", args.timeout_ms)
            continue

        t1 = time.perf_counter()
        wait_ms = (t1 - t0) * 1000.0
        wait_ms_list.append(wait_ms)
        wait_ms_max = max(wait_ms_max, wait_ms)
        if wait_ms > 500:
            slow_wait += 1
            logging.warning("slow wait_for_frames: %.1f ms", wait_ms)

        try:
            cf = fr.get_color_frame()
            df = fr.get_depth_frame() if args.use_depth else None
        except Exception as e:
            logging.warning("get_*_frame error: %s", e)
            continue

        if not cf:
            miss_color += 1
            continue
        if args.use_depth and not df:
            miss_depth += 1
            continue

        delivered += 1
        now = time.monotonic()
        if prev_ts is not None:
            inter_ms_list.append((now - prev_ts) * 1000.0)
        prev_ts = now

    pipe.stop()

    logging.info("=== Summary (serial=%s, %dx%d@%dfps, T=%.1fs, depth=%s, align=%s) ===",
                 args.serial, args.width, args.height, args.fps, args.seconds, args.use_depth, args.align)
    eff_fps = delivered / args.seconds if args.seconds > 0 else 0.0
    logging.info("Delivered frames: %d (eff_fps=%.2f)", delivered, eff_fps)
    logging.info("Timeouts: %d (%.1f%%)", timeouts, 100.0 * timeouts / max(1, timeouts + delivered))
    logging.info("Missing color: %d, Missing depth: %d", miss_color, miss_depth)
    if inter_ms_list:
        p95 = float(np.percentile(inter_ms_list, 95))
        logging.info("Inter-frame: mean=%.1fms  p95=%.1fms  max=%.1fms (ideal %.1fms)",
                     float(np.mean(inter_ms_list)), p95, float(np.max(inter_ms_list)), 1000.0 / max(1, args.fps))
    if wait_ms_list:
        logging.info("wait_for_frames: mean=%.1fms  max=%.1fms  slow(>500ms)=%d",
                     float(np.mean(wait_ms_list)), wait_ms_max, slow_wait)

    if eff_fps < args.fps * 0.7 or timeouts > 0:
        logging.info("Hints: 分散 USB 口/控制器、提高 --timeout-ms、禁用 autosuspend、减少并发相机/降低分辨率或深度流。")

    if delivered == 0:
        return 10
    if eff_fps < args.fps * 0.3:
        return 11
    return 0


if __name__ == "__main__":
    sys.exit(main())
