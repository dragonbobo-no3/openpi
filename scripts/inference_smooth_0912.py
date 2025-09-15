import numpy as np
import matplotlib.pyplot as plt
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import time
import argparse
import draccus
import logging
import multiprocessing as mp
import collections
import yaml
from scripts.numpy_logger import NumpyCSVLogger

from openpi.policies import policy_config as _policy_config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training import config as _config
from third_party.agilex.agilexfollower import AlohaAgileXFollower
from third_party.agilex.agilexconfig import AlohaAgileXFollowerConfig
from third_party.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from third_party.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig

def make_camera_config(cfg: dict):
    t = cfg.get('type')
    if t == 'opencv':
        return OpenCVCameraConfig(
            index_or_path=cfg['index_or_path'],
            width=cfg.get('width', 640),
            height=cfg.get('height', 480),
            fps=cfg.get('fps', 30),
        )
    elif t == 'orbbec':
        return OrbbecCameraConfig(
            index_or_path=cfg['index_or_path'],
            width=cfg.get('width', 640),
            height=cfg.get('height', 480),
            fps=cfg.get('fps', 30),
        )
    else:
        raise ValueError(f"Unsupported camera type: {t!r}")

# ---------- 子进程：推理循环 ----------
def inference_worker(
    in_q: mp.Queue,
    out_q: mp.Queue,
    config,
    checkpoint_dir,
):

    # 1. 只在该进程里加载一次模型 / CUDA
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    while True:
        item = in_q.get()
        if item is None:            # 收到结束标识
            del policy
            break
        idx, obs = item       # idx 用来对应主进程里的顺序
        start_time = time.time()
        result = policy.infer(obs)
        infer_time = time.time() - start_time
        print(f"Step {idx}: infer time = {infer_time:.4f} seconds")
        out_q.put((idx, result["actions"]))

def linear_transition(old_actions, new_actions):
    """
    线性插值平滑衔接，返回平滑后的动作序列。
    old_actions: list[np.ndarray]，未执行的旧动作
    new_actions: np.ndarray:shape=(N, action_dim)，新推理动作
    返回:list[np.ndarray]，平滑衔接后的动作序列
    """
    n_old = len(old_actions)
    n_interp = min(n_old, len(new_actions))
    result = []
    for i_interp in range(n_interp):
        t = (i_interp + 1) / (n_interp + 1)
        interp_action = (1 - t) * old_actions[i_interp] + t * new_actions[i_interp]
        result.append(interp_action)
    for a in new_actions[n_interp:]:
        result.append(a)
    return result

def cubic_transition(old_actions, new_actions):
    """
    三次插值平滑衔接，返回平滑后的动作序列。
    old_actions: list[np.ndarray]，未执行的旧动作
    new_actions: np.ndarrayshape=(N, action_dim)，新推理动作
    返回:list[np.ndarray]，平滑衔接后的动作序列
    """
    n_old = len(old_actions)
    n_interp = min(n_old, len(new_actions))
    result = []
    for i_interp in range(n_interp):
        t = (i_interp + 1) / (n_interp + 1)
        # 三次Hermite插值（ease in/out）：h(t) = 3t^2 - 2t^3
        h = 3 * t**2 - 2 * t**3
        interp_action = (1 - h) * old_actions[i_interp] + h * new_actions[i_interp]
        result.append(interp_action)
    for a in new_actions[n_interp:]:
        result.append(a)
    return result

def quintic_transition(old_actions, new_actions):
    """
    五次多项式插值平滑衔接，返回平滑后的动作序列。
    old_actions: list[np.ndarray]，未执行的旧动作
    new_actions: np.ndarray,shape=(N, action_dim)，新推理动作
    返回:list[np.ndarray]，平滑衔接后的动作序列
    """
    n_old = len(old_actions)
    n_interp = min(n_old, len(new_actions))
    result = []
    for i_interp in range(n_interp):
        t = (i_interp + 1) / (n_interp + 1)
        # 五次多项式插值：h(t) = 10t^3 - 15t^4 + 6t^5
        h = 10 * t**3 - 15 * t**4 + 6 * t**5
        interp_action = (1 - h) * old_actions[i_interp] + h * new_actions[i_interp]
        result.append(interp_action)
    for a in new_actions[n_interp:]:
        result.append(a)
    return result

def ema_transition(old_actions, new_actions, alpha=0.7):
    """
    指数加权平滑(EMA)，返回平滑后的动作序列。
    old_actions: list[np.ndarray]，未执行的旧动作
    new_actions: np.ndarray,shape=(N, action_dim)，新推理动作
    alpha: 新动作权重,0~1
    返回:list[np.ndarray]，平滑衔接后的动作序列
    """
    n_old = len(old_actions)
    n_interp = min(n_old, len(new_actions))
    result = []
    for i_interp in range(n_interp):
        interp_action = alpha * new_actions[i_interp] + (1 - alpha) * old_actions[i_interp]
        result.append(interp_action)
    for a in new_actions[n_interp:]:
        result.append(a)
    return result

def main():
    parser = argparse.ArgumentParser(description="Inference script for AgileX follower robot")
    parser.add_argument("--port", type=str, required=True, help="port name")
    parser.add_argument("--checkpoint_dir", required=True, type=str, help="path to checkpoint directory")
    parser.add_argument("--fps", type=int, required=False, default=30, help="frames per second")
    parser.add_argument("--task", type=str, required=True, help="task prompt")
    parser.add_argument("--id", type=str, required=False, help="robot id", default="left")
    parser.add_argument("--cameras", type=str, required=False, help="camera config yaml", default=None)
    parser.add_argument("--max_relative_target", type=int, required=False, default=None)
    parser.add_argument("--use_degrees", action="store_true")
    parser.add_argument("--action_steps", type=int, required=False, default=20, help="number of action steps to execute before next inference")
    parser.add_argument("--smooth_type", type=str, default="cubic", choices=["linear", "cubic", "quintic", "ema"], help="动作平滑策略: linear/cubic/quintic/ema")
    parser.add_argument("--ema_alpha", type=float, default=0.7, help="EMA平滑时新动作权重alpha,0~1")
    parser.add_argument("--align_mode", type=str, default="step", choices=["step", "euclidean"], help="新动作对齐方式: step(步数) 或 euclidean(欧氏距离)")
    args = parser.parse_args()

    logger = NumpyCSVLogger("logs/1a.csv", mode="w")
    print_log = False

    # 解析摄像头配置
    if args.cameras is not None:
        raw = yaml.safe_load(args.cameras)
        cameras = {name: make_camera_config(cfg) for name, cfg in raw.items()}
    else:
        cameras = {}


    robot_config = AlohaAgileXFollowerConfig(
        port=args.port,
        id=args.id,
        cameras=cameras,
        max_relative_target=args.max_relative_target,
        use_degrees=args.use_degrees,
    )
    # 选择配置和 checkpoint
    robot = AlohaAgileXFollower(robot_config)

    # Load pretrained policy
    # policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

    # ==== 1. 启动推理子进程 ====
    ctx = mp.get_context("spawn")        # "spawn" 更安全，尤其 CUDA
    in_q: mp.Queue = ctx.Queue(maxsize=4)   # 根据实时性调节 maxsize
    out_q: mp.Queue = ctx.Queue(maxsize=4)
    config = _config.get_config("pi0_agileX")
    checkpoint_dir = args.checkpoint_dir #"/home/agx/jemodel/test/40000"
    logging.info(f"policy path: {checkpoint_dir}")

    proc = ctx.Process(
        target=inference_worker,
        args=(in_q, out_q, config, checkpoint_dir)
    )
    proc.daemon = True
    proc.start()

    robot.connect()
    i, sent_idx, recv_idx = 0, 0, 0
    kMaxTimeStamps = 600000

    # rows = []
    step = 1
    step_time = step/(args.fps)
    prompt = args.task
    tokenizer = PaligemmaTokenizer()
    tokenized, mask = tokenizer.tokenize(prompt)

    action_queue = collections.deque()  # 存储当前动作序列
    waiting_for_infer = False
    action_step_counter = 0  # 记录已执行的动作步数
    first = True

    while i < kMaxTimeStamps:
        t0 = time.perf_counter()

        # 1. 只有在执行了action_steps步后才采集观测并推理
        if not waiting_for_infer and (action_step_counter >= args.action_steps or first):
            first = False
            obs = robot.get_observation()
            obs["state"] = obs["state"]
            obs["tokenized_prompt"] = tokenized[None]
            obs["tokenized_prompt_mask"] = mask[None]
            obs["token_ar_mask"] = None
            obs["token_loss_mask"] = None
            try:
                in_q.put_nowait((sent_idx, obs))
                sent_idx += 1
                waiting_for_infer = True
                action_step_counter = 0
            except mp.queues.Full:
                logging.debug("inference queue full, dropping frame")

        # 2. 如果有新推理结果，立即清空并更新 action_queue
        try:
            idx, action_vals = out_q.get_nowait()
            recv_idx = idx
            logging.debug(f"got result #{recv_idx}")

            # 1. 记录未执行的旧动作
            old_actions = list(action_queue)
            action_queue.clear()

            # 2. 新推理动作起点
            if args.align_mode == "step":
                start_idx = action_step_counter
            elif args.align_mode == "euclidean" and len(old_actions) > 0 and len(action_vals) > 0:
                # 取旧队列第一个动作，与新动作序列做欧氏距离最小匹配
                old_action = old_actions[0]
                dists = np.linalg.norm(action_vals - old_action, axis=1)
                start_idx = int(np.argmin(dists))
            else:
                start_idx = 0
            new_actions = action_vals[start_idx:]

            # 3. 平滑衔接（可通过参数切换）
            if args.smooth_type == "linear":
                smooth_actions = linear_transition(old_actions, new_actions)
            elif args.smooth_type == "cubic":
                smooth_actions = cubic_transition(old_actions, new_actions)
            elif args.smooth_type == "quintic":
                smooth_actions = quintic_transition(old_actions, new_actions)
            elif args.smooth_type == "ema":
                smooth_actions = ema_transition(old_actions, new_actions, alpha=args.ema_alpha)
            else:
                raise ValueError(f"Unknown smooth_type: {args.smooth_type}")
            for a in smooth_actions:
                action_queue.append(a)

            waiting_for_infer = False
        except mp.queues.Empty:
            pass

        # 3. 如果 action_queue 有动作，发给 robot
        if action_queue:
            action_to_send = action_queue.popleft()
            if print_log:
                logger.log(action_to_send[:7])
            robot.send_action_np(action_to_send[:7])
            action_step_counter += 1
            # print(f'publish an action:{time.perf_counter()},action counter:{action_step_counter}')

        # 2.5 统计
        i += 1
        dt_s = time.perf_counter() - t0
        # print(f"loop {i} dt={dt_s:.3f} s")
        time.sleep(max(step_time - dt_s,0))

    # ==== 3. 结束 ====
    in_q.put(None)      # 通知子进程退出
    proc.join()
    robot.disconnect()

if __name__ == "__main__":
    main()