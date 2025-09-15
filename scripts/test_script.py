from operator import truediv

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


# ---------- 子进程：推理循环 ----------
# def inference_worker(
#         in_q: mp.Queue,
#         out_q: mp.Queue,
#         config,
#         checkpoint_dir,
# ):
#     # 1. 只在该进程里加载一次模型 / CUDA
#     policy = _policy_config.create_trained_policy(config, checkpoint_dir)
#     logger_action = NumpyCSVLogger("logs/action_0819_2_cameras.csv", mode="w")
#     logger_obs = NumpyCSVLogger("logs/obs_0819_2_cameras.csv", mode="w")
#     while True:
#         item = in_q.get()
#         if item is None:  # 收到结束标识
#             del policy
#             break
#         idx, obs = item  # idx 用来对应主进程里的顺序
#         start_time = time.time()
#         result = policy.infer(obs)
#         infer_time = time.time() - start_time
#         print(f"Step {idx}: infer time = {infer_time:.4f} seconds")
#         logger_obs.log(obs['state'])
#         for row in result['actions']:
#             logger_action.log(row)
#         # print(f"Inference result: {obs['state'].shape}: prediction: {result['actions'].shape}")
#         out_q.put((idx, result["actions"]))


def main():
    parser = argparse.ArgumentParser(description="Inference script for AgileX follower robot")
    parser.add_argument("--port", type=str, required=True, help="port name")
    parser.add_argument("--checkpoint_dir", required=True, type=str, help="path to checkpoint directory")
    parser.add_argument("--fps", type=int, required=False, default=30, help="frames per second")
    parser.add_argument("--task", type=str, required=False, help="task prompt",
                        default="pick up the circular chip and place it on the yellow pot")
    parser.add_argument("--id", type=str, required=False, help="robot id", default="left")
    parser.add_argument("--cameras", type=str, required=False, help="camera config yaml", default=None)
    parser.add_argument("--max_relative_target", type=int, required=False, default=None)
    parser.add_argument("--use_degrees", action="store_true")
    args = parser.parse_args()

    logger_send_action = NumpyCSVLogger("logs/inference_sended_action_0819_2_cameras_a.csv", mode="w")
    logger_state_at_action = NumpyCSVLogger("logs/inference_state_at_action_0819_2_cameras_b.csv", mode="w")
    print_log = True

    # 解析摄像头配置
    if args.cameras is not None:
        raw = yaml.safe_load(args.cameras)
        cameras = {name: OpenCVCameraConfig(index_or_path=cfg['index_or_path'], width=640, height=480, fps=30) for
                   name, cfg in raw.items()}
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
    ctx = mp.get_context("spawn")  # "spawn" 更安全，尤其 CUDA
    in_q: mp.Queue = ctx.Queue(maxsize=4)  # 根据实时性调节 maxsize
    out_q: mp.Queue = ctx.Queue(maxsize=4)
    config = _config.get_config("pi0_agileX")
    checkpoint_dir = args.checkpoint_dir  # "/home/agx/jemodel/test/40000"
    logging.info(f"policy path: {checkpoint_dir}")

    # proc = ctx.Process(
    #     target=inference_worker,
    #     args=(in_q, out_q, config, checkpoint_dir)
    # )
    # proc.daemon = True
    # proc.start()

    robot.connect()
    i, sent_idx, recv_idx = 0, 0, 0
    kMaxTimeStamps = 60000000000

    # rows = []
    step = 1
    step_time = step / (args.fps)
    prompt = args.task
    tokenizer = PaligemmaTokenizer()
    tokenized, mask = tokenizer.tokenize(prompt)

    action_queue = collections.deque()  # 存储当前动作序列
    waiting_for_infer = False

    while i < kMaxTimeStamps:
        t0 = time.perf_counter()

        current_state = robot.get_joint_state()
        logger_state_at_action.log(current_state['state'])
        # 2.5 统计
        i += 1
        dt_s = time.perf_counter() - t0
        # print(f"loop {i} dt={dt_s:.3f} s")
        time.sleep(max(step_time - dt_s, 0))

    # ==== 3. 结束 ====
    in_q.put(None)  # 通知子进程退出
    # proc.join()
    robot.disconnect()
    i, sent_idx, recv_idx = 0, 0, 0
    kMaxTimeStamps = 6000


if __name__ == "__main__":
    main()