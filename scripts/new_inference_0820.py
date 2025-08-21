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

from openpi.serving.real_time_chunk import NewActionChunkBroker
from scripts.numpy_logger import NumpyCSVLogger

from openpi.policies import policy_config as _policy_config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training import config as _config
from third_party.agilex.agilexfollower import AlohaAgileXFollower
from third_party.agilex.agilexconfig import AlohaAgileXFollowerConfig
from third_party.cameras.opencv.configuration_opencv import OpenCVCameraConfig


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

    logger_send_action = NumpyCSVLogger("logs/inference_sended_action_0820a.csv", mode="w")
    logger_state_at_action = NumpyCSVLogger("logs/inference_state_at_action_0820a.csv", mode="w")
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
    robot.connect()

    # Load pretrained policy
    # ==== 1. 启动推理子进程 ====
    config = _config.get_config("pi0_agileX")
    checkpoint_dir = args.checkpoint_dir  # "/home/agx/jemodel/test/40000"
    logging.info(f"policy path: {checkpoint_dir}")
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    # rows = []
    step = 1
    step_time = step / (args.fps)
    prompt = args.task
    tokenizer = PaligemmaTokenizer()
    tokenized, mask = tokenizer.tokenize(prompt)

    action_queue = collections.deque()  # 存储当前动作序列
    waiting_for_infer = False

    kMaxTimeStamps = 6000000000
    n_action_buffer = 25
    inference_delay_cycles = 5

    broker = NewActionChunkBroker(
        policy=policy,
        n_action_buffer=n_action_buffer,
        dt=step_time,
        mode="async",
        inference_delay_cycles=inference_delay_cycles,
        do_preprocess_images=False,  # 你的obs里图像若是HWC且已就绪，就关掉
    )

    index = 0
    while index < kMaxTimeStamps:
        t0 = time.perf_counter()
        obs = robot.get_observation()
        obs["state"] = obs["state"]
        obs["tokenized_prompt"] = tokenized[None]
        obs["tokenized_prompt_mask"] = mask[None]
        obs["token_ar_mask"] = None
        obs["token_loss_mask"] = None
        # obs["inference_delay"] = None
        # obs["token_loss_mask"] = None

        out = broker.infer(obs)
        action = out["actions"]  # 注意：broker返回的是 {"actions": 单步动作}
        robot.send_action_np(action[:7])
        if print_log:
            logger_send_action.log(action[:7])
            current_state = robot.get_joint_state()
            logger_state_at_action.log(current_state['state'])
        # 2.5 统计
        index += 1
        dt_s = time.perf_counter() - t0
        # print(f"loop {index} dt={dt_s:.3f} s")
        time.sleep(max(step_time - dt_s, 0))

    robot.disconnect()


if __name__ == "__main__":
    main()
