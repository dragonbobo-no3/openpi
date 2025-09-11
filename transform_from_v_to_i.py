import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import cv2
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-episode', type=int, default=0, help='Start converting from this episode index')
    args = parser.parse_args()

    default_prompt = "Pick up the PCB board on the round yellow base and place it into the circular recess of the yellow square container."
    video_repo_id = "lerobot/test"
    video_root = "/jedata/test_0807a_modified"
    image_repo_id = "lerobot/test"
    image_root = "/jedata/lerobot_test_image_out"
    # 构造 features 字典
    features = {
        "observation.state": {"dtype": "float32", "shape": (7,)},
        "action": {"dtype": "float32", "shape": (7,)},
        "observation.images.camera0": {"dtype": "image", "shape": (3, 480, 640)},
        "observation.images.camera1": {"dtype": "image", "shape": (3, 480, 640)},
        "observation.images.camera2": {"dtype": "image", "shape": (3, 480, 640)},
        "observation.images.camera3": {"dtype": "image", "shape": (3, 480, 640)},
    }
    N = 5  # 每N个episode保存并重建dataset对象
    episode_count = 0
    dataset = lerobot_dataset.LeRobotDataset.create(
        repo_id=image_repo_id,
        root=image_root,
        fps=30,
        robot_type="agilex",
        features=features,
    )
    # 加载视频数据集
    video_dataset = lerobot_dataset.LeRobotDataset(video_repo_id, root=video_root)
    episode_indices = [item["episode_index"].item() for item in video_dataset.hf_dataset]
    unique_episode_ids = sorted(set(episode_indices))
    # 只处理大于等于start_episode的episode
    for eid in unique_episode_ids:
        if eid < args.start_episode:
            continue
        episode = [i for i, ep_idx in enumerate(episode_indices) if ep_idx == eid]
        for t, idx in enumerate(episode):
            step = video_dataset[idx]
            frame = {
                "observation.state": step["observation.state"],
                "action": step["action"],
                "observation.images.camera0": None,
                "observation.images.camera1": None,
                "observation.images.camera2": None,
                "observation.images.camera3": None,
            }
            for cam_key in [
                "observation.images.camera0",
                "observation.images.camera1",
                "observation.images.camera2",
                "observation.images.camera3"
            ]:
                img = step[cam_key]
                if hasattr(img, 'numpy'):
                    img = img.numpy()
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                if img.shape[0] == 3 and img.ndim == 3:
                    img = img.transpose(1, 2, 0)
                frame[cam_key] = img
            frame["task"] = step.get("task", default_prompt)
            # frame["timestamp"] = step.get("timestamp", t / 30.0)
            dataset.add_frame(frame)
        dataset.save_episode()
        print(f"Saved episode {eid} to {image_root}")
        episode_count += 1

    del dataset

if __name__ == "__main__":
    main()