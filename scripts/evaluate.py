import numpy as np
import matplotlib.pyplot as plt
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import time

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.models.tokenizer import PaligemmaTokenizer


def main():
    # 选择配置和 checkpoint
    config = _config.get_config("pi0_agileX")
    checkpoint_dir = "/home/kleist/Documents/Model/cloud_server/model/0808_openpi/14999"
    max_frame = 20000
    # default_prompt="pick up the circular chip and place it on the yellow pot"
    id = 4
    action_horizon = 50

    # 直接用 LeRobotDataset 读取 episode
    repo_id = "lerobot/test"
    root = "/home/kleist/Documents/Database/test_0807a_modified"
    dataset = lerobot_dataset.LeRobotDataset(repo_id, root=root)

    # 获取所有 step 的 episode_index
    episode_indices = [item["episode_index"].item() for item in dataset.hf_dataset]

    # 找到目标 episode 的所有 step 索引
    episode = [i for i, ep_idx in enumerate(episode_indices) if ep_idx == id]

    # 创建已训练的 policy
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    gt_actions_list = []
    pred_actions_list = []
    obs = config.model.fake_obs()
    tokenizer = PaligemmaTokenizer()
    bias = 0
    for t, idx in enumerate(episode):
        if t >= max_frame:
            break
        index = t
        step = dataset[idx]
        # print(step.keys())
        gt_action = step["action"]
        prompt = step["task"]
        tokenized, mask = tokenizer.tokenize(prompt)
        gt_actions_list.append(np.array(gt_action))
        if t < bias:
            pred_actions_list.append(np.array(gt_action))

        # 只在每个 action_horizon 的起点做一次推理
        if (index-bias) % action_horizon == 0:
            prompt = step["task"]
            tokenized, mask = tokenizer.tokenize(prompt)
            print(f"shape{step['observation.images.camera0'].shape}")
            obs = {
                "images": {
                    "camera0": step["observation.images.camera0"],
                    "camera1": step["observation.images.camera1"],
                    "camera2": step["observation.images.camera2"],
                    "camera3": step["observation.images.camera3"],
                },
                "image_masks": {
                    "camera0": np.array([True]),
                    "camera1": np.array([True]),
                    "camera2": np.array([True]),
                    "camera3": np.array([True]),
                },
                "state": step["observation.state"],
                "tokenized_prompt": tokenized[None],
                "tokenized_prompt_mask": mask[None],
                "token_ar_mask": None,
                "token_loss_mask": None,
            }
            start_time = time.time()
            result = policy.infer(obs)
            infer_time = time.time() - start_time
            print(f"Step {t}: infer time = {infer_time:.4f} seconds")
        
            pred_actions = result["actions"][:action_horizon]  # shape: (action_horizon, action_dim)
            # 存 action_horizon 步预测
            for i in range(pred_actions.shape[0]):
                pred_actions_list.append(np.array(pred_actions[i]))


        # 截断 pred_actions_list 以和 gt_actions_list 对齐（防止最后一段超出）
        # min_len = min(len(gt_actions_list), len(pred_actions_list))
        # gt_actions_arr = np.stack(gt_actions_list[:min_len])
        # pred_actions_arr = np.stack(pred_actions_list[:min_len])
    gt_actions_arr = np.stack(gt_actions_list)
    pred_actions_arr = np.stack(pred_actions_list)

    # 绘制所有动作分量的纵向排列图
    plt.figure(figsize=(12, 6))
    action_dim = gt_actions_arr.shape[1]
    fig, axes = plt.subplots(action_dim, 1, figsize=(10, 4 * action_dim), sharex=True)

    for i in range(action_dim):
        ax = axes[i]
        ax.plot(gt_actions_arr[:, i], label=f"GT action {i}", linestyle='--')
        ax.plot(pred_actions_arr[:, i], label=f"Pred action {i}")
        highlight_idx = np.arange(bias, len(pred_actions_arr), action_horizon)
        ax.scatter(highlight_idx, pred_actions_arr[highlight_idx, i], color='red', label='First pred in action_horizon', zorder=5)
        ax.set_ylabel(f"Action dim {i}")
        ax.legend()
        ax.set_title(f"GT vs Predicted Actions (dim {i})")

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(f"./id{id}bias{bias}action_horizon{action_horizon}_action_compare_all.png")
    plt.close(fig)

    # 释放内存
    del policy

if __name__ == "__main__":
    main()