import os
import sys
# 替换为你的项目实际路径
project_path = "/home/kleist/Documents/Code/openpi/openpi_modified/openpi/"
sys.path.append(project_path)

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
    checkpoint_dir = "/home/kleist/Documents/Model/pi0_openpi_0723/76500"
    default_prompt="pick up the circular chip and place it on the yellow pot"
    id = 30
    period = 10

    # 直接用 LeRobotDataset 读取 episode
    repo_id = "lerobot/test"
    root = "/home/kleist/Documents/Database/test_0711a_test"
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
    for t, idx in enumerate(episode):
        step = dataset[idx]
        # print(step.keys())
        gt_action = step["action"]
        prompt = step["task"]
        tokenized, mask = tokenizer.tokenize(prompt)
        gt_actions_list.append(np.array(gt_action))

        # 只在每个 period 的起点做一次推理
        if t % period == 0:
            prompt = step["task"]
            tokenized, mask = tokenizer.tokenize(prompt)
            print(f"shape{step['observation.images.camera0'].shape}")
            obs = {
                "images": {
                    "camera0": step["observation.images.camera0"],
                    "camera1": step["observation.images.camera1"],
                    "camera2": step["observation.images.camera2"],
                },
                "image_masks": {
                    "camera0": np.array([True]),
                    "camera1": np.array([True]),
                    "camera2": np.array([True]),
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
        
            pred_actions = result["actions"][:period]  # shape: (period, action_dim)
            # 存 period 步预测
            for i in range(pred_actions.shape[0]):
                pred_actions_list.append(np.array(pred_actions[i]))


        # 截断 pred_actions_list 以和 gt_actions_list 对齐（防止最后一段超出）
        min_len = min(len(gt_actions_list), len(pred_actions_list))
        gt_actions_arr = np.stack(gt_actions_list[:min_len])
        pred_actions_arr = np.stack(pred_actions_list[:min_len])

    action_dim = gt_actions_arr.shape[1]

    # ─── 可配置的字体设置 ───
    TITLE_FONT_SIZE = 16
    LABEL_FONT_SIZE = 12
    LEGEND_FONT_SIZE = 10

    font_title  = {"fontsize": TITLE_FONT_SIZE}
    font_label  = {"fontsize": LABEL_FONT_SIZE}
    font_legend = {"fontsize": LEGEND_FONT_SIZE}

    # ─── 在一个 Figure 中纵向排列子图 ───
    # ⬇️ 将宽度拉长到 12 英寸、高度缩小到每行 2 英寸
    fig, axes = plt.subplots(
        nrows=action_dim,
        ncols=1,
        figsize=(12, 2 * action_dim),
        sharex=True,
    )

    if action_dim == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(gt_actions_arr[:, i], linestyle="--", label=f"GT action {i}")
        ax.plot(pred_actions_arr[:, i], linestyle="-",  label=f"Pred action {i}")
        ax.set_ylabel(f"Joint value{i}", **font_label)
        # ax.set_title(f"Action Dimension {i}", **font_title)
        ax.legend(**font_legend)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step", **font_label)

    fig.suptitle(
        f"Episode {id}: GT vs Predicted Actions (period={period}) (0.001 deg)",
        **font_title,
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f"period{period}_episode{id}_all_dims_compare.png")
    plt.close(fig)


    # # 绘制每个动作分量的曲线
    # plt.figure(figsize=(12, 6))
    # action_dim = gt_actions_arr.shape[1]
    # for i in range(action_dim):
    #     plt.figure(figsize=(8, 4))
    #     plt.plot(gt_actions_arr[:, i], label=f"GT action {i}", linestyle='--')
    #     plt.plot(pred_actions_arr[:, i], label=f"Pred action {i}")
    #     plt.xlabel("Step")
    #     plt.ylabel(f"Action dim {i} value")
    #     plt.title(f"Episode {id}: GT vs Predicted Actions (dim {i})")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(f"period{period}_action_dim_{i}_compare.png")
    #     plt.close()

    # 释放内存
    del policy

if __name__ == "__main__":
    main()