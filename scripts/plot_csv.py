import numpy as np
import matplotlib.pyplot as plt
import time


def main():
    # default_prompt="pick up the circular chip and place it on the yellow pot"
    id = 4
    action_horizon = 50

    # 直接用 LeRobotDataset 读取 episode
    root = "/home/kleist/Documents/Code/openpi/logs/1a.csv"

    gt_actions_arr = np.genfromtxt(root, delimiter=",", dtype=float, filling_values=np.nan)

    # 绘制所有动作分量的纵向排列图
    plt.figure(figsize=(12, 6))
    action_dim = gt_actions_arr.shape[1]
    fig, axes = plt.subplots(action_dim, 1, figsize=(10, 4 * action_dim), sharex=True)

    for i in range(action_dim):
        ax = axes[i]
        ax.plot(gt_actions_arr[:int(0.25*len(gt_actions_arr)), i], label=f"GT action {i}")
        # highlight_idx = np.arange(0, len(gt_actions_arr), action_horizon)
        # ax.scatter(highlight_idx, gt_actions_arr[highlight_idx, i], color='red', label='First pred in action_horizon',
        #            zorder=5, s=6)
        ax.set_ylabel(f"Action dim {i}")
        ax.legend()
        ax.set_title(f"Predicted Actions (dim {i})")

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(f"./inference_id{id}_action_compare_all.png")
    plt.close(fig)


if __name__ == "__main__":
    main()