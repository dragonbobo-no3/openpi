from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config

def main():
    # 选择配置和 checkpoint
    config = _config.get_config("pi0_fast_droid")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")

    # 创建已训练的 policy
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    # 构造 dummy 输入并推理
    example = droid_policy.make_droid_example()
    result = policy.infer(example)

    print("Actions shape:", result["actions"].shape)
    print("Actions:", result["actions"])

    # 释放内存
    del policy

if __name__ == "__main__":
    main()