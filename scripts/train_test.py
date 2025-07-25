import dataclasses
import os
import pathlib

import pytest

# os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

import train


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    assets = _config.AssetsConfig(asset_id="test", assets_dir="/home/agx/jedata/test_0711a")
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=32,
        checkpoint_base_dir=str(tmp_path + "/checkpoint"),
        exp_name="test",
        overwrite=True,
        resume=False,
        num_train_steps=1000,
        log_interval=2,
        assets_base_dir="/home/agx/jedata/test_0711a",
        wandb_enabled=False,
    )
    # config = dataclasses.replace(config, assets=assets)
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)

if __name__ == "__main__":
    test_train("/home/agx/jemodel/test/", "pi0_aloha")

