"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
uv run scripts/compute_norm_stats.py --config-name pi05_agileX
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms

#删掉字典中值为字符串的键值对
class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}

class KeepOnly(transforms.DataTransformFn):
    def __init__(self, keys): self.keys = set(keys)
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if k in self.keys}

def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    # dataset = _data_loader.TransformedDataset(
    #     dataset,
    #     [
    #         *data_config.repack_transforms.inputs, # Group(inputs=[RepackTransform(structure={'images': {'camera0': 'observation.images.camera0', 'camera1': 'observation.images.camera1', 'camera2': 'observation.images.camera2', 'camera3': 'observation.images.camera3'}, 'state': 'observation.state', 'actions': 'action'})], outputs=())
    #         *data_config.data_transforms.inputs, # Group(inputs=(AgileXInputs(action_dim=7, adapt_to_pi=True), DeltaActions(mask=(True, True, True, True, True, True, False))), outputs=(AbsoluteActions(mask=(True, True, True, True, True, True, False)), AgileXOutputs(adapt_to_pi=True)))
    #         # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
    #         RemoveStrings(),
    #     ],
    # )
    # print(data_config.repack_transforms)
    # print(data_config.data_transforms)
    # exit(1)
    stats_repack = transforms.Group(inputs=[
        transforms.RepackTransform({
            "state": "observation.state",
            "actions": "action",
            # 注意：不要再出现任何 images/camera 的映射
            "effort": "effort",
        })
    ])

    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *stats_repack.inputs, # Group(inputs=[RepackTransform(structure={'images': {'camera0': 'observation.images.camera0', 'camera1': 'observation.images.camera1', 'camera2': 'observation.images.camera2', 'camera3': 'observation.images.camera3'}, 'state': 'observation.state', 'actions': 'action'})], outputs=())
            *data_config.data_transforms.inputs, # Group(inputs=(AgileXInputs(action_dim=7, adapt_to_pi=True), DeltaActions(mask=(True, True, True, True, True, True, False))), outputs=(AbsoluteActions(mask=(True, True, True, True, True, True, False)), AgileXOutputs(adapt_to_pi=True)))
            RemoveStrings(),
        ],
    )

    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model, False)
    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    if data_config.use_effort:
        keys.append("effort")
    stats = {key: normalize.RunningStats() for key in keys}
    # for b in data_loader:
    #     print(b.keys())
    
    # print("here")
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        # print(batch.keys())
        # exit(1)
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
