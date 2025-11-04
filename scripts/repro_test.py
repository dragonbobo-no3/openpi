#!/usr/bin/env python3
"""Quick reproducibility test.

Runs multiple inference calls with identical inputs in the same process and
prints per-dimension differences. Use this to check intra-process determinism.

Usage examples:
python3 scripts/repro_test.py --config pi05_agileX --checkpoint_dir /path/to/ckpt --repo_id lerobot/test --episode_id 85 --step_in_episode 0 --repeats 5

Set --cpu to force CPU run (export CUDA_VISIBLE_DEVICES='' also works).
"""

import argparse
import os
import sys
import time
import random
import numpy as np

import torch
import jax

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def _select_episode_indices(dataset, episode_id: int):
    ds = dataset.hf_dataset
    ep_col = np.asarray(ds["episode_index"])
    idxs = np.nonzero(ep_col == episode_id)[0].tolist()
    if not idxs:
        raise ValueError(f"Episode {episode_id} not found.")
    return idxs


def build_obs_from_step(step):
    cur_state = np.asarray(step["observation.state"])
    obs = {
        "images": {
            "camera0": step["observation.images.camera0"],
            "camera1": step["observation.images.camera1"],
            "camera2": step["observation.images.camera2"],
            "camera3": step["observation.images.camera3"],
        },
        "image_masks": {
            "camera0": np.array([True], dtype=bool),
            "camera1": np.array([True], dtype=bool),
            "camera2": np.array([True], dtype=bool),
            "camera3": np.array([True], dtype=bool),
        },
        "state": cur_state,
        # tokenized_prompt not required for this quick test; policy should handle missing prompt
    }
    return obs


def print_env_info(seed):
    print("===== ENV INFO =====")
    print("PID:", os.getpid())
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("Current CUDA device:", torch.cuda.current_device())
    print("torch.version:", torch.__version__)
    try:
        import jax
        print("jax.version:", jax.__version__)
    except Exception:
        pass
    print("numpy.version:", np.__version__)
    print("seed:", seed)
    print("PYTHONHASHSEED:", os.environ.get("PYTHONHASHSEED"))
    print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
    print("MKL_NUM_THREADS:", os.environ.get("MKL_NUM_THREADS"))
    print("XLA_FLAGS:", os.environ.get("XLA_FLAGS"))
    print("====================")


def set_seeds(seed):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    # JAX PRNG is handled when creating keys in the code that uses it.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi05_agileX")
    parser.add_argument("--checkpoint_dir", default="/home/test/jemotor/jemodel/pi05/1017_pi05_test/60000/")
    parser.add_argument("--repo_id", default="lerobot/test")
    parser.add_argument("--root", default="/home/test/jemotor/jedata/test_0928_100_v2/")
    parser.add_argument("--episode_id", type=int, default=85)
    parser.add_argument("--step_in_episode", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cpu", action="store_true", help="Force CPU run via CUDA_VISIBLE_DEVICES=''")
    parser.add_argument("--deterministic", action="store_true", help="Try to set deterministic flags (torch)")
    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    seed = 12345
    # set_seeds(seed)

    if args.deterministic:
        try:
            import torch
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            print("Set PyTorch deterministic flags")
        except Exception as e:
            print("Could not set torch deterministic flags:", e)

    print_env_info(seed)

    cfg = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(cfg, args.checkpoint_dir)
    if hasattr(policy, "reset"):
        policy.reset()

    dataset = lerobot_dataset.LeRobotDataset(args.repo_id, root=args.root)
    indices = _select_episode_indices(dataset, args.episode_id)
    if args.step_in_episode < 0 or args.step_in_episode >= len(indices):
        raise ValueError("step_in_episode out of range")
    idx = indices[args.step_in_episode]
    step = dataset[idx]

    obs = build_obs_from_step(step)

    # Warm-up run (JAX will compile on first call) -- do not record this timing
    try:
        if hasattr(policy, "_rng"):
            # reset RNG to a known key before warmup so behavior is reproducible
            policy._rng = jax.random.PRNGKey(seed)
    except Exception:
        pass
    _ = policy.infer(obs)

    # Run multiple repeats (reset RNG before each repeat to force identical sampling)
    outputs = []
    for i in range(args.repeats):
        try:
            if hasattr(policy, "_rng"):
                policy._rng = jax.random.PRNGKey(seed)
        except Exception:
            pass
        t0 = time.time()
        out = policy.infer(obs)
        t1 = time.time()
        print(f"Repeat {i}: infer time {1000*(t1-t0):.2f} ms")
        # convert to numpy arrays for comparison (keep nested structures)
        out_np = out
        outputs.append(out_np)

    # Compare outputs to the first run
    ref = outputs[0]

    # Helper: flatten nested dict/list structures into a 1D numeric array for comparison
    def flatten_to_1d(x):
        if x is None:
            return np.array([], dtype=np.float64)
        if isinstance(x, (np.ndarray, jax.Array)):
            a = np.asarray(x)
            return a.ravel().astype(np.float64)
        if isinstance(x, (float, int, bool)):
            return np.asarray([x], dtype=np.float64)
        if isinstance(x, dict):
            parts = []
            for k in sorted(x.keys()):
                parts.append(flatten_to_1d(x[k]))
            return np.concatenate(parts) if parts else np.array([], dtype=np.float64)
        if isinstance(x, (list, tuple)):
            parts = [flatten_to_1d(v) for v in x]
            return np.concatenate(parts) if parts else np.array([], dtype=np.float64)
        # Fallback: try converting to numpy
        try:
            a = np.asarray(x)
            return a.ravel().astype(np.float64)
        except Exception:
            return np.array([], dtype=np.float64)
    print("\n===== DIFFS VS FIRST RUN =====")
    for i, out in enumerate(outputs[1:], start=1):
        print(f"-- Repeat {i} vs 0 --")
        for k in ref.keys():
            if k not in out:
                print(f"  key {k} missing in repeat {i}")
                continue
            a_flat = flatten_to_1d(ref[k])
            b_flat = flatten_to_1d(out[k])
            if a_flat.size != b_flat.size:
                print(f"  key {k} flattened size mismatch: {a_flat.size} vs {b_flat.size}")
                continue
            diff = a_flat - b_flat
            print(f"  {k}: max_abs={np.max(np.abs(diff)) if diff.size>0 else 0.0}, mean_abs={np.mean(np.abs(diff)) if diff.size>0 else 0.0}, rms={np.sqrt(np.mean(diff**2)) if diff.size>0 else 0.0}")
    print("==============================")


if __name__ == '__main__':
    main()
