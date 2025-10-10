from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import logging
import numpy as np
import tree
import threading
import time

import einops
from termcolor import colored
from typing_extensions import Literal, override

from openpi_client import base_policy as _base_policy
from openpi_client import image_tools

import jax
import jax.numpy as jnp

MICROSEC = 1_000_000


# def get_soft_guidance_mask(d, s, H, schedule: str = "exp"):
#     # 关键：H 必须是 Python int（编译时常量）
#     H_int = int(H)
#
#     # d、s 可能来自张量，这里压成 0-D JAX 标量
#     d = jnp.asarray(d, dtype=jnp.int32).reshape(())
#     s = jnp.asarray(s, dtype=jnp.int32).reshape(())
#
#     # ===== 打印输入参数 =====
#     # jax.debug.print(
#     #     "[mask] inputs -> d(raw)={}, s(raw)={}, H(py)={}, schedule={}",
#     #     d, s, H_int, schedule, ordered=True
#     # )
#     # ===== 打印输入参数 =====
#
#     # 用 Python int 做上界，不会引发 tracer->bool 转换
#     d = jnp.clip(d, 0, H_int)
#
#     # 分母保护，仍然用 H_int 参与运算
#     denom = jnp.maximum(H_int - s - d + 1, 1)
#
#     # 进一步打印标准化后的关键中间量
#     # jax.debug.print(
#     #     "[mask] normalized -> d(clip)={}, s={}, denom={}",
#     #     d, s, denom, ordered=True
#     # )
#
#     # 这里用常量 H_int 构造 arange，OK
#     i = jnp.arange(H_int, dtype=jnp.int32)
#
#     # 后续计算
#     ci = (H_int - s - i).astype(jnp.float32) / denom.astype(jnp.float32)
#
#     if schedule == "exp":
#         mid = ci * (jnp.exp(ci)- 1.0) / (jnp.e - 1.0)
#     elif schedule == "linear":
#         mid = ci
#     else:
#         raise ValueError(f"Unknown schedule: {schedule}")
#
#     mid = jnp.clip(mid, 0.0, 1.0)
#
#     W = jnp.where(
#         i < d, 1.0,
#         jnp.where(i >= (H_int - s), 0.0, mid)
#     ).astype(jnp.float32)
#
#     # ---- 打印整个 W（分块）----
#     # 说明：直接 print(W) 可能会被省略为 "..."；这里按块完整输出。
#     # chunk = 256  # 每块打印 256 个元素
#     # n_chunks = (H_int + chunk - 1) // chunk  # Python 端计算，避免动态 shape
#     #
#     # def _print_chunk(k, _):
#     #     start = k * chunk
#     #     end = jnp.minimum(start + chunk, H_int)
#     #     # 注意：format 的 start/end 用 host 值；切片在设备上完成
#     #     jax.debug.print("[mask] W[{start}:{end}] = {}", W[start:end],
#     #                     start=start, end=end, ordered=True)
#     #     return _
#     #
#     # # 小数组直接一次性打印；大数组分块打印
#     # if H_int <= chunk:
#     #     jax.debug.print("[mask] W (len={}) = {}", H_int, W, ordered=True)
#     # else:
#     #     jax.lax.fori_loop(0, n_chunks, _print_chunk, None)
#     # ---- 打印整个 W（分块）----
#
#     return W

def get_soft_guidance_mask(
    start: int, end: int, total: int, schedule: Literal["linear", "exp", "ones", "zeros"]
) -> jax.Array:
    """With start=2, end=6, total=10, the output will be:
    1  1  4/5 3/5 2/5 1/5 0  0  0  0
           ^              ^
         start           end
    `start` (inclusive) is where the chunk starts being allowed to change. `end` (exclusive) is where the chunk stops
    paying attention to the prefix. if start == 0, then the entire chunk is allowed to change. if end == total, then the
    entire prefix is attended to.

    `end` takes precedence over `start` in the sense that, if `end < start`, then `start` is pushed down to `end`. Thus,
    if `end` is 0, then the entire prefix will always be ignored.
    """
    start = jnp.minimum(start, end)
    if schedule == "ones":
        w = jnp.ones(total)
    elif schedule == "zeros":
        w = jnp.arange(total) < start
    elif schedule in ["linear", "exp"]:
        w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * jnp.expm1(w) / (jnp.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return jnp.where(jnp.arange(total) >= end, 0, w)

# 小工具：在 numpy 下模仿 jnp.at[].set
def _np_set(arr, mask, values):
    arr = arr.copy()
    arr[mask] = values
    return arr


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(
            self,
            policy: _base_policy.BasePolicy,
            action_horizon: int,
            do_preprocess_images: bool = False,  # whether to resize and permute image tensor
    ):
        self._policy = policy

        self._action_horizon = action_horizon
        self._cur_step: int = 0
        self._do_preprocess_images = do_preprocess_images
        self._last_results: dict[str, np.ndarray] | None = None

    @override
    def infer(self, obs: dict) -> dict:  # noqa: UP006
        if self._last_results is None:
            if self._do_preprocess_images:
                obs = dict(obs)  # shallow copy
                for cam_name in obs["images"]:
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(obs["images"][cam_name], 224, 224))
                    obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        results = tree.map_structure(
            lambda x: x[self._cur_step, ...] if isinstance(x, np.ndarray) else x, self._last_results
        )
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0


class NewActionChunkBroker(_base_policy.BasePolicy):
    """
    This wrapper:
    - Calls the underlying policy to get an action chunk. We'll call this "inference".
    - Manages actions in a lookup table according to their timestamps. When the `infer` method is called, the timestamp
      of the call is used to retrieve an action from the lookup table.
    - Can run inference synchronously (wait while inference is running, don't send new commands to the robot).
    - Or, can run inference asynchronously (keep trying to execute actions on demand based on what's available from
      prior chunks).
        - In this mode, we also send the prior action sequence to the policy for inpainting.

    See https://www.pi.website/download/real_time_chunking.pdf for more details on inpainting.

    When running in synchronous mode, the logic looks like:

    1. Run inference synchronously to get a sequence of actions. Return the first action.
    2. Keep returning actions from the previously generated sequence.
    3. When the sequence is depleted, go back to 1.

    When running in asynchronous mode, the logic looks like:

    1. Run inference synchronously to get a sequence of actions. Return the first action.
    2. Keep returning actions from the previously generated sequence.
    3. When the sequence has <= n_action_buffer actions in it, run asynchronous inference. In the meantime go
       back to 2.

    The implementation details are more involved and are documented inline. The main point though is that we
    can set n_action_buffer to be just large enough such that before we run out of actions, we already have a
    new sequence ready.

    A more feature complete version can be found here:
    https://github.com/huggingface/lerobot/blob/d10a493c6f137d09b86cbb681cb9fce188371df5/lerobot/common/policies/rollout_wrapper.py
    """

    def __init__(
            self,
            policy: _base_policy.BasePolicy,
            n_action_buffer: int,
            dt: float,
            do_preprocess_images: bool = False,
            mode: Literal["async", "sync"] = "async",
            inference_delay_cycles: int | None = None,
            X_RE: np.ndarray | None = None,
            Slist: np.ndarray | None = None,
    ):
        """
        Args:
            policy: The policy to wrap.
            n_action_buffer: As soon as the action buffer has <= n_action_buffer actions left, start an
                inference run.
            dt: The control loop period.
            do_preprocess_images: Whether to resize and permute image tensor.
            mode: Choose from "async"(hronous) or "sync"(hronous) mode (as described in the class docstring).
            inference_delay_cycles: Must be provided for "async" mode. This is an estimate for how many control loop
                periods inference runs for. It's used to generate the appropriate inpainting mask.
            X_RE: Home pose of end-effector in robot base frame. Used for plotting waypoints in Foxglove.
            Slist: The joint screw axes in the space frame when the end effector is at the home position, in the format
                   of a matrix with axes as the columns. Used for plotting waypoints in Foxglove.

        A more feature-complete version may be found here
        https://github.com/huggingface/lerobot/blob/d10a493c6f137d09b86cbb681cb9fce188371df5/lerobot/common/policies/rollout_wrapper.py
        """
        self._policy = policy
        self._dt_us = int(round(MICROSEC * dt))
        # We'll allow (almost) a full clock cycle of tolerance on timestamp retrieval.
        self._timestamp_tolerance_us = int(round(MICROSEC * dt)) - 1
        self._n_action_buffer = n_action_buffer

        # Set up async related logic.
        self._threadpool_executor = ThreadPoolExecutor(max_workers=1)
        self._thread_lock = threading.Lock()

        self._do_preprocess_images = do_preprocess_images
        self._mode = mode
        if mode == "async" and inference_delay_cycles is None:
            raise ValueError("inference_delay_cycles should be set in async mode")
        self._inference_delay_cycles = inference_delay_cycles

        # Some policies need to be called a few times with representative inputs for JIT compilation or other forms of
        # runtime optimization.
        self._is_warmed_up = False

        # self.X_RE = X_RE
        # if X_RE is not None:
        #     if Slist is None:
        #         raise ValueError("`X_RE` and `Slist` should be provided together")
        #     # Normalize screw axes
        #     self.Slist = Slist
        #     self.left_waypoint_publisher = WaypointPublisher("/waypoints_left")
        #     self.right_waypoint_publisher = WaypointPublisher("/waypoints_right")
        #     self._waypoint_publisher_pool = ThreadPoolExecutor(max_workers=1)

        self.reset()

    def __del__(self):
        """TODO(alexander-cobot): This isn't really working. Runtime exits with an exception."""
        self._threadpool_executor.shutdown(wait=True, cancel_futures=True)

    def reset(self):
        """Reset policy, observation cache, and action cache."""
        # Some policies need to be reset in between rollouts.
        self._policy.reset()
        with self._thread_lock:
            # Store a mapping from action timestamp (the moment the policy intends for the action to be)
            # executed.
            self._action_cache: dict[int, np.ndarray] = {}
        self._timeout = None
        # Fake timestamp counter used in sync mode
        self._fake_timestamp_us = 0

    def _get_contiguous_action_sequence_from_cache(
            self, first_action_timestamp_us: float
    ) -> dict[str, np.ndarray] | None:
        """
        Get the longest available contiguous action sequence from the cache, starting from the requested timestamp.
        "contiguous" means actions are separated by about dt.
        """
        with self._thread_lock:
            action_cache = deepcopy(self._action_cache)
        action_cache_timestamps_us = np.array(sorted(action_cache))
        if len(action_cache) == 0 or action_cache_timestamps_us.max() < first_action_timestamp_us:
            return None
        # We want to retrieve a series of actions spaced by dt.
        action_timestamps_us = np.arange(
            first_action_timestamp_us, action_cache_timestamps_us.max() + self._dt_us, self._dt_us
        )
        # Distance matrix where rows index the desired action timestamps, and columns index the cached action
        # timestamps.
        dist = np.abs(action_timestamps_us[:, None] - action_cache_timestamps_us[None])
        # For each desired action timestamp, what is the closest cached timestamp, and how close is it?
        argmin_ = dist.argmin(axis=1)
        min_ = dist[np.arange(len(argmin_)), argmin_]
        # Ignore timestamps that are out of tolerance.
        where_outside_tolerance = np.where(min_ > self._timestamp_tolerance_us)[0]
        if len(where_outside_tolerance) > 0:
            if where_outside_tolerance[0] == 0:
                return None  # couldn't even get the first timestamp
            argmin_ = argmin_[: where_outside_tolerance[0] + 1]
        selected_action_cache_timestamps_us = action_cache_timestamps_us[argmin_]
        # Deduplicate (there are some edge cases where the timestamps of the endpoints can be repeated).
        ts_diff = np.diff(selected_action_cache_timestamps_us, n=1)
        mask = np.append(True, ts_diff != 0)
        selected_action_cache_timestamps_us = selected_action_cache_timestamps_us[mask]
        assert np.all(np.diff(selected_action_cache_timestamps_us, n=1) > 0), "Retrieved duplicate action timestamps"
        # Retrieve and stack the actions.
        action_sequence = np.stack([action_cache[ts] for ts in selected_action_cache_timestamps_us], axis=0)
        return {"actions": action_sequence, "action_timestamps_us": selected_action_cache_timestamps_us}

    def _compute_and_publish_waypoints(self, actions: np.ndarray):
        """
        actions is (chunk_size, action_dim)
        """
        # Position of left waypoints (l) wrt left arm's base frame's (L) origin (o).
        ls_p_Lol = []
        # Position of right waypoints (r) wrt left arm's base frame's (L) origin (o).
        ls_p_Ror = []
        # Linearly sample 10 points from the chunk, making sure to keep the endpoint in place.
        for action in actions[-1:: -self._policy_chunk_size // 10]:
            state_len = int(len(action) / 2)
            left_action = action[:state_len]
            right_action = action[state_len:]
            for waypoints, joints in zip([ls_p_Lol, ls_p_Ror], [left_action, right_action], strict=True):
                waypoints.append(
                    mr.FKinSpace(
                        self.X_RE,
                        self.Slist,
                        joints[:6],
                    )[:3, -1]
                )
        self.left_waypoint_publisher.publish(np.array(list(reversed(ls_p_Lol))), frame_id="puppet_left/base_link")
        self.right_waypoint_publisher.publish(np.array(list(reversed(ls_p_Ror))), frame_id="puppet_right/base_link")

    def run_inference(self, obs: dict, action_timestamp_us: int):
        # print(f"start inference: {self._mode}")
        """Call the policy server using the latest observation."""
        start_inference_t = time.monotonic()

        # Preprocess images.
        if self._do_preprocess_images:
            obs = dict(obs)  # shallow copy
            for cam_name in obs["images"]:
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(obs["images"][cam_name], 224, 224))
                obs["images"] = dict(obs["images"])  # shallow copy
                obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")

        # Set parameters for inpainting.
        prior = self._get_contiguous_action_sequence_from_cache(action_timestamp_us)
        if prior is not None and self._mode == "async":
            # print(f"here: {self._mode}")
            assert np.abs(prior["action_timestamps_us"][0] - action_timestamp_us) < self._dt_us
            # Note: Here we provide inpainting parameters via the "observation". This is part of a hack for plumbing
            # these parameters back to the model.
            npad = self._policy_chunk_size - prior["actions"].shape[0]
            # Provide the policy with the prior actions. It will use these for inpainting. We pad with zeros to get
            # the right chunk size, but these shouldn't be attended to.
            obs["actions"] = np.pad(prior["actions"], ((0, npad), (0, 0)))
            print(obs["actions"])
            # Inpainting gives full attention to the first `inference_delay_cycles` actions of the prior action chunk.
            obs["inference_delay"] = self._inference_delay_cycles
            # Inpainting gives soft attention to all actions of the prior action chunk thereafter.
            # NOTE: really, this can be set to any number in [inference_delay_cycles, prior_actions_length]. We set it
            # to the maximum here, but we can make it into a configurable setting. Making it smaller trades off
            # inter-chunk consistency in favor of reactivity to new observations.
            obs["prior_attention_horizon"] = prior["actions"].shape[0]

        # Run inference.
        # t0 = time.perf_counter()
        results = self._policy.infer(obs)
        # t1 = time.perf_counter()
        # print(f"Inference took {t1 - t0} seconds")

        # Cache chunk size for later use in the inpainting logic (assuming it doesn't change dynamically).
        if not hasattr(self, "_policy_chunk_size"):
            self._policy_chunk_size = results["actions"].shape[0]
        elif results["actions"].shape[0] != self._policy_chunk_size:
            raise ValueError(f"Chunk size has changed from {self._policy_chunk_size} to {results['actions'].shape[0]}")

        # if self.X_RE is not None:
        #     self._waypoint_publisher_pool.submit(self._compute_and_publish_waypoints, results["actions"].copy())

        # Update action cache. We need to "deduplicate" some actions in the cache. That is, we need to remove prior
        # actions that align with new actions.
        # By now, some time has passed since action_timestamp_us.
        switchover_timestamp_us = self._get_timestamp_us()
        prior = self._get_contiguous_action_sequence_from_cache(switchover_timestamp_us)
        new_actions = results["actions"].copy()
        if self._mode == "async" and prior is not None:
            new_action_timestamps_us = np.array(
                [action_timestamp_us + i * self._dt_us for i in range(len(new_actions))]
            )
            prior_actions = prior["actions"]
            prior_action_timestamps_us = prior["action_timestamps_us"]
            # Align the two action sequences wherever the first action of the prior action sequence matches the new
            # action sequence.
            align_at = np.argmin(np.abs(new_action_timestamps_us - prior_action_timestamps_us[0]))
            print(new_action_timestamps_us)
            print(prior_action_timestamps_us)
            print(align_at)
            print(self._action_cache)
            print(new_actions)
            with self._thread_lock:
                # Overwrite prior action sequence.
                for prior_action_timestamp_us, new_action_timestamp_us, new_action in zip(
                        prior_action_timestamps_us,
                        new_action_timestamps_us[align_at: align_at + len(prior_actions)],
                        new_actions[align_at: align_at + len(prior_actions)],
                        strict=True,
                ):
                    del self._action_cache[prior_action_timestamp_us]
                    self._action_cache[new_action_timestamp_us] = new_action
                # Also add on the end of the new action sequence (for which there were no prior actions to overwrite).
                for new_action_timstamp_us, new_action in zip(
                        new_action_timestamps_us[align_at + len(prior_actions):],
                        new_actions[align_at + len(prior_actions):],
                        strict=True,
                ):
                    self._action_cache[new_action_timstamp_us] = new_action
            print(self._action_cache)
        else:
            with self._thread_lock:
                self._action_cache.update(
                    {action_timestamp_us + i * self._dt_us: action for i, action in enumerate(new_actions)}
                )

        inference_time = time.monotonic() - start_inference_t
        logging.info(colored(f"Inference time: {inference_time * 1000:.0f} ms", "yellow"))
        if self._mode == "async" and inference_time > (self._n_action_buffer * self._dt_us + self._dt_us) / MICROSEC:
            logging.warning(
                "Inference is taking longer than your buffer.\n"
                f"  Buffer time   : {self._n_action_buffer * self._dt_us + self._dt_us / 1000=} ms\n"
                f"  Inference time: {inference_time * 1000:.0f} ms"
            )
        # print("end inference")

    def _get_timestamp_us(self):
        """Get the current timestamp in microseconds.

        This is the timestamp used to align actions in the cache.

        In async mode this is the real system timestamp.
        In sync mode this is a fake timestamp that ticks once every time infer is called.
        """
        if self._mode == "async":
            return int(round(time.monotonic() * MICROSEC))
        elif self._mode == "sync":
            return self._fake_timestamp_us
        else:
            raise AssertionError

    @staticmethod
    def _increment_fake_timestamp(func):
        def wrapped(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self._fake_timestamp_us += self._dt_us
            return result

        return wrapped

    @_increment_fake_timestamp
    @override
    def infer(self, obs: dict) -> np.ndarray | None:
        # print("start infer")
        if not self._is_warmed_up:
            # Warm up causes JIT compilation on the policy side.
            logging.info("Warming up...")
            for _ in range(5):
                start = time.monotonic()
                first_action_timestamp_us = self._get_timestamp_us()
                self.run_inference(obs, first_action_timestamp_us)
                elapsed = time.monotonic() - start
                if (sleep_for := self._dt_us / MICROSEC - elapsed) > 0:
                    time.sleep(sleep_for)

            self.reset()
            self._is_warmed_up = True
            logging.info("Done warming up...")

        start = time.monotonic()
        # This timeout will be used to wait for inference threads. It defaults to None meaning we wait till the thread
        # returns a result.
        timeout = self._timeout
        if timeout is None and self._mode == "async":
            # In async mode, we set the number to something very small (enough time for code to run, but not so much
            # that it slows down the control loop significantly).
            # NOTE: For the first inference step we leave the timeout as None (here we are just changing self._timeout
            # which will be used next time this method is called).
            self._timeout = 1e-3

        # This is effectively the requested timestamp for inference.
        first_action_timestamp_us = self._get_timestamp_us()

        # placeholder for the return value of `_get_contiguous_action_sequence_from_cache`
        action_cache_results: dict[str, np.ndarray] | None = None

        # Try retrieving an action sequence from the cache starting from `first_action_timestamp` and spaced
        # by `dt`. While doing so remove stale actions (those which are older and outside tolerance).
        with self._thread_lock:
            action_cache_timestamps_us = np.array(sorted(self._action_cache))
        if len(action_cache_timestamps_us) > 0:
            diff = action_cache_timestamps_us - first_action_timestamp_us
            to_delete = np.where(np.bitwise_and(diff < 0, np.abs(diff) > self._timestamp_tolerance_us * 2))[0]
            for ix in to_delete:
                with self._thread_lock:
                    del self._action_cache[action_cache_timestamps_us[ix.item()].item()]
            # If the first action is in the cache, construct the action sequence.
            if np.argmin(np.abs(diff)) <= self._timestamp_tolerance_us:
                action_cache_results = self._get_contiguous_action_sequence_from_cache(first_action_timestamp_us)

        # We would like to run inference if we don't have many actions left in the cache.
        want_to_run_inference = action_cache_results is None or (
                action_cache_results is not None and action_cache_results["actions"].shape[0] <= self._n_action_buffer
        )
        logging.info(
            colored(
                f"ts: {first_action_timestamp_us / 1000:.0f} ms, cache size: "
                f"{action_cache_results['actions'].shape[0] if action_cache_results is not None else 0}, "
                f"{want_to_run_inference=}",
                "yellow",
            )
        )
        # Return an action right away if we know we don't want to run inference.
        if not want_to_run_inference:
            # print(f"end infer 1: {action_cache_results['actions'].shape}")
            return {"actions": action_cache_results["actions"][0]}

        # If we couldn't get any actions from the cache, not only do we want to run inference, but we must.
        must_run_inference = action_cache_results is None

        # We can't run inference if a previous inference is already running.
        if hasattr(self, "_future") and self._future.running():
            # If we don't have an action. We have no choice but to wait and hope inference finishes on time.
            if must_run_inference:
                logging.info("Attempting to wait for previous inference to complete.")
                # If this fails, we have reached the end of the runway and have no choice but to raise a TimeoutError.
                self._future.result(timeout=timeout)
            else:
                # Nothing else to do. We have an action to return, and can't start a new inference.
                # print(f"end infer 2: {action_cache_results['actions'].shape}")
                return {"actions": action_cache_results["actions"][0]}

        # Start the inference job.
        self._future = self._threadpool_executor.submit(self.run_inference, obs, first_action_timestamp_us)

        # If we must, attempt to wait for inference to complete, within the bounds of the `timeout` parameter.
        # If this fails, we have reached the end of the runway and have no choice but to raise a TimeoutError.
        if must_run_inference:
            self._future.result(timeout=timeout)

        # If inference is complete, get the fresher actions from the cache.
        if not self._future.running():
            action_cache_results = self._get_contiguous_action_sequence_from_cache(first_action_timestamp_us)
        # print(f"end infer 3: {action_cache_results['actions'].shape}")
        return {"actions": action_cache_results["actions"][0]}
