import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_aloha_example() -> dict:
    """Creates a random input example for the Aloha policy."""
    return {
        "state": np.ones((7,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class AgileXInputs(transforms.DataTransformFn):
    """Inputs for the Aloha/AgileX policy.

    Expected inputs when use_images=True:
    - images: dict[name, img] where img is [C, H, W]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]  (optional during inference)
    """

    action_dim: int
    adapt_to_pi: bool = True
    # ← 新增：是否处理相机图像。False 时不访问 data["images"]，也不调用 _decode_aloha。
    use_images: bool = True

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("camera0", "camera1", "camera2", "camera3")

    def __call__(self, data: dict) -> dict:
        # 仅在需要图像时才调用 _decode_aloha（其内部会访问 data["images"]）
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi, use_images=self.use_images)

        # ---- state ----
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        # ---- images（可选）----
        images = {}
        image_masks = {}

        if self.use_images:
            in_images = data["images"]
            if set(in_images) - set(self.EXPECTED_CAMERAS):
                raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

            # Assume that base image always exists.
            base_image = in_images["camera0"]
            right_wrist_image = in_images["camera1"]
            feng_image = in_images["camera2"]
            bao_image = in_images["camera3"]

            images = {
                "base_rgb": base_image,
                "right_wrist_rgb": right_wrist_image,
                "feng_rgb": feng_image,
                "bao_rgb": bao_image,
            }
            image_masks = {
                "base_rgb": np.True_,
                "right_wrist_rgb": np.True_,
                "feng_rgb": np.True_,
                "bao_rgb": np.True_,
            }

            # Add the extra images.
            extra_image_names = {
            }

            # # 从这开始
            # images = {
            #     "right_wrist_rgb": right_wrist_image,
            #     "right_pole_rgb": right_pole_image,
            # }
            # image_masks = {
            #     "right_wrist_rgb": np.True_,
            #     "right_pole_rgb": np.True_,
            # }

            # # Add the extra images.
            # extra_image_names = {
            # }
            # # 到这结束

            for dest, source in extra_image_names.items():
                if source in in_images:
                    images[dest] = in_images[source]
                    image_masks[dest] = np.True_
                else:
                    images[dest] = np.zeros_like(base_image)
                    image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # ---- actions（训练阶段才有）----
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AgileXOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy."""

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # pi0 gripper data is normalized (0, 1) between encoder counts (2405, 3110).
    # There are 4096 total encoder counts and aloha uses a zero of 2048.
    # Converting this to radians means that the normalized inputs are between (0.5476, 1.6296)
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # We do not scale the output since the trossen model predictions are already in radians.
    # See the comment in _gripper_to_angular for a derivation of the constant
    value = value + 0.5476

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return value - 0.5476


def _decode_aloha(
    data: dict,
    *,
    adapt_to_pi: bool = False,
    use_images: bool = True,   # ← 新增开关
) -> dict:
    # --- state 始终解码 ---
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)
    data["state"] = state

    # --- 图像可选 ---
    if not use_images:
        # 不处理/不访问 data["images"]，保持原样或缺省
        return data

    images = data.get("images")
    if not isinstance(images, dict) or len(images) == 0:
        # 没有图像就直接返回（也可改成 raise KeyError("images")，看你需要的严格程度）
        return data

    def convert_image(img):
        arr = np.asarray(img)
        # 浮点图转 uint8
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        # 只在明显是 CHW 时才转成 HWC
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = einops.rearrange(arr, "c h w -> h w c")
        return arr

    images_dict = {name: convert_image(img) for name, img in images.items()}
    data["images"] = images_dict
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    # 支持 7 或 14 维输入
    if adapt_to_pi:
        # 只处理前7维
        state_main = state[:7]
        state_rest = state[7:] if state.shape[0] > 7 else None
        state_main = _joint_flip_mask() * state_main
        state_main[[6]] = _gripper_to_angular(state_main[[6]])
        if state_rest is not None:
            state = np.concatenate([state_main, state_rest], axis=-1)
        else:
            state = state_main
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        actions[:, [6]] = _gripper_from_angular(actions[:, [6]])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6]] = _gripper_from_angular_inv(actions[:, [6]])
    return actions
