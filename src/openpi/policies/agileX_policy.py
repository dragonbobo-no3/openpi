import dataclasses
from typing import ClassVar

import einops
import numpy as np
from PIL import Image
import cv2

from openpi import transforms


def depth_rgb_u8_to_u16(depth_rgb_u8: np.ndarray, order: str = "HI_LO") -> np.ndarray:
    """
    把 (H,W,3) uint8（R/G存放高/低8位）还原为 (H,W) uint16 深度（单位通常为 mm）
    """
    assert depth_rgb_u8.dtype == np.uint8 and depth_rgb_u8.ndim == 3 and depth_rgb_u8.shape[2] >= 2
    r = depth_rgb_u8[..., 0].astype(np.uint16)
    g = depth_rgb_u8[..., 1].astype(np.uint16)
    if order.upper() == "HI_LO":
        depth_u16 = (r << 8) | g
    elif order.upper() == "LO_HI":
        depth_u16 = (g << 8) | r
    else:
        raise ValueError("order must be 'HI_LO' or 'LO_HI'")
    return depth_u16  # (H,W) uint16

def depth_u16_to_u8x3(
    depth_u16: np.ndarray,
    mode: str = "disparity",   # "disparity" 或 "log"
    zmin_m: float = 0.2,
    zmax_m: float = 4.0,
) -> np.ndarray:
    """
    depth_u16: (H, W) 的 uint16 深度图（单位 mm）
    返回: (H, W, 3) 的 uint8，可直接作为3通道图像使用（灰度复制到3通道）
    """
    assert depth_u16.dtype == np.uint16 and depth_u16.ndim == 2, "输入应为 (H,W) uint16"
    d = depth_u16.astype(np.float32)  # 毫米
    H, W = d.shape
    valid = d > 0

    # 将无效像素先临时设为最大深度，避免影响归一化；最后再置黑
    zmin_mm = int(zmin_m * 1000)
    zmax_mm = int(zmax_m * 1000)
    tmp = d.copy()
    tmp[~valid] = zmax_mm
    tmp = np.clip(tmp, zmin_mm, zmax_mm)

    if mode.lower() == "disparity":
        # 视差映射：近处值更大 → 归一化后更亮
        x = 1.0 / tmp
        xmin, xmax = 1.0 / zmax_mm, 1.0 / zmin_mm  # 注意先大后小
        y = (x - xmin) / (xmax - xmin + 1e-8)
    elif mode.lower() == "log":
        # 对数映射（米），并取反让近处更亮
        tmp_m = tmp / 1000.0
        x = np.log1p(tmp_m)
        xmin, xmax = np.log1p(zmin_m), np.log1p(zmax_m)
        y = 1.0 - (x - xmin) / (xmax - xmin + 1e-8)
    else:
        raise ValueError("mode 只能是 'disparity' 或 'log'")

    y = np.clip(y, 0.0, 1.0)
    vis8 = (y * 255.0 + 0.5).astype(np.uint8)  # 四舍五入
    rgb = np.repeat(vis8[..., None], 3, axis=-1)  # 灰度复制到3通道
    rgb[~valid] = 0  # 无效像素置黑
    return rgb

# 1) 读入深度（米），做裁剪与视差/对数映射
def depth_to_unit(depth_m, zmin=0.2, zmax=4.0, mode="disparity"):
    depth_m = np.array(depth_m, dtype=np.float32)
    depth_m = np.clip(depth_m, zmin, zmax)
    if mode == "disparity":
        x = 1.0 / depth_m
        xmin, xmax = 1.0 / zmax, 1.0 / zmin
    else:  # "log"
        x = np.log1p(depth_m)
        xmin, xmax = np.log1p(zmin), np.log1p(zmax)
    x = (x - xmin) / (xmax - xmin + 1e-8)  # [0,1]
    return x


# 2) 堆成3通道并转PIL
def depth_to_rgb_like(depth_m):
    u = depth_to_unit(depth_m)  # [H,W] in [0,1]
    rgb = np.stack([u, u, u], axis=-1)  # [H,W,3]
    rgb = (rgb * 255.0).astype(np.uint8)
    return Image.fromarray(rgb)

def save_depth_images(
    base_image_depth,
    out_raw_path="depth_raw.png",
    out_vis_path="depth_vis.png",
    zmin_m=0.2, zmax_m=4.0,          # 预览用的可视范围（米）
    vis_mode="log",            # "disparity" 或 "log"
    apply_colormap=False              # 伪彩色
):
    d = np.asarray(base_image_depth)

    # 1) 统一成 uint16（单位 mm），用于无损保存
    print(d.dtype)
    if d.dtype.kind == "f":  # float: 认为是米
        d_m = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        d_u16 = np.clip(d_m * 1000.0, 0, 65535).astype(np.uint16)
    elif d.dtype == np.uint16:
        d_u16 = d
    else:
        # 其它整数类型，按毫米处理并安全转为 uint16
        d_u16 = np.clip(d.astype(np.int64), 0, 65535).astype(np.uint16)

    # 1a) 无损深度图：16-bit PNG（灰度），注意普通看图工具可能显示很暗但数据是正确的
    cv2.imwrite(out_raw_path, d_u16)

    # 2) 预览图：把深度映射到 [0,255] 再可选伪彩
    zmin_mm, zmax_mm = int(zmin_m * 1000), int(zmax_m * 1000)
    tmp = d_u16.copy()
    valid = tmp > 0
    # 无效像素设为最大深度，避免造成无穷/NaN
    tmp[~valid] = zmax_mm
    tmp = np.clip(tmp, zmin_mm, zmax_mm).astype(np.float32)

    if vis_mode == "disparity":
        x = 1.0 / tmp  # 视差
        xmin, xmax = 1.0 / zmax_mm, 1.0 / zmin_mm
    else:  # "log"
        x = np.log1p(tmp / 1000.0)  # log(1+米)
        xmin, xmax = np.log1p(zmin_m), np.log1p(zmax_m)

    x = (x - xmin) / (xmax - xmin + 1e-8)
    x = np.clip(x, 0.0, 1.0)
    vis8 = (x * 255.0).astype(np.uint8)

    if apply_colormap:
        vis_img = cv2.applyColorMap(vis8, cv2.COLORMAP_JET)
        # 把无效像素染成黑色（可选）
        vis_img[~valid] = (0, 0, 0)
    else:
        vis_img = vis8
        vis_img[~valid] = 0

    cv2.imwrite(out_vis_path, vis_img)

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
    use_depth: bool = False

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("camera0", "camera1", "camera2",
                                                   "camera3", "camera4", "camera0_depth", 
                                                   "camera2_depth", "camera3_depth")

    def __call__(self, data: dict) -> dict:
        # 仅在需要图像时才调用 _decode_aloha（其内部会访问 data["images"]）
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi, use_images=self.use_images)

        # ---- state ---- 
        # pose for now , for one arm: pose:0-7, velocity:7-14, effort:14-21
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
            left_wrist_image = in_images["camera4"]

            images = {
                "base_rgb": base_image,
                "right_wrist_rgb": right_wrist_image,
                "feng_rgb": feng_image,
                "bao_rgb": bao_image,
                "left_wrist_image": left_wrist_image,
            }
            image_masks = {
                "base_rgb": np.True_,
                "right_wrist_rgb": np.True_,
                "feng_rgb": np.True_,
                "bao_rgb": np.True_,
                "left_wrist_image": np.True_,
            }

            # Add the extra images.
            extra_image_names = {
            }

            if self.use_depth:
                base_image_depth = depth_rgb_u8_to_u16(in_images["camera0_depth"])
                base_image_depth_processed = depth_u16_to_u8x3(base_image_depth, mode="disparity")
                # print(base_image_depth.shape)
                # print(base_image_depth)
                # vis = ((base_image_depth.astype(np.float32) - base_image_depth[base_image_depth > 0].min())
                #        / (base_image_depth.max() - base_image_depth[base_image_depth > 0].min() + 1e-8) * 255).astype(
                #     np.uint8)
                # cv2.imwrite("depth_linear_hilo.png", vis)
                # cv2.imwrite("rgb.png", in_images["camera0"])
                # print(type(in_images["camera0"]))
                # print(np.asarray(in_images["camera0"]).dtype)
                # exit(1)
                feng_image_depth = depth_rgb_u8_to_u16(in_images["camera2_depth"])
                feng_image_depth_processed = depth_u16_to_u8x3(feng_image_depth, mode="disparity")
                bao_image_depth = depth_rgb_u8_to_u16(in_images["camera3_depth"])
                bao_image_depth_processed = depth_u16_to_u8x3(bao_image_depth, mode="disparity")

                images["base_depth"] = base_image_depth_processed
                images["feng_depth"] = feng_image_depth_processed
                images["bao_depth"] = bao_image_depth_processed
                image_masks["base_depth"] = np.True_
                image_masks["feng_depth"] = np.True_
                image_masks["bao_depth"] = np.True_

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
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


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
        value = (horn_radius ** 2 + linear_position ** 2 - arm_length ** 2) / (2 * horn_radius * linear_position)
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
        use_images: bool = True,  # ← 新增开关
) -> dict:
    # --- state 始终解码 ---
    state = np.asarray([*data["state"][:7], *data["state"][21:28]])
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
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions
