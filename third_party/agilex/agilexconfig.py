from dataclasses import dataclass, field
import abc
from pathlib import Path

import draccus

from ..cameras import CameraConfig



@dataclass(kw_only=True)
class RobotConfig(draccus.ChoiceRegistry, abc.ABC):
    # Allows to distinguish between different robots of the same type
    id: str | None = None
    # Directory to store calibration file
    calibration_dir: Path | None = None

    def __post_init__(self):
        if hasattr(self, "cameras") and self.cameras:
            for _, config in self.cameras.items():
                print(config)
                for attr in ["width", "height", "fps"]:
                    if getattr(config, attr) is None:
                        raise ValueError(
                            f"Specifying '{attr}' is required for the camera to be used in a robot"
                        )

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

@RobotConfig.register_subclass("aloha_agilex_follower")
@dataclass
class AlohaAgileXFollowerConfig(RobotConfig):
    # Port to connect to the arm
    port: str | None = "can_left"

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False

    # id
    id: str = "left"


