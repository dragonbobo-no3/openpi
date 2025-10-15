fine tune
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 XLA_PYTHON_CLIENT_MEM_PREALLOCATE=false uv run scripts/train.py pi0_agileX --exp-name=/home/agx/jemodel/test --data.repo_id=lerobot/test --overwrite --no_wandb_enabled

NCCL_NVLS_ENABLE=0 uv run scripts/train.py pi05_agileX --exp-name=/jedata/jemotor/model/0911_pi05_test --data.repo_id=lerobot/test --resume --no_wandb_enabled

inference

evaluate

record
python -m lerobot.record_aloha_agilex_single_arm    --robot1.type=aloha_agilex_follower    --robot1.port=can_right            --robot1.id=right            --teleop.type=aloha_agilex_leader            --teleop.port=/dev/tty.usbmodem58760431551            --teleop.id=blue        --robot1.cameras="{camera0: {type: orbbec, index_or_path: CP02653000ZL, width: 640, height: 480, fps: 30},camera1: {type: orbbec, index_or_path: CP02653000YJ, width: 640, height: 480, fps: 30}, camera2: {type: orbbec, index_or_path: CP02653000YR, width: 640, height: 480, fps: 30}, camera3: {type: orbbec, index_or_path: CP02653000R4, width: 640, height: 480, fps: 30}}"        --dataset.single_task="Pick up the PCB board from the green conveyor belt and place it into the yellow container."            --robot1.tactiles="{
