fine tune
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 XLA_PYTHON_CLIENT_MEM_PREALLOCATE=false uv run scripts/train.py pi0_agileX --exp-name=/home/agx/jemodel/test --data.repo_id=lerobot/test --overwrite --no_wandb_enabled

NCCL_NVLS_ENABLE=0 XLA_PYTHON_CLIENT_MEM_PREALLOCATE=false uv run scripts/train.py pi05_agileX --exp-name=/jedata/jemotor/model/0911_pi05_test --data.repo_id=lerobot/test --overwrite --no_wandb_enabled

inference

evaluate