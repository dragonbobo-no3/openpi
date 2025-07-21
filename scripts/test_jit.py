import jax
from jax import random, numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
# 定义初始化函数
def init(key):
    w_key, b_key = random.split(key)
    w = random.normal(w_key, (10,))
    b = jnp.zeros(())
    return {'params': {'w': w, 'b': b}}

print(f"Default device:{jax.default_backend()}")  # 查看可用设备列表

# 设置设备和分片策略
devices = jax.devices()[:2]
print(devices)
mesh = Mesh(devices, ('data',))
spec_w = P('data')  # 向量分片策略
spec_b = P()        # 标量分片策略
sharding_w = NamedSharding(mesh, spec_w)
sharding_b = NamedSharding(mesh, spec_b)

# 执行初始化
with mesh:
    # 只对w分片，对b使用标量分片
    @jax.jit
    def init_and_shard(key):
        state = init(key)
        w = jax.device_put(state['params']['w'], sharding_w)
        b = jax.device_put(state['params']['b'], sharding_b)
        return {'params': {'w': w, 'b': b}}
    train_state = init_and_shard(random.PRNGKey(42))
    
print(train_state)  # 查看分片后的训练状态结构