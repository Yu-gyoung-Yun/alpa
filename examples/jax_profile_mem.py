import jax
import jax.numpy as jnp
from jax import make_jaxpr
import math
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "" # -> CPU 사용. defaults는 모든 GPU가 다 사용됨.
print(f"jax.devices: {jax.devices()}")

def func1(x):
  return jnp.tile(x, 10) * 0.5

def func2(x):
  y = func1(x)
  return y, jnp.tile(x, 10) + 1

x = jax.random.normal(jax.random.PRNGKey(42), (1000, 1000))
y, z = func2(x)

z.block_until_ready()
closed_jaxpr = make_jaxpr(func2)(x) # ClosedJaxpr
for eqn in closed_jaxpr.jaxpr.eqns:
  print("eqn: ", eqn)
for inv in closed_jaxpr.jaxpr.invars:
  print("input: ", inv.aval.shape)
  print("GiB: ", math.prod(inv.aval.shape) / 2** 30)
#jax.profiler.save_device_memory_profile("memory.prof")