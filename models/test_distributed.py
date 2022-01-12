import jax
import jax.numpy as jnp
import os
from absl import app
from absl import flags
from absl import logging

flags.DEFINE_string('host_address', None, help='')
FLAGS = flags.FLAGS


def test_step(x):
  x = x + jax.lax.pmean(x, axis_name='i')
  return x + 1


def main(argv):
  world_size = int(os.environ.get('WORLD_SIZE', '0'))
  node_rank = int(os.environ.get('NODE_RANK', '0'))
  if world_size > 0:
    jax.distributed.initialize(FLAGS.host_address, world_size, node_rank)
    print('global devices=', jax.devices())
    print('local devices=', jax.local_devices())
    print('host id=', jax.host_id())
    print('host count=', jax.host_count())

  pmap_operation = jax.pmap(test_step, axis_name='i')
  print("start testing, local_devices_count=", len(jax.local_devices()))
  sample = jnp.ones((len(jax.local_devices()), 2)) * node_rank
  print(sample)
  sample = pmap_operation(sample)
  print(sample)


if __name__ == '__main__':
  app.run(main)
