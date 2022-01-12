# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import contextlib
import functools
import timeit
from typing import Iterable, Mapping, NamedTuple, Tuple, Callable, Any
from functools import partial
import os

from absl import app
from absl import flags
from absl import logging
import haiku as hk
from models import dataset
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import xarray
import tree

import gin
import tensorflow.compat.v1 as tf
import jax.lib.xla_bridge as xb
import pkg_resources
import jax_cfd.base as cfd
import jax_cfd.ml as cfd_ml
from my_model_builder import my_decoded_trajectory_with_inputs
from flax import jax_utils
from flax.training import checkpoints

try:
  tf.flags.DEFINE_multi_string("gin_file", None, "Path to a Gin file.")
  tf.flags.DEFINE_multi_string("gin_param", None, "Gin parameter binding.")
  tf.flags.DEFINE_list("gin_location_prefix", [], "Gin file search path.")
except tf.flags.DuplicateFlagError:
  pass

# Hyper parameters.
flags.DEFINE_integer('eval_batch_size', 1000, help='')
flags.DEFINE_integer('train_device_batch_size', 128, help='')
flags.DEFINE_integer('train_eval_every', -1, help='')
flags.DEFINE_integer('train_init_random_seed', 42, help='')
flags.DEFINE_integer('train_log_every', 100, help='')
flags.DEFINE_float('train_epochs', 90, help='')
flags.DEFINE_float('train_lr_warmup_epochs', 5, help='')
flags.DEFINE_float('train_lr_init', 0.1, help='')
flags.DEFINE_float('train_weight_decay', 1e-4, help='')
flags.DEFINE_string('mp_policy', 'p=f32,c=f32,o=f32', help='')
flags.DEFINE_string('mp_bn_policy', 'p=f32,c=f32,o=f32', help='')
flags.DEFINE_enum('mp_scale_type', 'NoOp', ['NoOp', 'Static', 'Dynamic'], help='')
flags.DEFINE_float('mp_scale_value', 2 ** 15, help='')
flags.DEFINE_bool('mp_skip_nonfinite', False, help='')
flags.DEFINE_bool('dataset_transpose', False, help='')
flags.DEFINE_bool('dataset_zeros', False, help='')

# My Hyper parameters.
flags.DEFINE_string('eval_split', 'TEST', help='')
flags.DEFINE_string('predict_split', None, help='')
flags.DEFINE_string('predict_result', "predict.nc", help='')
flags.DEFINE_string('train_split', 'TRAIN_AND_VALID', help='')
flags.DEFINE_float('delta_time', 0.001, help='')
flags.DEFINE_float('simulation_time', 30.0, help='')
flags.DEFINE_integer('model_input_size', 64, help='')
flags.DEFINE_integer('save_grid_size', 64, help='')
flags.DEFINE_integer('model_encode_steps', 64, help='')
flags.DEFINE_integer('model_decode_steps', 1, help='')
flags.DEFINE_integer('model_predict_steps', 64, help='')
flags.DEFINE_bool('use_real_resnet', False, help='')
flags.DEFINE_float('model_bn_decay', 0.9, help='')
flags.DEFINE_bool('use_exponential_decay', False, help='')
flags.DEFINE_string('output_dir', None, help='')
flags.DEFINE_bool('resume_checkpoint', False, help='')
flags.DEFINE_bool('warm_start', False, help='')
flags.DEFINE_bool('no_dropout', False, help='')
flags.DEFINE_integer('warm_start_steps', 100, help='')
flags.DEFINE_float('max_velocity', 7.0, help='')
flags.DEFINE_bool('do_eval', False, help='')
flags.DEFINE_bool('do_predict', False, help='')
flags.DEFINE_bool('no_train', False, help='')
flags.DEFINE_integer('decoding_warmup_steps', 0, help='')
flags.DEFINE_integer('decoding_warmup_stages', 1, help='')
flags.DEFINE_integer('inner_steps', 1, help='')
flags.DEFINE_integer('explicit_inner_steps', 1, help='')
flags.DEFINE_string('host_address', None, help='')


FLAGS = flags.FLAGS
Scalars = Mapping[str, jnp.ndarray]


class TrainState(NamedTuple):
  step: int
  params: hk.Params
  state: hk.State
  opt_state: optax.OptState
  loss_scale: jmp.LossScale


class SaveState(NamedTuple):
  step: int
  params: hk.Params
  state: hk.State
  opt_state: optax.OptState


get_policy = lambda: jmp.get_policy(FLAGS.mp_policy)
get_bn_policy = lambda: jmp.get_policy(FLAGS.mp_bn_policy)


def get_initial_loss_scale() -> jmp.LossScale:
  cls = getattr(jmp, f'{FLAGS.mp_scale_type}LossScale')
  return cls(FLAGS.mp_scale_value) if cls is not jmp.NoOpLossScale else cls()


def _forward(
    batch: dataset.Batch,
    is_training: bool,
    inner_steps: int,
    outer_steps: int,
) -> jnp.ndarray:
  """Forward application of the resnet."""
  inputs = batch['inputs']
  size = FLAGS.model_input_size
  grid = cfd.grids.Grid((size, size), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
  dt = FLAGS.delta_time
  physics_specs = cfd_ml.physics_specifications.get_physics_specs()
  stable_time_step = cfd.equations.stable_time_step(FLAGS.max_velocity, 0.5,
                                                    physics_specs.viscosity, grid, implicit_diffusion=True)
  logging.info("Stable time step: %.10f" % stable_time_step)
  inner_steps = inner_steps * round(dt / stable_time_step)
  model = cfd_ml.model_builder.get_model_cls(grid, stable_time_step, physics_specs)()

  if FLAGS.no_dropout:
    trajectory = jax.vmap(
        partial(
            cfd_ml.model_utils.decoded_trajectory_with_inputs(
                model=model,
                num_init_frames=FLAGS.model_encode_steps),
            outer_steps=outer_steps,
            inner_steps=inner_steps,
        ),
        axis_name='i')
  else:
    trajectory = jax.vmap(
        partial(
            my_decoded_trajectory_with_inputs(
                model=model,
                num_init_frames=FLAGS.model_encode_steps),
            outer_steps=outer_steps,
            inner_steps=inner_steps,
            is_training=is_training,
        ),
        axis_name='i')

  final, predictions = trajectory(inputs)
  return predictions


# Transform our forwards function into a pair of pure functions.
forward = hk.transform_with_state(_forward)


def lr_schedule(step: jnp.ndarray, num_examples: int) -> jnp.ndarray:
  """Cosine learning rate schedule."""
  total_batch_size = FLAGS.train_device_batch_size * jax.device_count()
  steps_per_epoch = num_examples / total_batch_size
  warmup_steps = int(FLAGS.train_lr_warmup_epochs * steps_per_epoch)
  training_steps = int(FLAGS.train_epochs * steps_per_epoch)
  logging.info(f'warmup_steps: {warmup_steps}')
  logging.info(f'training_steps: {training_steps}')

  if FLAGS.use_exponential_decay:
    lr = FLAGS.train_lr_init
    decay_rate = 0.1
    decay_steps = 300000.0
    lr = lr * decay_rate ** (step / decay_steps)
  else:
    lr = FLAGS.train_lr_init
    scaled_step = (jnp.maximum(step - warmup_steps, 0) /
                   (training_steps - warmup_steps))
    lr *= 0.5 * (1.0 + jnp.cos(jnp.pi * scaled_step))
    if warmup_steps:
      lr *= jnp.minimum(step / warmup_steps, 1.0)
  return lr


def make_optimizer(num_examples: int) -> optax.GradientTransformation:
  """SGD with nesterov momentum and a custom lr schedule."""
  return optax.chain(
      # optax.trace(decay=0.9),
      optax.clip_by_global_norm(0.25),
      optax.scale_by_adam(b2=0.98, eps=1e-06),
      # optax.add_decayed_weights(weight_decay=FLAGS.train_weight_decay),
      optax.scale_by_schedule(
          partial(lr_schedule, num_examples=num_examples)),
      optax.scale(-1))


def loss_fn(
    params: hk.Params,
    state: hk.State,
    loss_scale: jmp.LossScale,
    batch: dataset.Batch,
    inner_steps: int,
    decode_steps: int,
    dropout_rng: jax.random.PRNGKey,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, hk.State]]:
  """Computes a regularized loss for the given batch."""
  predictions, state = forward.apply(params, state, dropout_rng, batch, is_training=True,
                                     inner_steps=inner_steps, outer_steps=decode_steps)
  targets = batch['inputs']

  pu = predictions[0][:, FLAGS.model_encode_steps: FLAGS.model_encode_steps + decode_steps]
  pv = predictions[1][:, FLAGS.model_encode_steps: FLAGS.model_encode_steps + decode_steps]
  tu = targets[0][:, FLAGS.model_encode_steps: FLAGS.model_encode_steps + decode_steps]
  tv = targets[1][:, FLAGS.model_encode_steps: FLAGS.model_encode_steps + decode_steps]

  loss_u = optax.l2_loss(predictions=pu, targets=tu).mean()
  loss_v = optax.l2_loss(predictions=pv, targets=tv).mean()
  loss = loss_u + loss_v
  return loss_scale.scale(loss), (loss, state)


def train_step(
    train_state: TrainState,
    batch: dataset.Batch,
    inner_steps: int,
    decode_steps: int,
    num_examples: int,
    dropout_rng=None,
) -> Tuple[TrainState, Scalars, Any]:
  """Applies an update to parameters and returns new state."""
  step, params, state, opt_state, loss_scale = train_state
  dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
  grads, (loss, new_state) = (
      jax.grad(loss_fn, has_aux=True)(params, state, loss_scale, batch,
                                      inner_steps, decode_steps, dropout_rng))

  # Grads are in "param_dtype" (likely F32) here. We cast them back to the
  # compute dtype such that we do the all-reduce below in the compute precision
  # (which is typically lower than the param precision).
  policy = get_policy()
  grads = policy.cast_to_compute(grads)
  grads = loss_scale.unscale(grads)

  # Taking the mean across all replicas to keep params in sync.
  grads = jax.lax.pmean(grads, axis_name='i')

  # We compute our optimizer update in the same precision as params, even when
  # doing mixed precision training.
  grads = policy.cast_to_param(grads)

  # Compute and apply updates via our optimizer.
  updates, new_opt_state = make_optimizer(num_examples).update(grads, opt_state, params)
  new_params = optax.apply_updates(params, updates)

  if FLAGS.mp_skip_nonfinite:
    grads_finite = jmp.all_finite(grads)
    loss_scale = loss_scale.adjust(grads_finite)
    new_params, new_state, new_opt_state = jmp.select_tree(
        grads_finite,
        (new_params, new_state, new_opt_state),
        (params, state, opt_state))

  # Scalars to log (note: we log the mean across all hosts/devices).
  scalars = {'train_loss': loss, 'loss_scale': loss_scale.loss_scale}
  if FLAGS.mp_skip_nonfinite:
    scalars['grads_finite'] = grads_finite
  state, scalars = jmp.cast_to_full((state, scalars))
  scalars = jax.lax.pmean(scalars, axis_name='i')
  new_step = step + 1
  train_state = TrainState(new_step, new_params, new_state, new_opt_state, loss_scale)
  return train_state, scalars, new_dropout_rng


def initial_state(rng: jnp.ndarray, batch: dataset.Batch, num_examples: int) -> TrainState:
  """Computes the initial network state."""
  params, state = forward.init(rng, batch, is_training=True, inner_steps=1, outer_steps=13)
  opt_state = make_optimizer(num_examples).init(params)
  loss_scale = get_initial_loss_scale()
  train_state = TrainState(0, params, state, opt_state, loss_scale)
  return train_state


# NOTE: We use `jit` not `pmap` here because we want to ensure that we see all
# eval data once and this is not easily satisfiable with pmap (e.g. n=3).
# TODO(tomhennigan) Find a solution to allow pmap of eval.
def eval_batch(
    params: hk.Params,
    state: hk.State,
    batch: dataset.Batch,
    inner_steps: int,
) -> [jnp.ndarray, jnp.ndarray]:
  """Evaluates a batch."""
  predictions, _ = forward.apply(params, state, None, batch, is_training=False,
                                 inner_steps=inner_steps, outer_steps=FLAGS.model_predict_steps)
  targets = batch['inputs']
  weights = batch['weights']

  pu = predictions[0][:, -FLAGS.model_predict_steps:]
  pv = predictions[1][:, -FLAGS.model_predict_steps:]
  tu = targets[0][:, -FLAGS.model_predict_steps:]
  tv = targets[1][:, -FLAGS.model_predict_steps:]

  loss_u = optax.l2_loss(predictions=pu, targets=tu).mean(axis=list(range(1, pv.ndim)))
  loss_v = optax.l2_loss(predictions=pv, targets=tv).mean(axis=list(range(1, pv.ndim)))
  loss = loss_u + loss_v

  loss = jnp.where(weights > 0, loss, 0)
  loss = jax.lax.pmean(loss, axis_name='i')
  weights = jax.lax.pmean(weights, axis_name='i')

  return loss.astype(jnp.float32), weights


# NOTE: We use `jit` not `pmap` here because we want to ensure that we see all
# eval data once and this is not easily satisfiable with pmap (e.g. n=3).
# TODO(tomhennigan) Find a solution to allow pmap of eval.
def predict_batch(
    params: hk.Params,
    state: hk.State,
    batch: dataset.Batch,
    inner_steps: int,
) -> (jnp.ndarray, jnp.ndarray):
  """Evaluates a batch."""
  predictions, _ = forward.apply(params, state, None, batch, is_training=False,
                                 inner_steps=inner_steps, outer_steps=FLAGS.model_predict_steps)

  source_grid = cfd.grids.Grid((FLAGS.model_input_size, FLAGS.model_input_size),
                               domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
  destination_grid = cfd.grids.Grid((FLAGS.save_grid_size, FLAGS.save_grid_size),
                                    domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))

  def my_downsample(x):
    return cfd.resize.downsample_staggered_velocity(source_grid, destination_grid, x)
  my_downsample = jax.vmap(my_downsample, in_axes=((0, 0),), out_axes=(0, 0))
  my_downsample = jax.vmap(my_downsample, in_axes=((0, 0),), out_axes=(0, 0))
  des_predictions = my_downsample(predictions)

  des_predictions = (des_predictions[0][:, -FLAGS.model_predict_steps:],
                     des_predictions[1][:, -FLAGS.model_predict_steps:])

  predictions = (predictions[0][:, -FLAGS.model_encode_steps:],
                 predictions[1][:, -FLAGS.model_encode_steps:])

  return predictions, des_predictions


def evaluate(
    split: str,
    params: hk.Params,
    state: hk.State,
) -> Scalars:
  """Evaluates the model at the given params/state."""
  local_device_count = jax.local_device_count()
  test_data = dataset.DataLoader(split,
                                 is_training=False,
                                 batch_dims=[local_device_count, FLAGS.eval_batch_size],
                                 input_size=FLAGS.model_input_size,
                                 encode_steps=FLAGS.model_encode_steps,
                                 decode_steps=FLAGS.model_predict_steps,
                                 transpose=FLAGS.dataset_transpose,
                                 delta_time=FLAGS.delta_time,
                                 zeros=FLAGS.dataset_zeros)
  test_dataset = test_data.load()

  # if test_data.num_examples % FLAGS.eval_batch_size:
  #   raise ValueError(f'Eval batch size {FLAGS.eval_batch_size} must be a '
  #                    f'multiple of {split} num examples {test_data.num_examples}')

  # Params/state are sharded per-device during training. We just need the copy
  # from the first device (since we do not pmap evaluation at the moment).
  # params, state = jax.tree_map(lambda x: x[0], (params, state))
  l2_loss = jnp.array(0)
  weights = jnp.array(0)
  my_eval_batch = jax.pmap(partial(eval_batch, inner_steps=test_data.inner_steps), axis_name='i')
  total_eval_steps = test_data.num_examples // (FLAGS.eval_batch_size * jax.device_count())
  for iter, batch in enumerate(test_dataset):
    per_batch_l2_loss, per_batch_weights = my_eval_batch(params, state, batch)
    l2_loss += jnp.mean(per_batch_l2_loss)
    weights += jnp.mean(per_batch_weights)
    iter += 1
    if iter % FLAGS.train_log_every == 0:
      logging.info(f'Evaluated {iter}/{total_eval_steps} with L2 loss {l2_loss / weights}')

  return {'rmse': (l2_loss / weights) ** 0.5}


@contextlib.contextmanager
def time_activity(activity_name: str):
  logging.info('[Timing] %s start.', activity_name)
  start = timeit.default_timer()
  yield
  duration = timeit.default_timer() - start
  logging.info('[Timing] %s finished (Took %.2fs).', activity_name, duration)


def parse_gin_defaults_and_flags(skip_unknown=False, finalize_config=True):
  """Parses all default gin files and those provided via flags."""
  # Register .gin file search paths with gin
  for gin_file_path in FLAGS.gin_location_prefix:
    gin.add_config_file_search_path(gin_file_path)
  # Set up the default values for the configurable parameters. These values will
  # be overridden by any user provided gin files/parameters.
  gin.parse_config_files_and_bindings(
      FLAGS.gin_file, FLAGS.gin_param,
      skip_unknown=skip_unknown,
      finalize_config=finalize_config)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  world_size = int(os.environ.get('WORLD_SIZE', '0'))
  node_rank = int(os.environ.get('NODE_RANK', '0'))
  if world_size > 0:
    jax.distributed.initialize(FLAGS.host_address, world_size, node_rank)
    print('global devices=', jax.devices())
    print('local devices=', jax.local_devices())

  # Add search path for gin files stored in package.
  gin.add_config_file_search_path(
      pkg_resources.resource_filename(__name__, "gin"))

  parse_gin_defaults_and_flags()
  FLAGS.alsologtostderr = True

  local_device_count = jax.local_device_count()
  train_data = dataset.DataLoader(
      FLAGS.train_split,
      is_training=True,
      batch_dims=[local_device_count, FLAGS.train_device_batch_size],
      input_size=FLAGS.model_input_size,
      encode_steps=FLAGS.model_encode_steps,
      decode_steps=FLAGS.model_decode_steps,
      dtype=get_policy().compute_dtype,
      transpose=FLAGS.dataset_transpose,
      delta_time=FLAGS.delta_time,
      zeros=FLAGS.dataset_zeros)

  train_dataset = train_data.load()

  # The total batch size is the batch size accross all hosts and devices. In a
  # multi-host training setup each host will only see a batch size of
  # `total_train_batch_size / jax.host_count()`.
  total_train_batch_size = FLAGS.train_device_batch_size * jax.device_count()
  num_train_steps = (
      (train_data.num_examples * FLAGS.train_epochs) // total_train_batch_size)
  num_train_steps = int(num_train_steps)

  # Assign mixed precision policies to modules. Note that when training in f16
  # we keep BatchNorm in  full precision. When training with bf16 you can often
  # use bf16 for BatchNorm.
  mp_policy = get_policy()
  bn_policy = get_bn_policy().with_output_dtype(mp_policy.compute_dtype)
  # NOTE: The order we call `set_policy` doesn't matter, when a method on a
  # class is called the policy for that class will be applied, or it will
  # inherit the policy from its parent module.
  hk.mixed_precision.set_policy(hk.BatchNorm, bn_policy)
  hk.mixed_precision.set_policy(forward, mp_policy)

  if jax.default_backend() == 'gpu':
    # TODO(tomhennigan): This could be removed if XLA:GPU's allocator changes.
    train_dataset = dataset.double_buffer(train_dataset)

  # For initialization we need the same random key on each device.
  rng = jax.random.PRNGKey(FLAGS.train_init_random_seed)
  dropout_rngs = jax.random.split(rng, jax.local_device_count())
  # Initialization requires an example input.
  # batch = next(train_dataset)
  # train_state = initial_state(rng, jax.tree_map(lambda x: x[0], batch),
  #                             num_examples=train_data.num_examples)
  sample_vec = np.zeros((1, FLAGS.model_encode_steps, FLAGS.model_input_size, FLAGS.model_input_size))
  train_state = initial_state(rng, {'inputs': (sample_vec, sample_vec)}, num_examples=train_data.num_examples)

  start_step = 0
  if FLAGS.resume_checkpoint is not None:
    logging.info("Loading from %s" % FLAGS.output_dir)
    i1, i2, i3, i4, i5 = train_state
    save_state = SaveState(i1, i2, i3, i4)
    save_state = checkpoints.restore_checkpoint(FLAGS.output_dir, save_state)
    i1, i2, i3, i4 = save_state
    train_state = TrainState(i1, i2, i3, i4, i5)
    start_step = int(train_state.step)

  train_state = jax_utils.replicate(train_state)
  if FLAGS.decoding_warmup_steps == 0:
    my_train_step = partial(train_step, num_examples=train_data.num_examples,
                            inner_steps=train_data.inner_steps, decode_steps=FLAGS.model_decode_steps)
    my_train_step = jax.pmap(my_train_step, axis_name='i', donate_argnums=(0,))
  else:
    if FLAGS.resume_checkpoint is not None:
      current_stage = int(float(start_step + 1) / FLAGS.decoding_warmup_steps / FLAGS.decoding_warmup_stages * FLAGS.model_decode_steps) + 1
      current_stage = current_stage * FLAGS.decoding_warmup_stages
      next_stage = min(current_stage, FLAGS.model_decode_steps)
      model_decode_steps = next_stage
      logging.info("Decoding warmup stage %d" % next_stage)
      my_train_step = partial(train_step, num_examples=train_data.num_examples,
                              inner_steps=train_data.inner_steps, decode_steps=model_decode_steps)
      my_train_step = jax.pmap(my_train_step, axis_name='i', donate_argnums=(0,))
    else:
      my_train_step = partial(train_step, num_examples=train_data.num_examples,
                              inner_steps=train_data.inner_steps, decode_steps=FLAGS.model_decode_steps)
      my_train_step = jax.pmap(my_train_step, axis_name='i', donate_argnums=(0,))

  # Print a useful summary of the execution of our module.
  # summary = hk.experimental.tabulate(my_train_step)(train_state, batch)
  # for line in summary.split('\n'):
  #   logging.info(line)

  eval_every = FLAGS.train_eval_every
  log_every = FLAGS.train_log_every

  with time_activity('train'):
    for step_num in range(start_step, num_train_steps):
      if FLAGS.no_train:
        break
      # Take a single training step.
      with jax.profiler.StepTraceAnnotation('train', step_num=step_num):
        batch = next(train_dataset)

        if FLAGS.warm_start and step_num < FLAGS.warm_start_steps:
          batch = jax.tree_map(lambda x: x[:, :1], batch)

        if FLAGS.decoding_warmup_steps == 0:
          train_state, train_scalars, dropout_rngs = my_train_step(
              train_state, batch, dropout_rng=dropout_rngs)
        else:
          current_stage = int(float(step_num) / FLAGS.decoding_warmup_steps /
                              FLAGS.decoding_warmup_stages * FLAGS.model_decode_steps) + 1
          current_stage = current_stage * FLAGS.decoding_warmup_stages
          stage = min(current_stage, FLAGS.model_decode_steps)
          current_stage = int(float(step_num + 1) / FLAGS.decoding_warmup_steps /
                              FLAGS.decoding_warmup_stages * FLAGS.model_decode_steps) + 1
          current_stage = current_stage * FLAGS.decoding_warmup_stages
          next_stage = min(current_stage, FLAGS.model_decode_steps)
          if stage != next_stage:
            logging.info("Decoding warmup stage %d" % next_stage)
            del my_train_step
            model_decode_steps = next_stage
            my_train_step = partial(train_step, num_examples=train_data.num_examples,
                                    inner_steps=train_data.inner_steps, decode_steps=model_decode_steps)
            my_train_step = jax.pmap(my_train_step, axis_name='i', donate_argnums=(0,))
            # i1, i2, i3, i4, i5 = train_state
            # del train_state
            # train_state = TrainState(i1, i2, i3, jax.tree_map(lambda v: jnp.zeros_like(v), i4), i5)
          train_state, train_scalars, dropout_rngs = my_train_step(
              train_state, batch, dropout_rng=dropout_rngs)

      # By default we do not evaluate during training, but you can configure
      # this with a flag.
      if eval_every > 0 and step_num and step_num % eval_every == 0:
        with time_activity('eval during train'):
          eval_scalars = evaluate(FLAGS.eval_split,
                                  train_state.params, train_state.state)
        logging.info('[Eval %s/%s] %s', step_num, num_train_steps, eval_scalars)

      # Log progress at fixed intervals.
      if step_num and step_num % log_every == 0:
        train_scalars = jax.tree_map(lambda v: np.mean(v).item(),
                                     jax.device_get(train_scalars))
        logging.info('[Train %s/%s] %s',
                     step_num, num_train_steps, train_scalars)

        if step_num and step_num % (log_every * 100) == 0 and jax.host_id() == 0:
          if FLAGS.output_dir is not None:
            i1, i2, i3, i4, i5 = jax_utils.unreplicate(train_state)
            save_state = SaveState(i1, i2, i3, i4)
            checkpoints.save_checkpoint(
                FLAGS.output_dir, save_state, step_num, keep=3)

  if FLAGS.output_dir is not None and start_step + 1 < num_train_steps and jax.host_id() == 0:
    i1, i2, i3, i4, i5 = jax_utils.unreplicate(train_state)
    save_state = SaveState(i1, i2, i3, i4)
    checkpoints.save_checkpoint(
        FLAGS.output_dir, save_state, step_num, keep=3)

  if FLAGS.do_eval:
    # Once training has finished we run eval one more time to get final results.
    with time_activity('final eval'):
      eval_scalars = evaluate(FLAGS.eval_split, train_state.params, train_state.state)
    logging.info('[Eval FINAL]: %s', eval_scalars)

  if FLAGS.do_predict and jax.host_id() == 0:
    eval_data = xarray.open_dataset(FLAGS.predict_split or FLAGS.eval_split)
    eval_u = eval_data.variables['u'].values
    eval_v = eval_data.variables['v'].values
    total_steps = int(FLAGS.simulation_time / (FLAGS.delta_time * FLAGS.inner_steps * FLAGS.explicit_inner_steps))
    original_shape = eval_u.shape
    logging.info(original_shape)

    eval_u = eval_u.reshape((local_device_count, -1) + eval_u.shape[1:])
    eval_v = eval_v.reshape((local_device_count, -1) + eval_v.shape[1:])

    source_grid = cfd.grids.Grid((eval_u.shape[-1], eval_u.shape[-1]),
                                 domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    destination_grid = cfd.grids.Grid((FLAGS.model_input_size, FLAGS.model_input_size),
                                      domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))

    def my_downsample(x):
      return cfd.resize.downsample_staggered_velocity(source_grid, destination_grid, x)
    my_downsample = jax.vmap(my_downsample, in_axes=((0, 0),), out_axes=(0, 0))
    my_downsample = jax.vmap(my_downsample, in_axes=((0, 0),), out_axes=(0, 0))
    my_downsample = jax.pmap(my_downsample, axis_name='i')

    batch = {"inputs": my_downsample((eval_u[:, :, :FLAGS.model_encode_steps],
                                      eval_v[:, :, :FLAGS.model_encode_steps]))}

    u_batch = []
    v_batch = []
    # u_batch.append(batch["inputs"][0][:, :, :FLAGS.model_encode_steps])
    # v_batch.append(batch["inputs"][1][:, :, :FLAGS.model_encode_steps])

    step = FLAGS.model_encode_steps
    my_predict_batch = jax.pmap(partial(predict_batch, inner_steps=FLAGS.inner_steps),
                                axis_name='i', donate_argnums=(0,))
    while step < total_steps:
      logging.info('[Predict %s/%s]', step, total_steps)
      prediction, des_prediction = my_predict_batch(train_state.params, train_state.state, batch)
      u_batch.append(des_prediction[0])
      v_batch.append(des_prediction[1])

      # new_u = [prediction[0][:, :, -FLAGS.model_encode_steps:],
      #          prediction[0][:, :, :FLAGS.model_predict_steps]]
      # new_u = jnp.concatenate(new_u, axis=2)
      #
      # new_v = [prediction[1][:, :, -FLAGS.model_encode_steps:],
      #          prediction[1][:, :, :FLAGS.model_predict_steps]]
      # new_v = jnp.concatenate(new_v, axis=2)
      new_u = prediction[0]
      new_v = prediction[1]

      batch = {"inputs": (new_u, new_v)}
      step += (FLAGS.model_predict_steps // FLAGS.explicit_inner_steps)

    u_batch = jnp.concatenate(u_batch, axis=2).reshape(
        original_shape[:1] + (-1, FLAGS.save_grid_size, FLAGS.save_grid_size))[:, FLAGS.explicit_inner_steps-1::FLAGS.explicit_inner_steps]
    v_batch = jnp.concatenate(v_batch, axis=2).reshape(
        original_shape[:1] + (-1, FLAGS.save_grid_size, FLAGS.save_grid_size))[:, FLAGS.explicit_inner_steps-1::FLAGS.explicit_inner_steps]

    u_batch = xarray.Variable(("sample", "time", "x", "y"), u_batch)
    v_batch = xarray.Variable(("sample", "time", "x", "y"), v_batch)

    logging.info(u_batch.shape)

    grid = cfd.grids.Grid((FLAGS.save_grid_size, FLAGS.save_grid_size),
                          domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    x_len = grid.axes()[0].shape[0]
    x = 2 * np.double(grid.axes()[0]).mean() / x_len * np.arange(x_len)

    y_len = grid.axes()[1].shape[0]
    y = 2 * np.double(grid.axes()[1]).mean() / y_len * np.arange(y_len)

    ds = xarray.Dataset(
        {
            'u': (('sample', 'time', 'x', 'y'), u_batch),
            'v': (('sample', 'time', 'x', 'y'), v_batch),
        },
        coords={
            'time': (FLAGS.delta_time * FLAGS.inner_steps *
                     FLAGS.explicit_inner_steps * np.arange(u_batch.shape[1])),
            'x': x,
            'y': y,
            'sample': np.arange(u_batch.shape[0]),
        }
    )

    if not os.path.exists(FLAGS.output_dir):
      os.makedirs(FLAGS.output_dir)
    ds.to_netcdf(os.path.join(FLAGS.output_dir, FLAGS.predict_result))


if __name__ == '__main__':
  dataset.check_versions()
  app.run(main)
