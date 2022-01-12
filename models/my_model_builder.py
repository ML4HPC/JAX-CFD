import functools
from typing import Callable, Optional, Any

import gin
import haiku as hk
from functools import partial
from jax_cfd.base import grids
from jax_cfd.base import array_utils
# Note: decoders, encoders and equations contain standard gin-configurables;
from jax_cfd.ml import decoders  # pylint: disable=unused-import
from jax_cfd.ml import encoders  # pylint: disable=unused-import
from jax_cfd.ml import equations  # pylint: disable=unused-import
from jax_cfd.ml import physics_specifications
from jax_cfd.ml import model_builder
from jax_cfd.ml import model_utils


@gin.configurable
class MyModularStepModel(model_builder.DynamicalSystem):
  """Dynamical model based on independent encoder/decoder/step components."""

  def __init__(
      self,
      grid: grids.Grid,
      dt: float,
      physics_specs: physics_specifications.BasePhysicsSpecs,
      advance_module=gin.REQUIRED,
      encoder_module=gin.REQUIRED,
      decoder_module=gin.REQUIRED,
      name: Optional[str] = None
  ):
    """Constructs an instance of a class."""
    super().__init__(grid=grid, dt=dt, physics_specs=physics_specs, name=name)
    self.advance_module = advance_module(grid, dt, physics_specs)
    self.encoder_module = encoder_module(grid, dt, physics_specs)
    self.decoder_module = decoder_module(grid, dt, physics_specs)

  def encode(self, x):
    return self.encoder_module(x)

  def decode(self, x):
    return self.decoder_module(x)

  def advance(self, x, is_training=False):
    return self.advance_module(x, is_training)

  def trajectory(
      self,
      x,
      outer_steps: int,
      inner_steps: int = 1,
      *,
      start_with_input: bool = False,
      is_training: bool = False,
      post_process_fn: Callable = model_builder._identity,
  ):
    """Returns a final model state and trajectory."""
    return my_trajectory_from_step(
        self.advance, outer_steps, inner_steps,
        start_with_input=start_with_input,
        post_process_fn=post_process_fn,
        is_training=is_training,
    )(x)


@gin.configurable(allowlist=("set_checkpoint",))
def my_trajectory_from_step(
    step_fn: Callable,
    outer_steps: int,
    inner_steps: int,
    *,
    start_with_input: bool,
    post_process_fn: Callable,
    is_training: bool = False,
    set_checkpoint: bool = False,
):
  """Returns a function that accumulates repeated applications of `step_fn`.
  Compute a trajectory by repeatedly calling `step_fn()`
  `outer_steps * inner_steps` times.
  Args:
    step_fn: function that takes a state and returns state after one time step.
    outer_steps: number of steps to save in the generated trajectory.
    inner_steps: number of repeated calls to step_fn() between saved steps.
    start_with_input: if True, output the trajectory at steps [0, ..., steps-1]
      instead of steps [1, ..., steps].
    post_process_fn: function to apply to trajectory outputs.
    is_training: whether to use dropout.
    set_checkpoint: whether to use `jax.checkpoint` on `step_fn`.
  Returns:
    A function that takes an initial state and returns a tuple consisting of:
      (1) the final frame of the trajectory.
      (2) trajectory of length `outer_steps` representing time evolution.
  """
  step_fn = partial(step_fn, is_training=is_training)
  if set_checkpoint:
    step_fn = hk.remat(step_fn)

  if inner_steps != 1:
    step_fn = model_builder.repeated(step_fn, inner_steps)

  def step(carry_in, _):
    carry_out = step_fn(carry_in)
    frame = carry_in if start_with_input else carry_out
    return carry_out, post_process_fn(frame)

  def multistep(x):
    result = hk.scan(step, x, xs=None, length=outer_steps)
    return result

  return multistep


def my_decoded_trajectory_with_inputs(model, num_init_frames):
  """Returns trajectory_fn operating on decoded data.
  The returned function uses `num_init_frames` of the physics space trajectory
  provided as an input to initialize the model state, unrolls the trajectory of
  specified length that is decoded to the physics space using `model.decode_fn`.
  Args:
    model: model of a dynamical system used to obtain the trajectory.
    num_init_frames: number of time frames used from the physics trajectory to
      initialize the model state.
  Returns:
    Trajectory function that operates on physics space trajectories and returns
    unrolls in physics space.
  """
  def _trajectory_fn(x, outer_steps, inner_steps=1, is_training=False):
    trajectory_fn = functools.partial(
        model.trajectory, post_process_fn=model.decode)
    # add preprocessing to convert data to model state.
    trajectory_fn = model_utils.with_preprocessing(trajectory_fn, model.encode)
    # concatenate input trajectory to output trajectory for easier comparison.
    trajectory_fn = my_with_input_included(trajectory_fn)
    # make trajectories operate on full examples by splitting the init.
    trajectory_fn = model_utils.with_split_input(trajectory_fn, num_init_frames)
    return trajectory_fn(x, outer_steps, inner_steps, is_training=is_training)

  return _trajectory_fn


def my_with_input_included(trajectory_fn, time_axis=0):
  """Returns a `trajectory_fn` that concatenates inputs `x` to trajectory."""
  @functools.wraps(trajectory_fn)
  def _trajectory(x, *args, **kwargs):
    final, unroll = trajectory_fn(x, *args, **kwargs)
    # print(final[0].shape, x[0].shape, unroll[0].shape)
    return final, array_utils.concat_along_axis([x, unroll], time_axis)
  return _trajectory
