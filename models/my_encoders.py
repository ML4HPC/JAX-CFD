"""Encoder modules that help interfacing input trajectories to model states.
All encoder modules generate a function that given an input trajectory infers
the final state of the physical system in the representation defined by the
Encoder. Encoders can be either fixed functions, decorators or learned modules.
The input state is expected to consist of arrays with `time` as a leading axis.
"""

from typing import Any, Callable, Optional, Tuple
import gin
import jax
import jax.numpy as jnp
from jax_cfd.base import array_utils
from jax_cfd.base import boundaries
from jax_cfd.base import grids
from jax_cfd.ml import physics_specifications
from jax_cfd.ml import towers


EncodeFn = Callable[[Any], Any]  # maps input trajectory to final model state.
EncoderModule = Callable[..., EncodeFn]  # generate EncodeFn closed over args.
TowerFactory = towers.TowerFactory


@gin.configurable
def my_aligned_array_encoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    n_frames: int = 1,
    data_offsets: Optional[Tuple[Tuple[float, ...], ...]] = None,
) -> EncodeFn:
  """Generates encoder that wraps last data slice as GridVariables."""
  del dt, physics_specs  # unused.
  data_offsets = data_offsets or grid.cell_faces
  slice_last_fn = lambda x: array_utils.slice_along_axis(x, 0, slice(-n_frames, None))

  # TODO(pnorgaard) Make the encoder/decoder/network configurable for BC
  def encode_fn(inputs):
    bc = boundaries.periodic_boundary_conditions(grid.ndim)
    return tuple(
        grids.GridVariable(grids.GridArray(slice_last_fn(x), offset, grid), bc)
        for x, offset in zip(inputs, data_offsets))

  return encode_fn
