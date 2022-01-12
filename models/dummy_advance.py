from typing import Any, Callable, Tuple

import gin
import haiku as hk
from jax_cfd.base import grids
from jax_cfd.ml import physics_specifications


@gin.configurable
def dummy_model(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.NavierStokesPhysicsSpecs,
):
  del grid, dt, physics_specs

  def dummy_step_fn(state):
    """Advances Navier-Stokes state forward in time."""
    return state

  return hk.to_module(dummy_step_fn)()
