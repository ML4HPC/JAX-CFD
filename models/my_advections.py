"""Models for advection and convection components."""
import functools

from typing import Callable, Optional
import gin
from jax_cfd.base import advection
from jax_cfd.base import interpolation
from jax_cfd.base import grids
from jax_cfd.ml import interpolations
from jax_cfd.ml import physics_specifications


GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
InterpolationModule = interpolations.InterpolationModule
InterpolationFn = interpolation.InterpolationFn
InterpolationTransform = Callable[..., InterpolationFn]
AdvectFn = Callable[[GridVariable, GridVariableVector, float], GridArray]
AdvectionModule = Callable[..., AdvectFn]
ConvectFn = Callable[[GridVariableVector], GridArrayVector]
ConvectionModule = Callable[..., ConvectFn]


@gin.configurable
def my_modular_self_advection(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.NavierStokesPhysicsSpecs,
    interpolation_module: InterpolationModule,
    transformation: InterpolationTransform = interpolations.tvd_limiter_transformation,
    **kwargs
) -> AdvectFn:
  """Modular self advection using a single interpolation module."""
  # TODO(jamieas): Replace this entire function once
  # `single_tower_navier_stokes` is in place.
  interpolate_fn = interpolation_module(grid, dt, physics_specs, **kwargs)
  c_interpolate_fn = functools.partial(interpolate_fn, tag='c')
  c_interpolate_fn = transformation(c_interpolate_fn)
  u_interpolate_fn = functools.partial(interpolate_fn, tag='u')

  def advect(
      c: GridVariable,
      v: GridVariableVector,
      dt: Optional[float] = None
  ) -> GridArray:
    return advection.advect_general(
        c, v, u_interpolate_fn, c_interpolate_fn, dt)

  return advect
