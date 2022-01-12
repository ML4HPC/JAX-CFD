"""Interpolation modules."""

import collections
import functools
import logging
from typing import (
    Any, Callable, Dict, Optional, Sequence, Tuple, TypeVar, Union,
)

import gin
import jax
import jax.numpy as jnp
from jax_cfd.base import grids
from jax_cfd.base import interpolation
from jax_cfd.ml import layers
from jax_cfd.ml import physics_specifications
from jax_cfd.ml import towers
from jax_cfd.ml import layers_util
from jax_cfd.ml import tiling
import numpy as np
import scipy


GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
InterpolationFn = interpolation.InterpolationFn
InterpolationModule = Callable[..., InterpolationFn]
InterpolationTransform = Callable[..., InterpolationFn]
FluxLimiter = interpolation.FluxLimiter


StencilSizeFn = Callable[
    [Tuple[int, ...], Tuple[int, ...], Any], Tuple[int, ...]]


@gin.configurable
class MyFusedLearnedInterpolation:
  """Learned interpolator that computes interpolation coefficients in 1 pass.
  Interpolation function that has pre-computed interpolation
  coefficients for a given velocity field `v`. It uses a collection of
  `SpatialDerivativeFromLogits` modules and a single neural network that
  produces logits for all expected interpolations. Interpolations are keyed by
  `input_offset`, `target_offset` and an optional `tag`. The `tag` allows us to
  perform multiple interpolations between the same `offset` and `target_offset`
  with different weights.
  """

  def __init__(
      self,
      grid: grids.Grid,
      dt: float,
      physics_specs: physics_specifications.BasePhysicsSpecs,
      v,
      tags=(None,),
      stencil_size: Union[int, StencilSizeFn] = 4,
      tower_factory=towers.forward_tower_factory,
      name='fused_learned_interpolation',
      extract_patch_method='roll',
      fuse_constraints=False,
      fuse_patches=False,
      constrain_with_conv=False,
      tile_layout=None,
      is_training=False,
      pattern='simple',
  ):
    """Constructs object and performs necessary pre-computate."""
    del dt, physics_specs  # unused.

    derivative_orders = (0,) * grid.ndim
    derivatives = collections.OrderedDict()

    if isinstance(stencil_size, int):
      stencil_size_fn = lambda *_: (stencil_size,) * grid.ndim
    else:
      stencil_size_fn = stencil_size

    if pattern == 'simple' or pattern == 'softmax':
      for u in v:
        for target_offset in grids.control_volume_offsets(u):
          for tag in tags:
            key = (u.offset, target_offset, tag)
            derivatives[key] = MySpatialDerivativeFromLogits(
                stencil_size_fn(*key),
                u.offset,
                target_offset,
                derivative_orders=derivative_orders,
                steps=grid.step,
                extract_patch_method=extract_patch_method,
                tile_layout=tile_layout,
                pattern=pattern,
            )
    elif pattern == 'original':
      for u in v:
        for target_offset in grids.control_volume_offsets(u):
          for tag in tags:
            key = (u.offset, target_offset, tag)
            derivatives[key] = layers.SpatialDerivativeFromLogits(
                stencil_size_fn(*key),
                u.offset,
                target_offset,
                derivative_orders=derivative_orders,
                steps=grid.step,
                extract_patch_method=extract_patch_method,
                tile_layout=tile_layout,
            )

    if pattern == 'simple':
      output_sizes = [deriv.subspace_size + 1 for deriv in derivatives.values()]
    elif pattern == 'softmax':
      output_sizes = [deriv.subspace_size + 2 for deriv in derivatives.values()]
    elif pattern == 'original':
      output_sizes = [deriv.subspace_size for deriv in derivatives.values()]
    else:
      raise ValueError('Unknown pattern: {}'.format(pattern))
    cnn_network = tower_factory(sum(output_sizes), grid.ndim, name=name)
    inputs = jnp.stack([u.data for u in v], axis=-1)
    inputs = jnp.transpose(inputs, axes=(1, 2, 3, 0))
    inputs = jnp.reshape(inputs, inputs.shape[:2] + (-1,))

    logging.info('inputs shape: %s', inputs.shape)
    all_logits = cnn_network(inputs, is_training=is_training)

    logging.info(inputs.shape)
    logging.info(output_sizes)

    if fuse_constraints:
      raise NotImplementedError
    else:
      split_logits = jnp.split(all_logits, np.cumsum(output_sizes), axis=-1)
      self._interpolators = {
          k: functools.partial(derivative, logits=logits)
          for (k, derivative), logits in zip(derivatives.items(), split_logits)
      }

  def __call__(self,
               c: GridVariable,
               offset: Tuple[int, ...],
               v: GridVariableVector,
               dt: float,
               tag=None) -> GridVariable:
    del dt  # not used.
    # TODO(dkochkov) Add decorator to expand/squeeze channel dim.
    c = grids.GridVariable(
        grids.GridArray(jnp.expand_dims(c.data, -1), c.offset, c.grid), c.bc)
    # TODO(jamieas): Try removing the following line.
    if c.offset == offset:
      return c
    key = (c.offset, offset, tag)
    interpolator = self._interpolators.get(key)
    if interpolator is None:
      raise KeyError(f'No interpolator for key {key}. '
                     f'Available keys: {list(self._interpolators.keys())}')
    result = jnp.squeeze(interpolator(c.data), axis=-1)
    return grids.GridVariable(
        grids.GridArray(result, offset, c.grid), c.bc)


class MySpatialDerivativeFromLogits:
  """Module that transforms logits to polynomially accurate derivatives.
  Applies `PolynomialConstraint` layer to input logits and combines the
  resulting coefficients with basis. Compared to `SpatialDerivative`, this
  module does not compute `logits`, but takes them as an argument.
  """

  def __init__(
      self,
      stencil_shape: Tuple[int, ...],
      input_offset: Tuple[float, ...],
      target_offset: Tuple[float, ...],
      derivative_orders: Tuple[int, ...],
      steps: Tuple[float, ...],
      extract_patch_method: str = 'roll',
      tile_layout: Optional[Tuple[int, ...]] = None,
      pattern: str = 'simple',
      method: layers_util.Method = layers_util.Method.FINITE_VOLUME,
  ):
    self.stencil_shape = stencil_shape
    self.roll, shift = layers_util.get_roll_and_shift(
        input_offset, target_offset)
    stencils = layers_util.get_stencils(stencil_shape, shift, steps)
    self.constraint = layers.PolynomialConstraint(
        stencils, derivative_orders, method, steps)
    self._extract_patch_method = extract_patch_method
    self.tile_layout = tile_layout
    self.pattern = pattern

  @property
  def subspace_size(self) -> int:
    return self.constraint.subspace_size

  @property
  def stencil_size(self) -> int:
    return int(np.prod(self.stencil_shape))

  def extract_patches(self, inputs):
    rolled = jnp.roll(inputs, self.roll)
    patches = layers_util.extract_patches(
        rolled, self.stencil_shape,
        self._extract_patch_method, self.tile_layout)
    return patches

  @functools.partial(jax.named_call, name='SpatialDerivativeFromLogits')
  def __call__(self, inputs, logits):
    if self.pattern == 'simple':
      logits = logits * 0.1
      coefficients = logits - jnp.mean(logits, axis=-1, keepdims=True) + self.constraint.bias
    elif self.pattern == 'softmax':
      coefficients = jax.nn.softmax(logits[..., :-1], axis=-1)
    else:
      raise ValueError(f'Unknown pattern {self.pattern}')
    logging.info(logits.shape)
    patches = self.extract_patches(inputs)
    output = layers_util.apply_coefficients(coefficients, patches)
    if self.pattern == 'softmax':
      output = output + logits[..., -1:]
    return output


def _my_linear_along_axis(c: GridVariable,
                          offset: float,
                          axis: int) -> GridVariable:
  """Linear interpolation of `c` to `offset` along a single specified `axis`."""
  offset_delta = offset - c.offset[axis]

  # If offsets are the same, `c` is unchanged.
  if offset_delta == 0:
    return c

  new_offset = tuple(offset if j == axis else o
                     for j, o in enumerate(c.offset))

  # If offsets differ by an integer, we can just shift `c`.
  if int(offset_delta) == offset_delta:
    return grids.GridVariable(
        array=grids.GridArray(data=c.shift(int(offset_delta), axis).data,
                              offset=new_offset,
                              grid=c.grid),
        bc=c.bc)

  floor = int(np.floor(offset_delta))
  ceil = int(np.ceil(offset_delta))
  floor_weight = ceil - offset_delta
  ceil_weight = 1. - floor_weight
  data = (floor_weight * c.shift(floor, axis).data +
          ceil_weight * c.shift(ceil, axis).data)
  print(floor_weight, ceil_weight)
  return grids.GridVariable(
      array=grids.GridArray(data, new_offset, c.grid), bc=c.bc)


def _linear(
    c: GridVariable,
    offset: Tuple[float, ...],
    v: Optional[GridVariableVector] = None,
    dt: Optional[float] = None
) -> grids.GridVariable:
  """Multi-linear interpolation of `c` to `offset`.
  Args:
    c: quantitity to be interpolated.
    offset: offset to which we will interpolate `c`. Must have the same length
      as `c.offset`.
    v: velocity field. Not used.
    dt: size of the time step. Not used.
  Returns:
    An `GridArray` containing the values of `c` after linear interpolation
    to `offset`. The returned value will have offset equal to `offset`.
  """
  del v, dt  # unused
  if len(offset) != len(c.offset):
    raise ValueError('`c.offset` and `offset` must have the same length;'
                     f'got {c.offset} and {offset}.')
  interpolated = c
  for a, o in enumerate(offset):
    interpolated = _my_linear_along_axis(interpolated, offset=o, axis=a)
  return interpolated


@gin.configurable
def my_linear(*args, **kwargs):
  del args, kwargs
  return _linear
