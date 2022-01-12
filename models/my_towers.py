"""Definitions of towers (neural networks based on multioke CNN layers)."""

import functools
from typing import Any, Callable, List, Optional, Tuple, Union
import gin
import jax
import jax.numpy as jnp
import haiku as hk

from jax_cfd.ml import layers
from jax_cfd.ml import nonlinearities
from jax_cfd.ml import towers

Array = layers.Array
ConvModule = Callable[..., Any]
ScaleFn = Callable[[Array, List[int]], Array]
TowerFactory = Callable[..., Any]


PERIODIC_CONV_MODULES = {
    1: layers.PeriodicConv1D,
    2: layers.PeriodicConv2D,
    3: layers.PeriodicConv3D}

PERIODIC_CONV_TRANSPOSE_MODULES = {
    1: layers.PeriodicConvTranspose1D,
    2: layers.PeriodicConvTranspose2D,
    3: layers.PeriodicConvTranspose3D}


@gin.configurable
def my_forward_tower_factory(
    num_output_channels: int,
    ndim: int,
    num_hidden_channels: int = 16,
    kernel_size: int = 3,
    num_hidden_layers: int = 2,
    rates: Union[int, Tuple[int, ...]] = 1,
    strides: Union[int, Tuple[int, ...]] = 1,
    output_kernel_size: int = 3,
    output_dilation_rate: int = 1,
    output_stride: int = 1,
    conv_module: ConvModule = towers.periodic_convolution,
    nonlinearity: Callable[[Array], Array] = nonlinearities.relu,
    inputs_scale_fn: ScaleFn = lambda x, axes: x,
    output_scale_fn: ScaleFn = lambda x, axes: x,
    dropout_rate: float = 0.0,
    num_transformer_layers: int = 0,
    name: Optional[str] = 'forward_cnn_tower',
):
  """Constructs parametrized feed-forward CNN tower.
  Constructs CNN tower parametrized by fixed number of channels in hidden layers
  and fixed square kernels.
  Args:
    num_output_channels: number of channels in the output layer.
    ndim: number of spatial dimensions to expect in inputs to the network.
    num_hidden_channels: number of channels to use in hidden layers.
    kernel_size: size of the kernel to use along every dimension.
    num_hidden_layers: number of hidden layers to construct in the tower.
    rates: dilation rate(s) of the hidden layers.
    strides: strides to use. Must be `int` or same a `num_hidden_layers`.
    output_kernel_size: size of the output kernel to use along every dimension.
    output_dilation_rate: dilation_rate of the output layer.
    output_stride: stride of the final convolution.
    conv_module: convolution module to use. Must accept
      (output channels, kernel shape and ndim).
    nonlinearity: nonlinearity function to apply between hidden layers.
    inputs_scale_fn: scaling function to be applied to the inputs of the tower.
      Must take inputs as argument and return an `Array` of the same shape.
      Can expect an `axes` arguments specifying spatial axes in inputs.
    output_scale_fn: similar to `inputs_scale_fn` but applied to outputs.
    dropout_rate: dropout rate to apply between hidden layers.
    num_transformer_layers: number of transformer layers to use.
    name: a name for this CNN tower. This name will appear in Xprof traces.
  Returns:
    CNN tower with specified configuration.
  """
  channels = (num_hidden_channels,) * num_hidden_layers
  kernel_shapes = ((kernel_size,) * ndim,) * num_hidden_layers
  output_kernel_shape = (output_kernel_size,) * ndim
  return my_forward_flex_tower_factory(
      num_output_channels=num_output_channels, ndim=ndim, channels=channels,
      kernel_shapes=kernel_shapes, rates=rates, strides=strides,
      output_kernel_shape=output_kernel_shape, output_rate=output_dilation_rate,
      output_stride=output_stride, conv_module=conv_module,
      nonlinearity=nonlinearity, inputs_scale_fn=inputs_scale_fn,
      output_scale_fn=output_scale_fn, dropout_rate=dropout_rate,
      num_transformer_layers=num_transformer_layers, name=name)


@gin.configurable
def my_forward_flex_tower_factory(
    num_output_channels: int,
    ndim: int,
    channels: Tuple[int, ...] = (16, 16),
    kernel_shapes: Tuple[Tuple[int, ...], ...] = ((3, 3), (3, 3)),
    rates: Tuple[int, ...] = (1, 1),
    strides: Tuple[int, ...] = (1, 1),
    output_kernel_shape: Tuple[int, ...] = (3, 3),
    output_rate: int = 1,
    output_stride: int = 1,
    conv_module: ConvModule = towers.periodic_convolution,
    nonlinearity: Callable[[Array], Array] = nonlinearities.relu,
    inputs_scale_fn: ScaleFn = lambda x, axes: x,
    output_scale_fn: ScaleFn = lambda x, axes: x,
    dropout_rate: float = 0.0,
    num_transformer_layers: int = 0,
    name: Optional[str] = 'forward_flex_cnn_tower',
):
  """Constructs CNN tower with specified architecture.
  Args:
    num_output_channels: number of channels in the output layer.
    ndim: number of spatial dimensions to expect in inputs to the network.
    channels: tuple specifying number of channels in hidden layers.
    kernel_shapes: tuple specifying shapes of kernels in hidden layers.
      Each entry must be a tuple that specifies a valid kernel_shape for the
      provided `conv_module`. Must have the same length as `channels`.
    rates: dilation rates of the convolutions.
    strides: strides to use in convolutions.
    output_kernel_shape: shape of the output kernel.
    output_rate: dilation rate of the final convolution.
    output_stride: stride of the final convolution.
    conv_module: convolution module to use. Must accept
      (output channels, kernel shape and ndim).
    nonlinearity: nonlinearity function to apply between hidden layers.
    inputs_scale_fn: scaling function to be applied to the inputs of the tower.
      Must take `inputs`, `axes` arguments specifying input `Array` and
      spatial dimensions and return an `Array` of the same shape as `inputs`.
    output_scale_fn: similar to `inputs_scale_fn` but applied to outputs.
    dropout_rate: dropout rate to use in the network.
    num_transformer_layers: number of transformer layers to use.
    name: a name for this CNN tower. This name will appear in Xprof traces.
  Returns:
    CNN tower with specified architecture.
  """
  if isinstance(strides, int):
    strides = (strides,) * len(channels)
  if isinstance(rates, int):
    rates = (rates,) * len(channels)

  ndim_axes = list(range(ndim))
  n_convs = len(channels)
  if not all(len(arg) == n_convs for arg in [kernel_shapes, rates, strides]):
    raise ValueError('conflicting lengths for channels/kernels/rates/strides: '
                     f'{channels} / {kernel_shapes} / {rates} / {strides}')

  def forward_pass(inputs, is_training=False):
    components = [functools.partial(inputs_scale_fn, axes=ndim_axes)]
    num_cnn_layers = len(channels) - num_transformer_layers
    conv_args = zip(channels[:num_cnn_layers], kernel_shapes[:num_cnn_layers],
                    rates[:num_cnn_layers], strides[:num_cnn_layers])
    for num_channels, kernel_shape, rate, stride in conv_args:
      components.append(conv_module(num_channels, kernel_shape, ndim, rate=rate,
                                    stride=stride))
      components.append(nonlinearity)
    components.append(functools.partial(
        dropout_layer, dropout_rate=dropout_rate, is_training=is_training))

    if num_transformer_layers > 0:
      components.append(conv_module(channels[-1], kernel_shapes[-1], ndim, rate=rates[-1], stride=strides[-1]))
      components.append(functools.partial(
          Transformer(num_transformer_layers, dropout_rate=dropout_rate, name='transformer'),
          is_training=is_training))

    components.append(conv_module(num_output_channels, output_kernel_shape,
                                  ndim, rate=output_rate, stride=output_stride))
    components.append(functools.partial(output_scale_fn, axes=ndim_axes))

    return hk.Sequential(components)(inputs)

  module = hk.to_module(forward_pass)(name=name)
  return hk.experimental.named_call(module, name=name)


def dropout_layer(inputs, dropout_rate, is_training):
  if is_training:
    return hk.dropout(hk.next_rng_key(), dropout_rate, inputs)
  else:
    return inputs


@gin.configurable
def my_residual_block_tower_factory(
    num_output_channels: int,
    ndim: int,
    num_blocks: int = 2,
    block_factory: TowerFactory = my_forward_tower_factory,
    skip_connection_fn: Callable[..., Array] = lambda x, block_num: x,
    inputs_scale_fn: ScaleFn = lambda x, axes: x,
    output_scale_fn: ScaleFn = lambda x, axes: x,
    name: Optional[str] = 'residual_block_tower',
):
  """Constructs a tower with skip connections between blocks."""
  def forward_pass(inputs, is_training=False):
    inputs = inputs_scale_fn(inputs, list(range(ndim)))
    for block_num in range(num_blocks - 1):
      skip = skip_connection_fn(inputs, block_num)
      block = block_factory(skip.shape[-1], ndim)
      inputs = skip + block(inputs)
    last_block = block_factory(num_output_channels, ndim)
    return output_scale_fn(last_block(inputs), list(range(ndim)))

  module = hk.to_module(forward_pass)(name=name)
  return hk.experimental.named_call(module, name=name)


@gin.configurable
class Transformer(hk.Module):
  """A transformer stack."""

  def __init__(self,
               num_layers: int,
               num_heads: int = 4,
               dropout_rate: float = 0.1,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._num_layers = num_layers
    self._num_heads = num_heads
    self._dropout_rate = dropout_rate

  def __call__(self,
               h: jnp.ndarray,
               is_training: bool) -> jnp.ndarray:
    """Connects the transformer.
    Args:
      h: Inputs, [B, T, D].
      mask: Padding mask, [B, T].
      is_training: Whether we're training or not.
    Returns:
      Array of shape [B, T, D].
    """

    d1, d2, d3 = h.shape
    h = jnp.reshape(h, [1, d1 * d2, d3])

    init_scale = 2. / self._num_layers
    dropout_rate = self._dropout_rate if is_training else 0.
    for i in range(self._num_layers):
      h_norm = layer_norm(h, name=f'h{i}_ln_1')
      h_attn = hk.MultiHeadAttention(
          num_heads=self._num_heads,
          key_size=32,
          w_init_scale=init_scale,
          name=f'h{i}_attn')(h_norm, h_norm, h_norm)
      h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
      h = h + h_attn
      h_norm = layer_norm(h, name=f'h{i}_ln_2')
      h_dense = DenseBlock(init_scale, name=f'h{i}_mlp')(h_norm)
      h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
      h = h + h_dense
    h = layer_norm(h, name='ln_f')
    return jnp.reshape(h, [d1, d2, d3])


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
  """Apply a unique LayerNorm to x with default settings."""
  return hk.LayerNorm(axis=-1,
                      create_scale=True,
                      create_offset=True,
                      name=name)(x)


@gin.configurable
class DenseBlock(hk.Module):
  """A 2-layer MLP which widens then narrows the input."""

  def __init__(self,
               init_scale: float,
               widening_factor: int = 4,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._init_scale = init_scale
    self._widening_factor = widening_factor

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    hiddens = x.shape[-1]
    initializer = hk.initializers.VarianceScaling(self._init_scale)
    x = hk.Linear(self._widening_factor * hiddens, w_init=initializer)(x)
    x = jax.nn.gelu(x)
    return hk.Linear(hiddens, w_init=initializer)(x)


@gin.configurable
def my_fixed_scale(
    inputs: Array,
    axes: Tuple[int, ...],
    rescaled_one: float = gin.REQUIRED
) ->Array:
  """Linearly scales `inputs` such that `1` maps to `rescaled_one`."""
  del axes  # unused.
  return inputs * rescaled_one
