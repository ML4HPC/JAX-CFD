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
"""ImageNet dataset with typical pre-processing."""

import enum
import logging

import xarray
import itertools as it
import types
from typing import Generator, Iterable, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from packaging import version
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

import torch
import torch.utils.data

Batch = Mapping[str, np.ndarray]


def _check_min_version(mod: types.ModuleType, min_ver: str):
  actual_ver = getattr(mod, '__version__')
  if version.parse(actual_ver) < version.parse(min_ver):
    raise ValueError(
        f'{mod.__name__} >= {min_ver} is required, you have {actual_ver}')


def check_versions():
  _check_min_version(tf, '2.5.0')
  _check_min_version(tfds, '4.2.0')


class TimeseriesDataset(torch.utils.data.Dataset):
  def __init__(self, data, seq_len, num_sample, encode_steps, decode_steps, from_, to_):
    self.data = data
    self.seq_len = seq_len
    self.num_sample = num_sample
    self.from_ = from_
    self.to_ = to_
    self.encode_steps = encode_steps
    self.decode_steps = decode_steps

  def __len__(self):
    return self.to_ - self.from_

  def __getitem__(self, full_idx):
    full_idx = full_idx + self.from_
    sample_id = full_idx % self.num_sample
    seq_id = full_idx // self.num_sample

    beg_idx = seq_id
    end_idx = beg_idx + self.encode_steps + self.decode_steps

    sliced_input = self.data[dict(sample=sample_id, time=slice(beg_idx, end_idx))]
    input_u = torch.from_numpy(sliced_input.variables['u'].values)
    input_v = torch.from_numpy(sliced_input.variables['v'].values)

    return input_u, input_v


class DataLoader(object):
  def __init__(self,
               split: str,
               *,
               is_training: bool,
               batch_dims: Sequence[int],
               input_size: int,
               encode_steps: int,
               decode_steps: int,
               delta_time: float,
               dtype: jnp.dtype = jnp.float32,
               transpose: bool = False,
               zeros: bool = False,
               num_workers: int = 32,
               ):
    self.split = split
    self.zeros = zeros
    self.input_size = input_size
    self.encode_steps = encode_steps
    self.transpose = transpose
    self.is_training = is_training
    self.batch_dims = batch_dims
    self.decode_steps = decode_steps
    self.dtype = dtype
    self.num_workers = num_workers

    if not self.zeros:
      self.data = xarray.open_dataset(split)
      logging.info(str(self.data))
      for att in self.data.attrs:
        logging.info(f'{att}: {self.data.attrs[att]}')
      self.delta_time = self.data.coords["time"].values[1] - self.data.coords["time"].values[0]
      self.inner_steps = round(self.delta_time / delta_time)
      self.seq_len = self.data.sizes['time']
      self.num_sample = self.data.sizes['sample']
      self.num_examples = self.num_sample * (self.seq_len - encode_steps - decode_steps + 1)

  def load(self) -> Generator[Batch, None, None]:
    """Loads the given split of the dataset."""
    if self.zeros:
      h, w, c = self.input_size, self.input_size, self.encode_steps
      if self.transpose:
        input_dims = (*self.batch_dims[:-1], h, w, c, self.batch_dims[0])
      else:
        input_dims = (*self.batch_dims, h, w, c)

      h, w, c = self.input_size, self.input_size, self.decode_steps
      if self.transpose:
        ouput_dims = (*self.batch_dims[:-1], h, w, c, self.batch_dims[0])
      else:
        ouput_dims = (*self.batch_dims, h, w, c)

      batch = {'images': np.zeros(input_dims, dtype=self.dtype),
               'labels': np.zeros(ouput_dims, dtype=self.dtype)}
      if self.is_training:
        yield from it.repeat(batch)
      else:
        num_batches = 1000 // np.prod(self.batch_dims)
        yield from it.repeat(batch, num_batches)

    start, end, max_len = _shard(self.num_examples, jax.host_id(), jax.host_count())
    logging.info(f'Start: {start}, End:{end}')

    total_batch_size = np.prod(self.batch_dims)

    dataset = TimeseriesDataset(self.data, self.seq_len, self.num_sample,
                                self.encode_steps, self.decode_steps, from_=start, to_=end)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=total_batch_size.item(), shuffle=self.is_training,
        num_workers=self.num_workers, drop_last=self.is_training)

    if self.is_training:
      while True:
        for iu, iv in dataloader:
          iu, iv = iu.numpy(), iv.numpy()
          iu = iu.reshape(list(self.batch_dims) + list(iu.shape[1:]))
          iv = iv.reshape(list(self.batch_dims) + list(iv.shape[1:]))
          inputs = (iu, iv)
          batch = {"inputs": inputs}
          yield batch
    else:
      remaining_items = max_len
      expected_shape = None
      for iu, iv in dataloader:
        iu, iv = iu.numpy(), iv.numpy()

        expected_shape = list(self.batch_dims) + list(iu.shape[1:])
        if np.prod(expected_shape) != np.prod(iu.shape):
          remaining_batches = (np.prod(expected_shape) - np.prod(iu.shape)) // np.prod(iu.shape[1:])
          iu = np.pad(iu.reshape([-1] + list(iu.shape[1:])), [(0, remaining_batches)] + [(0, 0)] * len(iu.shape[1:]))
          iv = np.pad(iv.reshape([-1] + list(iv.shape[1:])), [(0, remaining_batches)] + [(0, 0)] * len(iv.shape[1:]))
        else:
          remaining_batches = 0

        remaining_items -= np.prod(self.batch_dims)
        iu = iu.reshape(list(self.batch_dims) + list(iu.shape[1:]))
        iv = iv.reshape(list(self.batch_dims) + list(iv.shape[1:]))
        inputs = (iu, iv)

        if remaining_batches == 0:
          weights = np.ones(np.prod(self.batch_dims))
        else:
          weights = np.ones(np.prod(self.batch_dims) - remaining_batches)
          weights = np.pad(weights, [(0, remaining_batches)], mode='constant')
        weights = weights.reshape(list(self.batch_dims))

        batch = {"inputs": inputs, 'weights': weights}
        yield batch

      while remaining_items > 0:
        iu = np.zeros(expected_shape)
        iv = np.zeros(expected_shape)
        inputs = (iu, iv)
        weights = np.zeros(list(self.batch_dims))
        batch = {"inputs": inputs, 'weights': weights}
        remaining_items -= np.prod(self.batch_dims)
        yield batch

    # if self.is_training:
    #   shuffle_idx = np.arange(start, end)
    #   while True:
    #     np.random.shuffle(shuffle_idx)
    #     for k in range(0, self.num_examples // total_batch_size):
    #       inputs = []
    #       outputs = []
    #       for j in range(total_batch_size):
    #         full_idx = shuffle_idx[j]
    #
    #         outputs.append(np.stack([u, v], axis=-1))
    #       inputs = np.stack(inputs, axis=0)
    #       outputs = np.stack(outputs, axis=0)
    #       inputs = inputs.reshape(list(self.batch_dims) + list(inputs.shape[1:]))
    #       outputs = outputs.reshape(list(self.batch_dims) + list(outputs.shape[1:]))
    #
    #       inputs = (inputs[..., 0], inputs[..., 1])
    #       outputs = (outputs[..., 0], outputs[..., 1])
    #
    #       batch = {"inputs": inputs, "outputs": outputs}
    #       yield batch
    # else:
    #   if self.num_examples % total_batch_size != 0:
    #     raise ValueError(f'Test/valid must be divisible by {total_batch_size}')
    #   shuffle_idx = np.arange(start, end)
    #   for i in range(0, self.num_examples // total_batch_size):
    #     inputs = []
    #     outputs = []
    #     for j in range(total_batch_size):
    #       full_idx = shuffle_idx[j]
    #       sample_id = full_idx % self.num_sample
    #       seq_id = full_idx // self.num_sample
    #
    #       beg_idx = seq_id
    #       mid_idx = beg_idx + self.encode_steps
    #       end_idx = mid_idx + self.decode_steps
    #       sliced_input = self.data[dict(sample=sample_id, time=slice(beg_idx, mid_idx))]
    #       u = sliced_input.variables['u'].values
    #       v = sliced_input.variables['v'].values
    #       inputs.append(np.stack([u, v], axis=-1))
    #       sliced_ouput = self.data[dict(sample=sample_id, time=slice(mid_idx, end_idx))]
    #       u = sliced_ouput.variables['u'].values
    #       v = sliced_ouput.variables['v'].values
    #       outputs.append(np.stack([u, v], axis=-1))
    #
    #     inputs = np.stack(inputs, axis=0)
    #     outputs = np.stack(outputs, axis=0)
    #     inputs = inputs.reshape(list(self.batch_dims) + list(inputs.shape[1:]))
    #     outputs = outputs.reshape(list(self.batch_dims) + list(outputs.shape[1:]))
    #
    #     inputs = (inputs[..., 0], inputs[..., 1])
    #     outputs = (outputs[..., 0], outputs[..., 1])
    #     batch = {"inputs": inputs, "outputs": outputs}
    #     yield batch

    # def cast_fn(batch):
    #   batch = dict(**batch)
    #   batch['images'] = tf.cast(batch['images'], tf.dtypes.as_dtype(dtype))
    #   return batch


def _device_put_sharded(sharded_tree, devices):
  leaves, treedef = jax.tree_flatten(sharded_tree)
  n = leaves[0].shape[0]
  return jax.device_put_sharded(
      [jax.tree_unflatten(treedef, [l[i] for l in leaves]) for i in range(n)],
      devices)


def double_buffer(ds: Iterable[Batch]) -> Generator[Batch, None, None]:
  """Keeps at least two batches on the accelerator.
  The current GPU allocator design reuses previous allocations. For a training
  loop this means batches will (typically) occupy the same region of memory as
  the previous batch. An issue with this is that it means we cannot overlap a
  host->device copy for the next batch until the previous step has finished and
  the previous batch has been freed.
  By double buffering we ensure that there are always two batches on the device.
  This means that a given batch waits on the N-2'th step to finish and free,
  meaning that it can allocate and copy the next batch to the accelerator in
  parallel with the N-1'th step being executed.
  Args:
    ds: Iterable of batches of numpy arrays.
  Yields:
    Batches of sharded device arrays.
  """
  batch = None
  devices = jax.local_devices()
  logging.info(devices)
  for next_batch in ds:
    assert next_batch is not None
    next_batch = _device_put_sharded(next_batch, devices)
    if batch is not None:
      yield batch
    batch = next_batch
  if batch is not None:
    yield batch


def _shard(num_examples: int, shard_index: int, num_shards: int) -> Tuple[int, int, int]:
  """Returns [start, end) for the given shard index."""
  assert shard_index < num_shards
  arange = np.arange(num_examples)
  shard_range = np.array_split(arange, num_shards)
  max_len = shard_range[0][-1] + 1 - shard_range[0][0]
  shard_range = shard_range[shard_index]
  start, end = shard_range[0], (shard_range[-1] + 1)
  return start, end, max_len
