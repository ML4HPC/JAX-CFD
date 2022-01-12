import os.path

import xarray
import seaborn
import numpy as np
import pandas as pd
import jax_cfd.data.xarray_utils as xru
from jax_cfd.data import evaluation
import matplotlib.pyplot as plt


if __name__ == "__main__":
  data_path = "/global/cfs/cdirs/m3898/zhiqings/cfd"

  baseline_palette = seaborn.color_palette('YlGnBu', n_colors=7)[1:]
  original_palette = seaborn.color_palette('PRGn', n_colors=7)[:1]
  models_color = seaborn.color_palette('YlOrRd', n_colors=7)[1:][::-1]
  palette = baseline_palette + models_color

  filenames = {
      f'baseline_{r}': f'my_dns_{r}'
      for r in [64, 128, 256, 512, 1024, 2048]
  }
  # filenames['learned_interp_64'] = 'learned_interpolation_long_eval_64x64_64x64.nc'

  models = {}
  for k, v in filenames.items():
    models[k] = xarray.open_dataset(os.path.join(data_path, f'models/{v}/predict.nc'), chunks={'time': '100MB'})
    models[k].attrs['ndim'] = 2
    print(k)
    print(models[k]['u'].shape)
    # print(models[k]['u'].as_numpy())

  new_filenames = {
      'learned_interp_64 (ours)': 'learned_64_orig_ppp',
      'learned_interp_64 (ours, prefix=4)': 'learned_64_orig_pre4_fc_short',
      'learned_interp_64 (ours, prefix=4, dropout)': 'learned_64_orig_pre4_fc_short_dp',
  }
  for k, v in new_filenames.items():
    models[k] = xarray.open_dataset(os.path.join(data_path, f'models/{v}/my_predict.nc'), chunks={'time': '100MB'})
    print(k)
    print(models[k]['u'].shape)

  # combined = xarray.concat(list(models.values()), dim='model')
  # combined.coords['model'] = list(models.keys())
  # combined['vorticity'] = xru.vorticity_2d(combined)
  #
  # def resize_64_to_32(ds):
  #   coarse = xarray.Dataset({
  #       'u': ds.u.isel(x=slice(1, None, 2)).coarsen(y=2, coord_func='max').mean(),
  #       'v': ds.v.isel(y=slice(1, None, 2)).coarsen(x=2, coord_func='max').mean(),
  #   })
  #   coarse.attrs = ds.attrs
  #   return coarse
  #
  # combined_32 = resize_64_to_32(combined)
  # combined_32['vorticity'] = xru.vorticity_2d(combined_32)
  #
  # models_32 = {k: resize_64_to_32(v) for k, v in models.items()}
  #
  # combined.vorticity.isel(sample=0).thin(time=50).head(time=5).plot.imshow(
  #     row='model', col='time', x='x', y='y', robust=True, size=2.3, aspect=0.9,
  #     add_colorbar=False, cmap=seaborn.cm.icefire, vmin=-10, vmax=10)
  #
  # plt.savefig("new_figs/fig2.png")

  summary = xarray.concat([
      evaluation.compute_summary_dataset(ds, models['baseline_2048'])
      for ds in models.values()
  ], dim='model')
  summary.coords['model'] = list(models.keys())

  correlation = summary.vorticity_correlation.sel(time=slice(25)).compute()
  plt.figure(figsize=(7, 6))
  for color, model in zip(palette, summary['model'].data):
    print(model)
    style = '-' if 'baseline' in model else '--'
    if model in new_filenames:
      correlation.sel(model=model).shift(time=-32).plot.line(
          color=color, linestyle=style, label=model, linewidth=3)
    else:
      correlation.sel(model=model).plot.line(
          color=color, linestyle=style, label=model, linewidth=3)
  plt.axhline(y=0.95, xmin=0, xmax=20, color='gray')
  plt.legend()
  plt.title('')
  plt.xlim(0, 20)
  plt.savefig("new_figs/fig4.png")

  # spectrum = summary.energy_spectrum_mean.tail(time=2000).mean('time').compute()
  # plt.figure(figsize=(10, 6))
  # for color, model in zip(palette, summary['model'].data):
  #   style = '-' if 'baseline' in model else '--'
  #   (spectrum.k ** 5 * spectrum).sel(model=model).plot.line(
  #       color=color, linestyle=style, label=model, linewidth=3)
  # plt.legend()
  # plt.yscale('log')
  # plt.xscale('log')
  # plt.title('')
  # plt.xlim(3.5, None)
  # plt.ylim(1e9, None)
  # plt.savefig("new_figs/fig4.png")
