import os.path

import xarray
import seaborn
import numpy as np
import pandas as pd
import jax_cfd.data.xarray_utils as xru
from jax_cfd.data import evaluation
import matplotlib.pyplot as plt


def correlation(x, y):
  state_dims = ['x', 'y']
  p = xru.normalize(x, state_dims) * xru.normalize(y, state_dims)
  return p.sum(state_dims)


def calculate_time_until(vorticity_corr):
  threshold = 0.95
  return (vorticity_corr.mean('sample') >= threshold).idxmin('time').rename('time_until')


def calculate_time_until_bootstrap(vorticity_corr, bootstrap_samples=10000):
  rs = np.random.RandomState(0)
  indices = rs.choice(16, size=(10000, 16), replace=True)
  boot_vorticity_corr = vorticity_corr.isel(
      sample=(('boot', 'sample2'), indices)).rename({'sample2': 'sample'})
  return calculate_time_until(boot_vorticity_corr)


def calculate_upscaling(time_until):
  slope = ((np.log(16) - np.log(8))
           / (time_until.sel(model='baseline_1024')
              - time_until.sel(model='baseline_512')))
  x = time_until.sel(model='learned_interp_64')
  x0 = time_until.sel(model='baseline_512')
  intercept = np.log(8)
  factor = np.exp(slope * (x - x0) + intercept)
  return factor


def calculate_speedup(time_until):
  runtime_baseline_8x = 44.053293
  runtime_baseline_16x = 412.725656
  runtime_learned = 1.155115
  slope = ((np.log(runtime_baseline_16x) - np.log(runtime_baseline_8x))
           / (time_until.sel(model='baseline_1024')
              - time_until.sel(model='baseline_512')))
  x = time_until.sel(model='learned_interp_64')
  x0 = time_until.sel(model='baseline_512')
  intercept = np.log(runtime_baseline_8x)
  speedups = np.exp(slope * (x - x0) + intercept) / runtime_learned
  return speedups


if __name__ == "__main__":
  baseline_filenames = {
      f'baseline_{r}': f'baseline_{r}x{r}.nc'
      for r in [64, 128, 256, 512, 1024, 2048]
  }
  learned_filenames = {
      f'learned_interp_{r}': f'learned_{r}x{r}.nc'
      for r in [32, 64, 128]
  }

  data_path = "/global/cfs/cdirs/m3898/zhiqings/cfd"

  models = {}
  for k, v in baseline_filenames.items():
    models[k] = xarray.open_dataset(os.path.join(data_path, f'content/kolmogorov_re_1000_fig1/{v}'), chunks={'time': '100MB'})
  for k, v in learned_filenames.items():
    ds = xarray.open_dataset(os.path.join(data_path, f'content/kolmogorov_re_1000_fig1/{v}'), chunks={'time': '100MB'})
    models[k] = ds.reindex_like(models['baseline_64'], method='nearest')

  print(models.keys())
  print(models['baseline_2048'])
  exit()

  combined_fig1 = xarray.concat(list(models.values()), dim='model')
  combined_fig1.coords['model'] = list(models.keys())
  combined_fig1['vorticity'] = xru.vorticity_2d(combined_fig1)

  df_raw = pd.read_csv(os.path.join(data_path, 'content/kolmogorov_re_1000_fig1/tpu-speed-measurements.csv')).reset_index(drop=True)

  v = combined_fig1.vorticity.thin(time=2).sel(time=slice(10))
  vorticity_correlation = correlation(v, v.sel(model='baseline_2048')).compute()

  times = calculate_time_until(vorticity_correlation)
  times_boot = calculate_time_until_bootstrap(vorticity_correlation)

  df = (
      df_raw
      .drop(['model', 'resolution', 'msec_per_sim_step'], axis=1)
      .set_index('model_name')
      .join(
          times.rename({'model': 'model_name'}).to_dataframe()
      )
      .join(
          times_boot
          .quantile(q=0.975, dim='boot')
          .drop('quantile')
          .rename('time_until_upper')
          .rename({'model': 'model_name'})
          .to_dataframe()
      )
      .join(
          times_boot
          .quantile(q=0.025, dim='boot')
          .drop('quantile')
          .rename('time_until_lower')
          .rename({'model': 'model_name'})
          .to_dataframe()
      )
      .reset_index()
  )
  df[['model', 'resolution']] = df.model_name.str.rsplit('_', 1, expand=True)
  df['resolution'] = df['resolution'].astype(int)
  # switch units from "msec per time step at 64x64" to
  # "sec per simulation time step"
  df['sec_per_sim_time'] = df['msec_per_dt'] / 0.007012 * 1e-3
  df = df.sort_values(['resolution', 'model'])

  # @title Pareto frontier with uncertainty
  plt.figure(figsize=(5, 5))

  df_baseline = df.query('model=="baseline"')
  plt.errorbar(df_baseline.time_until,
               df_baseline.sec_per_sim_time,
               xerr=(df_baseline.time_until - df_baseline.time_until_lower,
                     df_baseline.time_until_upper - df_baseline.time_until),
               marker='s',
               label='baseline')

  df_baseline = df.query('model=="learned_interp"')
  plt.errorbar(df_baseline.time_until,
               df_baseline.sec_per_sim_time,
               xerr=(df_baseline.time_until - df_baseline.time_until_lower,
                     df_baseline.time_until_upper - df_baseline.time_until),
               marker='s',
               label='learned')

  plt.xlim(0, 8.1)
  plt.ylim(1.5e-2, 1e3)
  plt.xlabel('Time until correlation < 0.95')
  plt.ylabel('Runtime per time unit (s)')
  plt.yscale('log')
  plt.legend()
  seaborn.despine()

  plt.savefig("figs/fig1.png")
