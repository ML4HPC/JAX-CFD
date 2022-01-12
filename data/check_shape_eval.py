import os.path

import xarray
import seaborn
import numpy as np
import pandas as pd
import jax_cfd.data.xarray_utils as xru
from jax_cfd.data import evaluation
import matplotlib.pyplot as plt

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

  def vorticity(ds):
    x = (ds.v.differentiate('x') - ds.u.differentiate('y'))
    x = x.rename('vorticity')
    return x

  with open("../logs/log_eval.txt", "w") as fout:
    models = {}
    for k, v in baseline_filenames.items():
      models[k] = xarray.open_dataset(os.path.join(data_path, f'content/kolmogorov_re_1000_fig1/{v}'), chunks={'time': '100MB'})
      fout.write(f"{k}\n")
      fout.write(f"{models[k]}\n")
      for attr in models[k].attrs:
        fout.write(f"{attr}: ")
        fout.write(str(models[k].__getattr__(attr)) + "\n")
      fout.write("\n\n\n\n\n")

      ds = models[k]

      (ds[dict(sample=0)].pipe(vorticity).head(time=200).thin(time=20).transpose()
       .plot.imshow(col='time', cmap=seaborn.cm.icefire, robust=True, col_wrap=5))
      plt.savefig("../samples/%s" % k)

    for k, v in learned_filenames.items():
      ds = xarray.open_dataset(os.path.join(data_path, f'content/kolmogorov_re_1000_fig1/{v}'), chunks={'time': '100MB'})
      models[k] = ds.reindex_like(models['baseline_64'], method='nearest')
      fout.write(f"{k}\n")
      fout.write(f"{models[k]}\n")
      for attr in models[k].attrs:
        fout.write(f"{attr}: ")
        fout.write(str(models[k].__getattr__(attr)) + "\n")
      fout.write("\n\n\n\n\n")

      ds = models[k]
      (ds[dict(sample=0)].pipe(vorticity).head(time=200).thin(time=20).transpose()
       .plot.imshow(col='time', cmap=seaborn.cm.icefire, robust=True, col_wrap=5))
      plt.savefig("../samples/%s" % k)
