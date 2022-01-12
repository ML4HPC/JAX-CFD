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
  models_color = seaborn.xkcd_palette(['burnt orange'])
  palette = baseline_palette + models_color

  filenames = {
      f'baseline_{r}': f'long_eval_{r}x{r}_64x64.nc'
      for r in [64, 128, 256, 512, 1024, 2048]
  }
  filenames['learned_interp_64'] = 'learned_interpolation_long_eval_64x64_64x64.nc'
  filenames['train'] = f'train_2048x2048_64x64.nc'

  with open("../logs/log_train.txt", "w") as fout:
    models = {}
    for k, v in filenames.items():
      models[k] = xarray.open_dataset(os.path.join(data_path, f'content/kolmogorov_re_1000/{v}'), chunks={'time': '100MB'})
      fout.write(f"{k}\n")
      fout.write(f"{models[k]}\n")
      print(f"{models[k].sizes}")
      for attr in models[k].attrs:
        fout.write(f"{attr}: ")
        fout.write(str(models[k].__getattr__(attr)) + "\n")
      fout.write("\n\n\n\n\n")
