import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import jax_cfd.ml as cfd_ml
import numpy as np
import seaborn
import xarray
import matplotlib.pyplot as plt
import logging
import argparse


def get_dt(args, size, cl):
  grid = cfd.grids.Grid((size, size), domain=((0, 2 * jnp.pi * cl * args.domain_scale),
                                              (0, 2 * jnp.pi * cl * args.domain_scale)))
  # Choose a time step.
  dt = cfd.equations.stable_time_step(
      args.max_velocity, args.cfl_safety_factor, args.viscosity / cl, grid)
  return dt


def get_trajectory(args, size, rng=None, outer_steps=50, v0=None):
  cl = args.characteristic_length
  grid = cfd.grids.Grid((size, size), domain=((0, 2 * jnp.pi * cl * args.domain_scale),
                                              (0, 2 * jnp.pi * cl * args.domain_scale)))

  # Choose a time step.
  dt = get_dt(args, size, cl)
  delta_t = get_dt(args, args.low_res, cl)
  inner_steps = round(delta_t / dt)
  logging.info("inner step %d" % inner_steps)

  # Define the physical dimensions of the simulation.

  if args.decay:
    forcing = None
  else:
    forcing = cfd_ml.forcings.kolmogorov_forcing(grid,
                                                 args.forcing_scale / cl,
                                                 args.peak_wavenumber / cl,
                                                 -0.1 / cl)

  # Construct a random initial velocity. The `filtered_velocity_field` function
  # ensures that the initial velocity is divergence free and it filters out
  # high frequency fluctuations.
  if v0 is None:
    v0 = cfd.initial_conditions.filtered_velocity_field(rng, grid, args.max_velocity,
                                                        args.peak_wavenumber / cl)
  elif size < args.high_res:
    large_grid = cfd.grids.Grid((args.high_res, args.high_res), domain=((0, 2 * jnp.pi * cl * args.domain_scale),
                                                                        (0, 2 * jnp.pi * cl * args.domain_scale)))
    v0 = cfd.resize.downsample_staggered_velocity(large_grid, grid, v0)

  # Define a step function and use it to compute a trajectory.
  step_fn = cfd.funcutils.repeated(
      cfd.equations.semi_implicit_navier_stokes(
          density=args.density, viscosity=args.viscosity / cl,
          dt=dt, grid=grid, forcing=forcing),
      steps=inner_steps)

  # trajectory_fn = cfd.funcutils.trajectory(step_fn, outer_steps)
  # _, trajectory = trajectory_fn(v0)

  rollout_fn = jax.jit(cfd.funcutils.trajectory(step_fn, outer_steps))
  _, trajectory = jax.device_get(rollout_fn(v0))
  return trajectory


def plot_trajectory(args, size, trajectory, file_name):
  cl = args.characteristic_length
  grid = cfd.grids.Grid((size, size), domain=((0, 2 * jnp.pi * cl * args.domain_scale),
                                              (0, 2 * jnp.pi * cl * args.domain_scale)))

  x_len = grid.axes()[0].shape[0]
  x = 2 * np.double(grid.axes()[0]).mean() / x_len * np.arange(x_len)

  y_len = grid.axes()[1].shape[0]
  y = 2 * np.double(grid.axes()[1]).mean() / y_len * np.arange(y_len)

  # load into xarray for visualization and analysis
  delta_t = get_dt(args, args.low_res, cl)
  ds = xarray.Dataset(
      {
          'u': (('time', 'x', 'y'), trajectory[0].data),
          'v': (('time', 'x', 'y'), trajectory[1].data),
      },
      coords={
          'x': x,
          'y': y,
          'time': (delta_t * np.arange(trajectory[0].shape[0]))
      }
  )

  def vorticity(ds):
    x = (ds.v.differentiate('x') - ds.u.differentiate('y'))
    x = x.rename('vorticity')
    return x

  (ds.pipe(vorticity).thin(time=args.demo_steps // 5).transpose()
   .plot.imshow(col='time', cmap=seaborn.cm.icefire, robust=True, col_wrap=5))

  plt.savefig(file_name)


def main(args):
  logger = logging.getLogger()
  logger.setLevel("INFO")
  seed = args.seed
  warmup_time = args.warmup_time
  outer_steps = args.outer_steps
  rng = jax.random.PRNGKey(seed)

  for iter in range(args.iters):
    rng, subrng = jax.random.split(rng)

    delta_t = get_dt(args, args.low_res, args.characteristic_length)
    warm_up_step = round(warmup_time / delta_t)

    count = 0
    warmup_result = None

    while count + outer_steps <= warm_up_step:
      logger.info(f"step {count} of {warm_up_step}")
      if warmup_result is not None:
        warmup_result[0].array.data = warmup_result[0].array.data[-1]
        warmup_result[1].array.data = warmup_result[1].array.data[-1]
      warmup_result = get_trajectory(args, size=args.high_res, rng=subrng,
                                     outer_steps=outer_steps, v0=warmup_result)
      count += outer_steps

    if warm_up_step > count:
      if warmup_result is not None:
        warmup_result[0].array.data = warmup_result[0].array.data[-1]
        warmup_result[1].array.data = warmup_result[1].array.data[-1]
      warmup_result = get_trajectory(args, size=args.high_res, rng=subrng,
                                     outer_steps=warm_up_step - count, v0=warmup_result)

    warmup_result[0].array.data = warmup_result[0].array.data[-1]
    warmup_result[1].array.data = warmup_result[1].array.data[-1]

    resolution_list = []
    res = args.low_res
    while res <= args.high_res:
      resolution_list.append(res)
      res *= 2

    if args.demo:
      for resolution in resolution_list:
        trajectory = get_trajectory(args, size=resolution, outer_steps=args.demo_steps, v0=warmup_result)
        file_name = f'../figs/%s_demo_{resolution}x{resolution}.png' % args.save_file
        logger.info(file_name)
        plot_trajectory(args, resolution, trajectory, file_name)
    else:
      pass


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--iters', type=int, default=1)
  parser.add_argument('--outer_steps', type=int, default=50)
  parser.add_argument('--demo_steps', type=int, default=50)
  parser.add_argument('--generate_steps', type=int, default=50)
  parser.add_argument('--warmup_time', type=float, default=40)
  parser.add_argument('--max_velocity', type=float, default=7.0)
  parser.add_argument('--cfl_safety_factor', type=float, default=0.5)
  parser.add_argument('--viscosity', type=float, default=1e-3)
  parser.add_argument('--density', type=float, default=1.0)
  parser.add_argument('--forcing_scale', type=float, default=1.0)
  parser.add_argument('--simulation_time', type=float, default=30.0)
  parser.add_argument('--peak_wavenumber', type=int, default=4)
  parser.add_argument('--low_res', type=int, default=64)
  parser.add_argument('--high_res', type=int, default=2048)
  parser.add_argument('--demo_file', type=str, default="re1000")
  parser.add_argument('--characteristic_length', type=int, default=1)
  parser.add_argument('--domain_scale', type=int, default=1)
  parser.add_argument('--decay', default=False, action='store_true')
  parser.add_argument('--demo', default=False, action='store_true')
  # For generating data
  parser.add_argument('--save_file', type=str, default="re1000")
  parser.add_argument('--save_index', type=int, default=1)
  main(parser.parse_args())
