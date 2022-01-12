# JAX-CFD
Machine Learning-accelerated Computational Fluid Dynamics (CFD)

# Datasets
The training data and evaluation data are in `/global/cfs/cdirs/m3898/zhiqings/cfd/content`

The trained models are in `/global/cfs/cdirs/m3898/zhiqings/cfd/models`

# Installation
```
conda env create -f cfd.yml
conda activate cfd
pip install git+https://github.com/google/jax-cfd.git
```

# Run

All the scripts for reproduction are in `scripts` directory.
For visualization, check `plot_fig*.py` files in the main directory.
