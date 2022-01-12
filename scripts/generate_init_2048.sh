#!/bin/bash
set -e
set -x

export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HAIKU_FLATMAPPING=0

STORAGE_PATH=/global/cfs/cdirs/m3898/zhiqings/cfd
MODEL_NAME=dns_2048x2048

python -u models/generate_data.py \
  --model_predict_steps=16 \
  --delta_time=0.007012483601762931 \
  --num_samples=16 \
  --inner_steps=1 \
  --model_input_size=2048 \
  --save_grid_size=64 \
  --warmup_time=40.0 \
  --simulation_time=0.1 \
  --output_dir="$STORAGE_PATH/models/$MODEL_NAME" \
  --gin_file="models/configs/implicit_diffusion_dns_config.gin" \
  --gin_file="models/configs/kolmogorov_forcing.gin" \
  2>&1 | tee ./logs/train_log.txt
