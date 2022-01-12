#!/bin/bash
set -e
set -x

export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HAIKU_FLATMAPPING=0

TRAINDATA=content/kolmogorov_re_1000/train_2048x2048_64x64.nc
EVALDATA=content/kolmogorov_re_1000/long_eval_2048x2048_64x64.nc
PREDICTDATA=models/dns_2048x2048/predict.nc
STORAGE_PATH=/global/cfs/cdirs/m3898/zhiqings/cfd
MODEL_NAME=my_dns_512

python -u models/train.py \
  --model_input_size=512 \
  --model_encode_steps=16 \
  --model_decode_steps=16 \
  --model_predict_steps=16 \
  --delta_time=0.007012483601762931 \
  --train_split="$STORAGE_PATH/$TRAINDATA" \
  --eval_split="$STORAGE_PATH/$EVALDATA" \
  --predict_split="$STORAGE_PATH/$PREDICTDATA" \
  --train_device_batch_size=4 \
  --eval_batch_size=16 \
  --train_lr_init=0.001 \
  --train_lr_warmup_epochs 0 \
  --train_log_every=10 \
  --train_epochs 0.0 \
  --do_predict \
  --inner_steps 10 \
  --output_dir="$STORAGE_PATH/models/$MODEL_NAME" \
  --gin_file="models/configs/implicit_diffusion_dns_config.gin" \
  --gin_file="models/configs/kolmogorov_forcing.gin" \
  2>&1 | tee ./logs/train_log.txt
