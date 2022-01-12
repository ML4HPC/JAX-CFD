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
MODEL_NAME=learned_64_orig_pre4_fc_short

python -u models/train.py \
  --model_encode_steps=16 \
  --model_decode_steps=32 \
  --train_device_batch_size=16 \
  --delta_time=0.007012483601762931 \
  --train_split="$STORAGE_PATH/$TRAINDATA" \
  --eval_split="$STORAGE_PATH/$EVALDATA" \
  --predict_split="$STORAGE_PATH/$PREDICTDATA" \
  --predict_result="my_predict.nc" \
  --eval_batch_size=48 \
  --train_weight_decay=0.0 \
  --train_lr_init=0.001 \
  --train_lr_warmup_epochs=0.0 \
  --mp_scale_value=1.0 \
  --train_epochs=4.0 \
  --train_log_every=10 \
  --resume_checkpoint \
  --mp_skip_nonfinite \
  --do_predict \
  --inner_steps 1 \
  --explicit_inner_steps 10 \
  --model_predict_steps=160 \
  --output_dir="$STORAGE_PATH/models/$MODEL_NAME" \
  --gin_file="models/configs/official_li_config.gin" \
  --gin_file="models/configs/kolmogorov_forcing.gin" \
  --gin_param="fixed_scale.rescaled_one = 0.2" \
  --gin_param="my_forward_tower_factory.num_hidden_channels = 128" \
  --gin_param="my_forward_tower_factory.num_hidden_layers = 6" \
  --gin_param="my_aligned_array_encoder.n_frames = 4" \
  --gin_param="my_trajectory_from_step.set_checkpoint = True" \
  --gin_param="MyFusedLearnedInterpolation.pattern = \"original\"" \
  2>&1 | tee ./logs/train_log.txt
