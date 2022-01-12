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
MODEL_NAME=learned_64_orig_pre2_fc_v2_nodp

python -u models/train.py \
  --model_encode_steps=16 \
  --model_decode_steps=160 \
  --model_predict_steps=16 \
  --train_device_batch_size=4 \
  --delta_time=0.007012483601762931 \
  --train_split="$STORAGE_PATH/$TRAINDATA" \
  --eval_split="$STORAGE_PATH/$EVALDATA" \
  --predict_split="$STORAGE_PATH/$PREDICTDATA" \
  --predict_result="my_predict.nc" \
  --eval_batch_size=48 \
  --train_weight_decay=0.0 \
  --train_lr_init=0.0001 \
  --train_lr_warmup_epochs=0.0 \
  --mp_scale_value=1.0 \
  --train_epochs=1.0 \
  --train_log_every=10 \
  --decoding_warmup_steps=400 \
  --decoding_warmup_stages=40 \
  --resume_checkpoint \
  --mp_skip_nonfinite \
  --do_predict \
  --do_eval \
  --inner_steps 10 \
  --output_dir="$STORAGE_PATH/models/$MODEL_NAME" \
  --gin_file="models/configs/official_li_config.gin" \
  --gin_file="models/configs/kolmogorov_forcing.gin" \
  --gin_param="fixed_scale.rescaled_one = 0.2" \
  --gin_param="my_forward_tower_factory.num_hidden_channels = 256" \
  --gin_param="my_forward_tower_factory.num_hidden_layers = 6" \
  --gin_param="my_aligned_array_encoder.n_frames = 2" \
  --gin_param="my_forward_tower_factory.dropout_rate = 0.1" \
  --gin_param="MyFusedLearnedInterpolation.pattern = \"original\"" \
  2>&1 | tee ./logs/train_log.txt
