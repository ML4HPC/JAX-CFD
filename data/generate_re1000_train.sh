#!/bin/bash
set -e
set -x

export CUDA_VISIBLE_DEVICES=0

DATASET=re1000
STORAGE_PATH=/global/cfs/cdirs/m3898/zhiqings/efficient-transformer
MODEL_PATH=${STORAGE_PATH}/models/${DATASET}/myformer5_16l_10a_410h_t1.0_el0.05_mlp_debug2
DATA_PATH=${STORAGE_PATH}/transformer-xl/data/wikitext-103

