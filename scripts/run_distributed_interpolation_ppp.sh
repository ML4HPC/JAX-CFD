#!/bin/bash -l
#SBATCH -C gpu
#SBATCH -q early_science
#SBATCH -t 6:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-task=4
#SBATCH --job-name=xs-transformer
#SBATCH -o /global/homes/z/zhiqings/tmp/slurm_logs/slurm-%j.out
#SBATCH -A m3898_g
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@90

# Load software
# module load cuda

source /global/homes/z/zhiqings/.bashrc
conda activate cfd

set -e
set -x

export SLURM_CPU_BIND="cores"
export MASTER_PORT=12346
export WORLD_SIZE=$SLURM_JOB_NUM_NODES

# Setup node list
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
master_node=${nodes_array[0]}
master_addr=$(srun --nodes=1 --ntasks=1 -w $master_node hostname --ip-address)
master_addr_array=( $master_addr )
master_addr=${master_addr_array[0]}
worker_num=$(($SLURM_JOB_NUM_NODES))
export MASTER_ADDR=$master_addr
export NNODES=$worker_num


#HOST_FILE="./ds_11b_hostfile"
#scontrol show hostnames $SLURM_JOB_NODELIST | awk '$0=$0" slots=4"' > $HOST_FILE

echo ${nodes}
echo ${master_addr}

export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HAIKU_FLATMAPPING=0

TRAINDATA=content/kolmogorov_re_1000/train_2048x2048_64x64.nc
EVALDATA=content/kolmogorov_re_1000/long_eval_2048x2048_64x64.nc
STORAGE_PATH=/global/cfs/cdirs/m3898/zhiqings/cfd
MODEL_NAME=learned_64_dist_ppp

export CMD=" \
    --model_encode_steps=32 \
    --model_decode_steps=640 \
    --model_predict_steps=16 \
    --train_device_batch_size=1 \
    --delta_time=0.007012483601762931 \
    --train_split="$STORAGE_PATH/$TRAINDATA" \
    --eval_split="$STORAGE_PATH/$EVALDATA" \
    --eval_batch_size=48 \
    --train_weight_decay=0.0 \
    --train_lr_init=0.0001 \
    --train_lr_warmup_epochs=0.0 \
    --mp_scale_value=1.0 \
    --train_epochs=1.0 \
    --train_log_every=10 \
    --decoding_warmup_steps=2000 \
    --decoding_warmup_stages=160 \
    --resume_checkpoint \
    --mp_skip_nonfinite \
    --do_eval \
    --do_predict \
    --inner_steps=10 \
    --output_dir=\"$STORAGE_PATH/models/$MODEL_NAME\" \
    --gin_file=\"models/configs/official_li_config_original.gin\" \
    --gin_file=\"models/configs/kolmogorov_forcing.gin\" \
    --gin_param=\"fixed_scale.rescaled_one = 0.2\" \
    --gin_param=\"forward_tower_factory.num_hidden_channels = 128\" \
    --gin_param=\"forward_tower_factory.num_hidden_layers = 6\" \
  "

# Loop over nodes and submit training tasks
for ((  node_rank=0; node_rank<$worker_num; node_rank++ ))
do
  node=${nodes_array[$node_rank]}
  export NODE_RANK="$node_rank"
  echo "Submitting node # $node_rank, $node"

  export NODE_RANK="$node_rank"
  export JOB_ID="$SLURM_JOB_ID"
  export NUM_TRAINERS="$NPROC_PER_NODE"
  export LOCAL_WORLD_SIZE=$NUM_TRAINERS
  export GROUP_RANK="$node_rank"
  export HOST_NODE_ADDR="$MASTER_ADDR:$MASTER_PORT"

  export LAUNCHER="python -u models/train.py \
      --host_address=$HOST_NODE_ADDR \
      "

  echo "$LAUNCHER $CMD"

  # Launch one SLURM task per node, and use torch distributed launch utility
  # to spawn training worker processes; one per GPU
  LOG_FILE=/global/homes/z/zhiqings/tmp/slurm_logs/slurm-${SLURM_JOB_ID}-$( printf '%03d' $node_rank ).out

  srun -l -N 1 -n 1 -w "$node" bash -c "$LAUNCHER $CMD" > $LOG_FILE 2>&1 &
  pids[${node_rank}]=$!
done

# Wait for completion
for pid in ${pids[*]}; do
    wait $pid
done
