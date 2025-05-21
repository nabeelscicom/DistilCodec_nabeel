#!/bin/bash
#SBATCH --job-name=distilcodec_training # create a short name for your job
#SBATCH --nodes=4 # node count
#SBATCH --gres=gpu:hgx:8 # number of gpus per node
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=32 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G # memory per cpu-core (4G is default)

#SBATCH -p pos
#SBATCH -o /path/to/log_path/%x-%j.log

# mode=debug
mode=train

nnodes=4
num_gpus=8
gpus_per_node=$num_gpus

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
master_port=$(shuf -n 1 -i 40000-65535)
echo Node IP: $head_node_ip
export LOGLEVEL=INFO

base_path=/path/to/config
model_config=${base_path}/model_config.json
train_config=${base_path}/train_config.json
CODE_PATH=/path/to/distilcodec

if [ "${mode}" == "debug" ]; then
  ## debug
  echo "Debug"
  srun torchrun \
    --nnodes=$nnodes \
    --nproc_per_node=$gpus_per_node \
    --max_restarts=1 \
    --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:$master_port \
    ${CODE_PATH}/train.py \
    --model_config ${model_config} \
    --train_config ${train_config} \
    --fsdp \
    --debug
elif [ "$mode" == "train" ]; then
  ## train
  echo "Train model..."
  srun torchrun \
    --nnodes=$nnodes \
    --nproc_per_node=$gpus_per_node \
    --max_restarts=3 \
    --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:$master_port \
    ${CODE_PATH}/train.py \
    --model_config ${model_config} \
    --train_config ${train_config} \
    --amp 
fi
