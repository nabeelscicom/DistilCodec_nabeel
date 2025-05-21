#!/bin/bash
#SBATCH --job-name=audio_eval # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --gres=gpu:hgx:1 # number of gpus per node
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=8 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G # memory per cpu-core (4G is default)

#SBATCH -p pos
#SBATCH -o /cognitive_comp/wangrui/codes/audio_codec/log/%x-%j.log

nnodes=1
num_gpus=1
gpus_per_node=$num_gpus

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
master_port=$(shuf -n 1 -i 40000-65535)
echo Node IP: $head_node_ip
export LOGLEVEL=INFO

date=0821
config_path=/cognitive_comp/wangrui/codes/audio_codec/scripts/workspace/${date}_24k_3s
model_config=${config_path}/model_config.json
ckpt_data_path=/cognitive_comp/wangrui/data/lam${date}
ckpt_path=${ckpt_data_path}/ckpt
valid_path=${ckpt_data_path}/valid_files.json

CODE_PATH=/cognitive_comp/wangrui/codes/audio_codec

## train
echo "Audio Codec evaluating..."
srun torchrun \
    --nnodes=$nnodes \
    --nproc_per_node=$gpus_per_node \
    --max_restarts=1 \
    --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:$master_port \
    ${CODE_PATH}/evaluate.py \
    --model_config ${model_config} \
    --checkpoint_path ${ckpt_path} \
    --valid_path ${valid_path} \
