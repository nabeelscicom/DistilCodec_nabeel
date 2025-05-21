#!/bin/bash
#SBATCH --job-name=codec
#SBATCH --nodes 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx:8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=30G
#SBATCH -p pos
#SBATCH -w ccnl03,ccnl04

#SBATCH -o ./log/codec.log


NUM_GPUS=8


input_path=/cognitive_comp/common_data/audio/output/lam/asr_tts/tts/zh/wenetspeech4tts/standard/ #/cognitive_comp/common_data/audio/output/24k_data/tmp/第五集活了一百万次的猫 #/cognitive_comp/common_data/audio/output/lam/asr_tts/asr/ #/cognitive_comp/common_data/audio/output/24k_data/tmp/第五集活了一百万次的猫 #/cognitive_comp/common_data/audio/output/24k_data/24k_1_10s/stage1/
model_config="/cognitive_comp/common_checkpoint/S_model_management/codec/20240923/0923_24k_3s/model_config.json" #/cognitive_comp/common_checkpoint/S_model_management/codec/20240930/qwen2.5-7b-ate-0930/codec_config.json #"/cognitive_comp/common_checkpoint/S_model_management/codec/20240923/0923_24k_3s/model_config.json"
ckpt_config="/cognitive_comp/common_checkpoint/S_model_management/codec/20240923/saved_ckpt" #/cognitive_comp/common_checkpoint/S_model_management/codec/20240930/qwen2.5-7b-ate-0930 #"/cognitive_comp/common_checkpoint/S_model_management/codec/20240923/saved_ckpt"

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
master_port=$(shuf -n 1 -i 40000-65535)
echo Node IP: $head_node_ip
echo master_port:$master_port

srun torchrun --nproc_per_node=$NUM_GPUS --nnodes=2 \
       --rdzv_id=$RANDOM \
       --rdzv_backend=c10d \
       --rdzv_endpoint=$head_node_ip:$master_port \
       inference_codec.py \
       --input_path ${input_path} \
       --model_config ${model_config} \
       --ckpt_config ${ckpt_config} 

