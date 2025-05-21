import os
import sys
import inspect

import subprocess

import torch
from torch import distributed as dist


def setup():
    # 初始化分布式多进程
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group('nccl')


def print_with_line(message):
    frame, filename, line_number, function_name, lines, index = inspect.getouterframes(inspect.currentframe())[1]
    print(f"{filename}:{line_number} {message}", file=sys.stdout)
    
    
def print_rank_0(*message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*message, flush=True)
    else:
        print(*message, flush=True)
    
    
def cleanup():
    # 关闭所有进程
    dist.barrier()
    dist.destroy_process_group()
    

def print_cuda_info():
    # 定义命令
    command = "nvidia-smi"

    # 运行命令
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    # 获取输出和错误信息
    output, error = process.communicate()

    # 打印输出
    if process.returncode == 0:
        print_rank_0("命令输出:\n", output.decode())
    else:
        print_rank_0("错误信息:\n", error.decode())
        
        
def is_primary():
    return get_rank() == 0


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0

    return dist.get_rank()


def local_rank():
    """Local rank of process"""
    local_rank = os.environ.get("LOCAL_RANK")

    if local_rank is None:
        local_rank = os.environ.get("SLURM_LOCALID")

    if local_rank is None:
        print(
            "utils.local_rank() environment variable LOCAL_RANK not set, defaulting to 0",
            flush=True,
        )
        local_rank = 0
        
    return int(local_rank)
