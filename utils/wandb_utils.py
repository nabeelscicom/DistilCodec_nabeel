import socket
import os

import torch
import wandb
from wandb import UsageError
from torch.utils.tensorboard import SummaryWriter


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
        
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


def wandb_log(key_values: dict, iteration_no):
    # logs to both tb and wandb (if present) from the zeroth rank
    do_log = torch.distributed.get_rank() == 0
    for ele_l1 in key_values.items():
        key_l1 = ele_l1[0]
        value_l1 = ele_l1[1]
        for ele_l2 in value_l1.items():
            key_l2 = ele_l2[0]
            value_l2 = ele_l2[1]
            wandb_key = f'{key_l1}/{key_l2}'
            if do_log and value_l2 is not None:
                wandb.log({wandb_key: value_l2}, 
                          step=iteration_no)
            
            
def init_wandb_tb(train_config, model_config) -> SummaryWriter:
    wandb_config = AttrDict(train_config['wandb'])
    if wandb_config.use_wandb:
        group_name = wandb_config.wandb_group
        name = f"{socket.gethostname()}-{local_rank()}" if group_name else None
        try:
            wandb.init(
                project=wandb_config.wandb_project,
                group=group_name,
                name=name,
                save_code=False,
                force=False,
                entity=wandb_config.wandb_team,
            )
        except UsageError as e:
            wandb_config.update_value("use_wandb", False)
            print(e)
            print(
                "Skipping wandb. Execute `wandb login` on local or main node machine to enable.",
                flush=True,
            )
        configs = {
            'train_config': train_config,
            'model_config': model_config
        }
        wandb.config.update(configs)
    
    sw = SummaryWriter(train_config.logger_path)
    
    return sw
    