import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import argparse
import json

import torch
torch.backends.cudnn.benchmark = True

from utils import AttrDict
from utils import setup, cleanup, print_rank_0
from evaluation import evaluation
    
    
def main():
    print('Initializing Evaluation Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='')
    parser.add_argument('--checkpoint_path', default='')
    parser.add_argument('--valid_path', default='')
    parser.add_argument("--skip_decoding", action="store_true", default=False)
    parser.add_argument("--bfloat16", action="store_true", default=False)
    args = parser.parse_args()
    print(args)
    
    setup()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    pid = os.getpid()
    print(f'current pid: {pid}, rank: {rank}, local rank: {local_rank}')
    
    with open(args.model_config) as f:
        model_config = json.loads(f.read())
        model_config = AttrDict(model_config)
        print_rank_0(f'Model config: {json.dumps(model_config, indent=4)}')
        
    with open(args.valid_path) as f:
        valid_config = json.loads(f.read())
        valid_config = AttrDict(valid_config)
        print_rank_0(f'Validation config: {json.dumps(valid_config, indent=4)}')
    
    model_config = args.model_config
    ckpt_path = args.checkpoint_path
    evaluation(valid_config=valid_config,
               ckpt_path=ckpt_path,
               config_path=model_config,
               skip_decoding=args.skip_decoding)
        
    cleanup()


if __name__ == '__main__':
    main()
