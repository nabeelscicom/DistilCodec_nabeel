import os
import pathlib
import glob
import re
import random
import functools
import itertools
import json
import copy

import numpy as np
import torch
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    MixedPrecision
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from models import MultiScaleSTFTDiscriminator
from models import HiFiGANGenerator
from models import MultiPeriodDiscriminator
from models import MultiScaleDiscriminator
from models import ConvNeXtEncoder
from vector_quantization import DownsampleGRFSQ
from vector_quantization import DownsampleGRVQ
from code_distiller import MultiGroupDistillation
from models import MelDataset


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=torch.device('cpu'))
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj, num_ckpt_keep=5):
    name = re.match(r'(do|g)_\d+', pathlib.Path(filepath).name).group(1)
    ckpts = sorted(pathlib.Path(filepath).parent.glob(f'{name}_*'))
    if len(ckpts) > num_ckpt_keep:
        [os.remove(c) for c in ckpts[:-num_ckpt_keep]]
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix, target_steps: int = -1):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    
    if target_steps != -1:
        for p in cp_list:
            if p.endswith(str(target_steps)):
                return p
    
    return sorted(cp_list)[-1]


def seed_everything(seed, cudnn_deterministic=False):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
    
    Args:
        seed: the integer value seed for global random state
    """
    if seed is not None:
        # print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        
bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
)


my_auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy, 
    min_num_params=20000)


def init_fsdp_model(model: torch.nn.Module):
    fsdp_model = FSDP(model, 
                      auto_wrap_policy=my_auto_wrap_policy,
                      device_id=torch.cuda.current_device(),
                      # mixed_precision=bfSixteen,
                      # sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
                      backward_prefetch=BackwardPrefetch.BACKWARD_PRE)
    
    return fsdp_model


def fsdp_state_dict(model: torch.nn.Module):
    print('fsdp state dict1')
    save_policy = FullStateDictConfig(offload_to_cpu=True, 
                                      rank0_only=True)
    with FSDP.state_dict_type(
                model, 
                StateDictType.FULL_STATE_DICT, 
                save_policy):
        cpu_state = model.state_dict()
    print('fsdp state dict1')
    
    return cpu_state


def is_startswith_filterd(param_name: str, filtered_params: list) -> bool:
    for filtered in filtered_params:
        if param_name.startswith(filtered):
            return True
    return False


def load_enocder_state_dict(encoder: torch.nn.Module, 
                            state_dict_g, 
                            mel_chs, 
                            enc_input_chs) -> torch.nn.Module:
    pretrained_enc_state_dict = state_dict_g["encoder"]
    if mel_chs != enc_input_chs:
        filtered_param_names = ["downsample_layers.0.0.weight", "downsample_layers.0.0.bias"]
        print(f'Mel channels [{mel_chs}] is not equal to encoder input channels [{enc_input_chs}], so we will not load weights of {filtered_param_names}')
        enc_state_dict = encoder.state_dict()
        filtered_state_dict = {k: v for k, v in pretrained_enc_state_dict.items() 
                               if not is_startswith_filterd(k, filtered_param_names) and k in enc_state_dict}
        enc_state_dict.update(filtered_state_dict)
        encoder.load_state_dict(enc_state_dict)
    else:
        encoder.load_state_dict(pretrained_enc_state_dict, strict=True)

    return encoder


def load_decoder_state_dict(generator: torch.nn.Module, 
                            state_dict_g, 
                            stu_configs,
                            tea_configs) -> torch.nn.Module:

    pretrained_gen_state_dict = state_dict_g["generator"]
    filtered_params = []
    stu_ups = stu_configs['upsample_rates']
    stu_up_kernals = stu_configs['upsample_kernel_sizes']
    tea_ups = tea_configs['upsample_rates']
    tea_up_kernals = tea_configs['upsample_kernel_sizes']
    if len(stu_ups) != len(tea_ups):
        filtered_params.append('conv_post')
    for i, z_ups in enumerate(zip(stu_configs['upsample_rates'],
                              tea_configs['upsample_rates'])):
        stu_ups, tea_ups = z_ups
        if stu_ups != tea_ups or stu_up_kernals[i] != tea_up_kernals[i]:
            filtered_params.append(f'ups.{i}')
    print(f'Filterd parameters: {filtered_params}')
    gen_state_dict = generator.state_dict()
    filtered_state_dict = {k: v for k, v in pretrained_gen_state_dict.items() 
                           if not is_startswith_filterd(k, filtered_params) and k in gen_state_dict}
    gen_state_dict.update(filtered_state_dict)
    generator.load_state_dict(gen_state_dict)

    return generator


def load_pretrain_quantizer_state_dict(quantizer: torch.nn.Module,
                                       state_dict_g) -> torch.nn.Module:
    pretrained_quantizer_state_dict = state_dict_g["quantizer"]
    quantizer_state_dict = quantizer.state_dict()
    print(f'Loading downsample and upsample pretrained params...')
    filtered_state_dict = {k: v for k, v in pretrained_quantizer_state_dict.items() 
                           if k.startswith("downsample") and k in quantizer_state_dict} # or k.startswith("upsample")) and k in quantizer_state_dict}
    quantizer_state_dict.update(filtered_state_dict)
    quantizer.load_state_dict(quantizer_state_dict)

    return quantizer


def init_model_and_optimizer(
        model_config: AttrDict, 
        train_config: AttrDict, 
        world_rank,
        local_rank,
        device, 
        use_fsdp=False, 
        is_debug=False,
        is_distill=False,
        quantizer_transfer=False):
    
    encoder_configs = model_config['encoder']
    mel_chs = model_config['spec_transform']['num_mels']
    stu_enc_input_chs = encoder_configs['input_channels']
    if mel_chs != stu_enc_input_chs:
        stu_encoder_configs = copy.deepcopy(encoder_configs)
        stu_encoder_configs['input_channels'] = mel_chs
        encoder = ConvNeXtEncoder(**stu_encoder_configs).to(device)
    else:
        encoder = ConvNeXtEncoder(**encoder_configs).to(device)
    
    generator_configs = model_config['decoder']
    generator = HiFiGANGenerator(**generator_configs).to(device)
    
    quantizer_config = model_config['quantizer']
    quantizer_type = quantizer_config['quantizer_type']
    quantizer_config.pop('quantizer_type')
    if quantizer_type == 'fsq':
        quantizer_config.pop('codebook_size')
        quantizer_config.pop('codebook_dim')
        quantizer = DownsampleGRFSQ(**quantizer_config).to(device)
    elif quantizer_type == 'grvq':
        quantizer = DownsampleGRVQ(**quantizer_config).to(device)
    else:
        raise ValueError('Quantier current surpport [grfsq, grvq]')
    
    teacher_quantizer_config = model_config['teacher_quantizer']
    tea_quantizer_type = teacher_quantizer_config['quantizer_type']
    teacher_quantizer_config.pop('quantizer_type')
    if tea_quantizer_type == 'fsq':
        teacher_quantizer_config.pop('codebook_size')
        teacher_quantizer_config.pop('codebook_dim')
        tea_quantizer = DownsampleGRFSQ(**teacher_quantizer_config).to(device)
    elif tea_quantizer_type == 'grvq':
        tea_quantizer = DownsampleGRVQ(**teacher_quantizer_config).to(device)
    else:
        raise ValueError('Quantier current surpport [grfsq, grvq]')
    n_group = teacher_quantizer_config['n_groups']
    group_in_dim = teacher_quantizer_config['codebook_dim']
    group_out_dim = quantizer_config['codebook_dim'] // n_group
    distiller = MultiGroupDistillation(n_group=n_group,
                                       group_in_dim=group_in_dim,
                                       group_out_dim=group_out_dim).to(device)
    
    discriminator_configs = model_config['descriminators']
    mpd = MultiPeriodDiscriminator(mpd_config=discriminator_configs['MultiPeriodDiscriminator']).to(device)
    msd = MultiScaleDiscriminator(msd_config=discriminator_configs['MultiScaleDiscriminator']).to(device)
    mstftd = MultiScaleSTFTDiscriminator(msstft_config=discriminator_configs['MultiScaleSTFTDiscriminator']).to(device)
    
    if world_rank == 0 and is_debug:
        print(encoder)
        print(quantizer)
        print(generator)
    
    is_distill_valid = False
    if is_distill:
        teacher_ckpt_path = train_config.distill['teacher_ckpt_path']
        distill_cp_g = scan_checkpoint(teacher_ckpt_path, 'g_')
        distill_cp_do = scan_checkpoint(teacher_ckpt_path, 'do_')
        if (distill_cp_g is not None) and (distill_cp_do is not None):
            print(f'Rank{world_rank} loading teacher ckpt...')
            tea_state_dict_g = load_checkpoint(distill_cp_g, device)
            tea_state_dict_do = load_checkpoint(distill_cp_do, device)
            is_distill_valid = True
        else:
            raise ValueError(f'Teacher ckpt path [{teacher_ckpt_path}] has no checkpoint')
    
    if 'load_path' in train_config:
        stu_cp_g = scan_checkpoint(train_config.load_path, 'g_')
        stu_cp_do = scan_checkpoint(train_config.load_path, 'do_')
    else:
        stu_cp_g = None
        stu_cp_do = None

    steps = 0
    last_epoch = -1
    min_mel_error = 100000.0
    if stu_cp_g is None or stu_cp_do is None:
        stu_state_dict_do = None
        if is_distill_valid:
            print(f'Rank{world_rank} loading teacher generator as initial model...')
            if 'teacher_generator' not in model_config:
                generator.load_state_dict(tea_state_dict_g['generator'])
            else:
                stu_gen_configs = model_config['decoder']
                tea_gen_configs = model_config['teacher_generator']
                generator = load_decoder_state_dict(generator=generator,
                                                    state_dict_g=tea_state_dict_g,
                                                    stu_configs=stu_gen_configs,
                                                    tea_configs=tea_gen_configs)
            # print(f'Rank{world_rank} loading teacher discriminators as initial model...')
            # mpd.load_state_dict(tea_state_dict_do['mpd'])
            # msd.load_state_dict(tea_state_dict_do['msd'])
            # mstftd.load_state_dict(tea_state_dict_do['mstftd'])

            encoder = load_enocder_state_dict(
                encoder=encoder,
                state_dict_g=tea_state_dict_g,
                mel_chs=mel_chs,
                enc_input_chs=stu_enc_input_chs
            )
            if quantizer_transfer:
                quantizer = load_pretrain_quantizer_state_dict(
                    quantizer=quantizer,
                    state_dict_g=tea_state_dict_g
                )
    else:
        print(f'Rank{world_rank} loading "encoder" "generator" "quantizer" ckpt...')
        stu_state_dict_g = load_checkpoint(stu_cp_g, device)
        stu_state_dict_do = load_checkpoint(stu_cp_do, device)
        generator.load_state_dict(stu_state_dict_g['generator'])
        encoder = load_enocder_state_dict(
            encoder=encoder,
            state_dict_g=stu_state_dict_g,
            mel_chs=mel_chs,
            enc_input_chs=stu_enc_input_chs
        )
        quantizer.load_state_dict(stu_state_dict_g['quantizer'])
        mpd.load_state_dict(stu_state_dict_do['mpd'])
        msd.load_state_dict(stu_state_dict_do['msd'])
        mstftd.load_state_dict(stu_state_dict_do['mstftd'])
        if mel_chs == stu_enc_input_chs:
            last_epoch = stu_state_dict_do['epoch']
            steps = stu_state_dict_do['steps'] + 1
            min_mel_error = stu_state_dict_do['min_mel_error'] if 'min_mel_error' in stu_state_dict_do else min_mel_error

    if train_config.num_gpus > 1:
        if not use_fsdp:
            generator = DistributedDataParallel(
                generator, device_ids=[local_rank]).to(device)
            encoder = DistributedDataParallel(encoder, device_ids=[local_rank]).to(device)
            quantizer = DistributedDataParallel(
                quantizer, device_ids=[local_rank]).to(device)
            mpd = DistributedDataParallel(mpd, device_ids=[local_rank]).to(device)
            msd = DistributedDataParallel(msd, device_ids=[local_rank]).to(device)
            mstftd = DistributedDataParallel(mstftd, device_ids=[local_rank]).to(device)
            if is_distill:
                distiller = DistributedDataParallel(distiller, device_ids=[local_rank]).to(device)
        else:
            print(f'Process-{world_rank} FSDP init...')
            encoder = init_fsdp_model(model=encoder)
            quantizer = init_fsdp_model(model=quantizer)
            generator = init_fsdp_model(model=generator)
            mpd = init_fsdp_model(model=mpd)
            msd = init_fsdp_model(model=msd)
            mstftd = init_fsdp_model(model=mstftd)
            if is_distill:
                distiller = init_fsdp_model(model=distiller)
            
    adam_config = train_config['adam']
    params_list = [generator.parameters(), encoder.parameters(), quantizer.parameters(), distiller.parameters()]
    optim_g = torch.optim.AdamW(
        itertools.chain(*params_list),
        adam_config['learning_rate'],
        betas=[adam_config['adam_b1'], 
               adam_config['adam_b2']])
    optim_d = torch.optim.AdamW(
        itertools.chain(msd.parameters(), 
                        mpd.parameters(),
                        mstftd.parameters()),
        adam_config['learning_rate'],
        betas=[adam_config['adam_b1'], 
               adam_config['adam_b2']])
    if stu_state_dict_do is not None:
        print(f'Rank{world_rank} loading "generaotr-adam" "discriminator-adam" ckpt...')
        if mel_chs == stu_enc_input_chs:
            optim_d.load_state_dict(stu_state_dict_do['optim_d'])
            optim_g.load_state_dict(stu_state_dict_do['optim_g'])
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, 
        gamma=adam_config['lr_decay'], 
        last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, 
        gamma=adam_config['lr_decay'], 
        last_epoch=last_epoch)
    
    ret_dict = {
        "encoder": encoder,
        "quantizer": quantizer,
        "quantizer_type": quantizer_type,
        "generator": generator,
        "mpd": mpd,
        "msd": msd,
        "mstft": mstftd,
        "steps": steps,
        "last_epoch": last_epoch,
        "min_mel_error": min_mel_error,
        "optim_g": optim_g,
        "optim_d": optim_d,
        "scheduler_g": scheduler_g,
        "scheduler_d": scheduler_d
    }
    ret_dict = AttrDict(ret_dict)
    
    return ret_dict


def init_validation_dataset(valid_dataset: list, train_config: AttrDict):
    valid_sampler = DistributedSampler(valid_dataset, 
                                       shuffle=False) if train_config.num_gpus > 1 else None
    valid_loader = DataLoader(
        valid_dataset,
        num_workers=train_config.num_workers,
        shuffle=False,
        sampler=valid_sampler,
        batch_size=1,
        pin_memory=True,
        drop_last=True)

    return valid_loader, valid_sampler


def init_dataset(train_config: AttrDict, model_config: AttrDict, world_rank, device):
    mel_config = model_config['spec_transform']
    with open(train_config.training_filelist_path, 'r') as f:
        print(f'Rank{world_rank} loading audio file list...')
        training_filelist = json.load(fp=f)
        print(f'Number of file: {len(training_filelist)}')
    trainset = MelDataset(
        training_filelist,
        **mel_config,
        n_cache_reuse=0,
        shuffle=False if train_config.num_gpus > 1 else True,
        device=device,
        fine_tuning=train_config.fine_tuning,
        base_mels_path=train_config.mels_path)
    train_sampler = DistributedSampler(trainset) if train_config.num_gpus > 1 else None
    train_loader = DataLoader(
        trainset,
        num_workers=train_config.num_workers,
        shuffle=False,
        sampler=train_sampler,
        batch_size=train_config.batch_size,
        pin_memory=True,
        drop_last=True)

    '''if world_rank == 0:
        print(f'Rank{world_rank} loading validation files...')
        with open(train_config.validation_filelist_path, 'r') as f:
            validation_filelist = json.load(fp=f)
        validset = MelDataset(
            validation_filelist,
            **mel_config,
            split=False,
            shuffle=False,
            n_cache_reuse=0,
            device=device,
            fine_tuning=train_config.fine_tuning,
            base_mels_path=train_config.mels_path)
        validation_loader = DataLoader(
            validset,
            num_workers=1,
            shuffle=False,
            sampler=None,
            batch_size=1,
            pin_memory=True,
            drop_last=True)
    else:
        validation_loader = None'''
    print(f'Rank{world_rank} loading validation files...')
    with open(train_config.validation_filelist_path, 'r') as f:
        validation_filelist = json.load(fp=f)
    validset = MelDataset(
        validation_filelist,
        **mel_config,
        split=False,
        shuffle=False,
        n_cache_reuse=0,
        device=device,
        fine_tuning=train_config.fine_tuning,
        base_mels_path=train_config.mels_path)
    valid_dataloader, valid_sampler = init_validation_dataset(valid_dataset=validset, train_config=train_config)
        
    return train_loader, train_sampler, valid_dataloader, valid_sampler

