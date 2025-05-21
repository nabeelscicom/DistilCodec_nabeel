import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import time
import argparse
import json

import torch
import torch.nn.functional as F
from torch import distributed as dist

from utils.env import AttrDict
from models import init_multi_mel_transforms
from models import feature_loss
from models import generator_loss
from models import discriminator_loss
from utils import plot_spectrogram
from utils import print_cuda_info, setup, cleanup, print_rank_0
from utils import wandb_log, init_wandb_tb
from models import get_dataset_filelist
from train_utils import *
from evaluation import split_group_and_residual, calc_codebook_ppl_usage
from aipal_codec import AIPalCodec

torch.backends.cudnn.benchmark = True


def update_train_config_dist(world_rank: int, train_config: dict):
    if world_rank == 0:
        save_path = train_config['save_path']
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        train_config['checkpoint_path'] = os.path.join(save_path, 'ckpt')
        if not os.path.exists(train_config.checkpoint_path):
            os.mkdir(train_config.checkpoint_path)
        train_config['logger_path'] = os.path.join(train_config.checkpoint_path, 'logs')
        if not os.path.exists(train_config.logger_path):
            os.mkdir(train_config.logger_path)
        
        ckpt_name = scan_checkpoint(train_config['checkpoint_path'], 'g_')
        if ckpt_name is not None:
            train_config['load_path'] = train_config['checkpoint_path']
            print(f'Pretrained ckpt path: {train_config["load_path"]}')
        
        training_filelist_path = os.path.join(save_path, 'train_files.json')
        train_config['training_filelist_path'] = training_filelist_path
        if not os.path.exists(training_filelist_path):
            training_filelist, validation_filelist = get_dataset_filelist(train_config)
            with open(training_filelist_path, 'w') as f:
                json.dump(training_filelist, fp=f, indent=4, ensure_ascii=False)
        else:
            print('Training file list existed, escape loading files.')
            
        validation_filelist_path = os.path.join(save_path, 'valid_files.json')
        train_config['validation_filelist_path'] = validation_filelist_path
        if not os.path.exists(validation_filelist_path):
            with open(validation_filelist_path, 'w') as f:
                json.dump(validation_filelist, fp=f, indent=4, ensure_ascii=False)
        else:
            print('Validationg file list existed, escape loading files.')
            
        broadcast_objects = [train_config]
    else:
        broadcast_objects = [None]
    dist.broadcast_object_list(object_list=broadcast_objects, src=0)
    dist.barrier()
    
    return broadcast_objects


def train(local_rank, 
          world_rank, 
          train_config: dict, 
          model_config: dict, 
          is_debug: bool = False, 
          use_fsdp: bool = False,
          is_amp: bool = False):
    
    if is_debug:
        warnings.filterwarnings('ignore', 
                                category=UserWarning, 
                                module='torch.nn.functional')

    torch.cuda.manual_seed(train_config.seed)
    device = torch.device('cuda:{:d}'.format(local_rank))
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    
    broadcast_objects = update_train_config_dist(world_rank=world_rank,
                                                 train_config=train_config)
    train_config = AttrDict(broadcast_objects[0])
    use_fm_distill = train_config.distill['use_fm_distill']
    
    if world_rank == 0:
        sw = init_wandb_tb(train_config, model_config)
        print_rank_0(f'Model config: {json.dumps(model_config, indent=4)}')
        print_rank_0(f'Train config: {json.dumps(train_config, indent=4)}')
    
    if 'distill' in train_config:
        is_distill = train_config.distill['is_distill']
        quantizer_transfer = train_config.distill['quantizer_transfer'] if 'quantizer_transfer' in train_config.distill else True
    else:
        is_distill = False
        quantizer_transfer = True
        
    inited_models = init_model_and_optimizer(
        model_config=model_config,
        train_config=train_config,
        world_rank=world_rank,
        local_rank=local_rank,
        device=device,
        use_fsdp=use_fsdp,
        is_debug=is_debug,
        is_distill=is_distill,
        quantizer_transfer=quantizer_transfer
    )
    encoder = inited_models.encoder
    quantizer = inited_models.quantizer
    generator = inited_models.generator
    mpd = inited_models.mpd
    msd = inited_models.msd
    mstftd = inited_models.mstft
    steps = inited_models.steps
    last_epoch = inited_models.last_epoch
    cur_mel_error = min_mel_error = inited_models.min_mel_error
    quantizer_type = inited_models.quantizer_type
    scheduler_g = inited_models.scheduler_g
    scheduler_d = inited_models.scheduler_d
    optim_g = inited_models.optim_g
    optim_d = inited_models.optim_d

    mel_config = model_config['spec_transform']
    mel_transforms = init_multi_mel_transforms(mel_config, device)
    mel_sepc_trans1, mel_spec_trans2, mel_spec_trans3, _ = mel_transforms

    commitment_loss_factor = 10.0
    codebook_diversity_factor = 0.0
    print(f'Commitment loss factor: {commitment_loss_factor: .2f}')
    
    train_loader, train_sampler, validation_loader, valid_sampler = init_dataset(train_config=train_config,
                                                                  model_config=model_config,
                                                                  world_rank=world_rank,
                                                                  device=device)
    
    generator.train()
    encoder.train()
    quantizer.train()
    mpd.train()
    msd.train()
        
    plot_gt_once = False
    min_save_ckpt = False
    for epoch in range(max(0, last_epoch), train_config.training_epochs):
        if world_rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))
            print_cuda_info()
            
        if train_config.num_gpus > 1:
            train_sampler.set_epoch(epoch)
            
        for batch in train_loader:
            if world_rank == 0:
                start_b = time.time()
                
            x, y, _, y_mel = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_mel = y_mel.to(device, non_blocking=True)
            y = y.unsqueeze(1)

            # Discriminator
            # if steps % train_config.accumulation_steps == 0:
            #    optim_d.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=is_amp):
                encoded_feature = encoder(y_mel)
                if is_debug:
                    print(f"After encoder c.shape: {encoded_feature.shape}")
                    
                rvq_result = quantizer(encoded_feature)
                quantized = rvq_result.quantized
                codebook_diversity_loss = rvq_result.codebook_diversity_loss
                commit_loss = rvq_result.commitment_loss
                if quantizer_type == 'fsq':
                    codebook_loss = 0.0
                elif quantizer_type == 'grvq':
                    # commitment loss is not taken part in training, only for observing
                    codebook_loss = commit_loss * commitment_loss_factor + codebook_diversity_factor * codebook_diversity_loss
                else:
                    raise ValueError(f'Quantizer {quantizer_type} not supported now')
                if is_debug and world_rank == 0:
                    print(f"After quantizer q.shape: {quantized.shape}")
                    
                y_g_hat = generator(quantized)
                if is_debug and world_rank == 0:
                    print(f'Source Audio shape: {y.shape}')
                    print(f'Generated Audio shape: {y_g_hat.shape}')
                
                min_len = min((y.shape[-1], y_g_hat.shape[-1]))
                y = y[:, :, :min_len]
                y_g_hat = y_g_hat[:, :, :min_len]
                loss_time = F.l1_loss(y, y_g_hat)
                
                y_g_hat_mel = mel_sepc_trans1(y_g_hat.squeeze(1)).to(device)
            
                y_r_mel_1 = mel_spec_trans2(y.squeeze(1)).to(device)
                y_g_mel_1 = mel_spec_trans2(y_g_hat.squeeze(1)).to(device)
                
                y_r_mel_2 = mel_spec_trans3(y.squeeze(1)).to(device)
                y_g_mel_2 = mel_spec_trans3(y_g_hat.squeeze(1)).to(device)

                # MPD
                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
                # MSD
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
                # MSTFTD
                y_disc_r, _ = mstftd(y)
                y_disc_gen, _ = mstftd(y_g_hat.detach())

            loss_disc_f, _, _ = discriminator_loss(
                y_df_hat_r, y_df_hat_g)
            loss_disc_s, _, _ = discriminator_loss(
                y_ds_hat_r, y_ds_hat_g)
            loss_disc_stft, _, _ = discriminator_loss(
                y_disc_r, y_disc_gen)
            loss_disc_all = loss_disc_s + loss_disc_f + loss_disc_stft
            loss_disc_all_norm = loss_disc_all  / train_config.accumulation_steps
            loss_disc_all_norm.backward()
            if steps % train_config.accumulation_steps == 0:
                optim_d.step()

            # Generator
            # if steps % train_config.accumulation_steps == 0:
            #    optim_g.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=is_amp):
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
                _, fmap_stftd_r = mstftd(y)
                y_stftd_hat_g, fmap_stftd_g = mstftd(y_g_hat)

            # L1 Mel-Spectrogram Loss
            loss_mel1 = F.l1_loss(y_r_mel_1, y_g_mel_1)
            loss_mel2 = F.l1_loss(y_r_mel_2, y_g_mel_2)
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45 + loss_mel1 + loss_mel2

            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_fm_stft = feature_loss(fmap_stftd_r, fmap_stftd_g)

            loss_gen_f, _ = generator_loss(y_df_hat_g)
            loss_gen_s, _ = generator_loss(y_ds_hat_g)
            loss_gen_stft, _ = generator_loss(y_stftd_hat_g)

            loss_disc_fms = loss_fm_f + loss_fm_s + loss_fm_stft
            loss_disc_gens = loss_gen_s + loss_gen_f + loss_gen_stft
            loss_gen_all = loss_disc_gens + loss_disc_fms + loss_mel + codebook_loss
            loss_gen_all_norm = loss_gen_all / train_config.accumulation_steps
            if local_rank == 0 and is_debug:
                print_cuda_info()
            loss_gen_all_norm.backward()
            if local_rank == 0 and is_debug:
                print_cuda_info()
            if steps % train_config.accumulation_steps == 0:
                optim_g.step()
                optim_g.zero_grad()
                optim_d.zero_grad()
            
            dist.barrier()
            if world_rank == 0:
                # STDOUT logging
                if steps % train_config.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
                    print(
                        'Steps : {:d}, Gen Loss Total: {:4.3f}, Commit loss: {:4.3f}, Diversity loss: {:4.3f}, Desc Loss Total: {:4.3f}, Mel-Spec. Error: {:4.3f}, s/b: {:4.3f}'.
                        format(steps, 
                               loss_gen_all, 
                               commit_loss.tolist(),
                               codebook_diversity_loss.tolist(),
                               loss_disc_all, 
                               mel_error,
                               time.time() - start_b))
                    
                    wandb_step = steps
                    wandb_train_key_values = {
                        "Generator": {
                            "loss_all_gen": loss_gen_all.tolist(),
                            "loss_mel": loss_mel.tolist(),
                            "loss_mpd_gen": loss_gen_f.tolist(),
                            "loss_msd_gen": loss_gen_s.tolist(),
                            "loss_mstft_gen": loss_gen_stft.tolist(),
                            "loss_mpd_fm": loss_fm_f.tolist(),
                            "loss_msd_fm": loss_fm_s.tolist(),
                            "loss_stft_fm": loss_fm_stft.tolist(),
                            "mel_error": mel_error,
                            "time_error": loss_time.tolist(),
                            "vq_commitment_loss": codebook_loss.tolist(),
                            "codebook_diversity_loss": codebook_diversity_loss.tolist()
                        },
                        "Descriminator": {
                            "loss_all_disc": loss_disc_all.tolist(),
                            "loss_multi_period": loss_disc_f.tolist(),
                            "loss_multi_scale": loss_disc_s.tolist(),
                            "loss_multi_stft": loss_disc_stft.tolist()
                        }
                    }
                    wandb_log(wandb_train_key_values, wandb_step)

                    cur_mel_error = mel_error
                    if cur_mel_error < min_mel_error:
                        min_save_ckpt = True
                        min_mel_error = cur_mel_error
                        print(f'Current minimum mel error: {min_mel_error: .3f}')
                    else:
                        min_save_ckpt = False
                    
                # checkpointing
                if (steps % train_config.checkpoint_interval == 0 and steps != 0) or min_save_ckpt:
                    if min_save_ckpt:
                        print('Saving new minimum mel loss ckpt')
                        min_save_ckpt = False
                    else:
                        print('Saving ckpt')
                    checkpoint_path = "{}/g_{:08d}".format(train_config.checkpoint_path,
                                                           steps)
                    save_checkpoint(
                        checkpoint_path, {
                            'generator': 
                                ((generator.module if train_config.num_gpus > 1 else generator).state_dict()) 
                                if not use_fsdp else fsdp_state_dict(generator),
                            'encoder': 
                                ((encoder.module if train_config.num_gpus > 1 else encoder).state_dict()) 
                                if not use_fsdp else fsdp_state_dict(encoder),
                            'quantizer': 
                                ((quantizer.module if train_config.num_gpus > 1 else quantizer).state_dict()) 
                                if not use_fsdp else fsdp_state_dict(quantizer)
                        },
                        num_ckpt_keep=train_config.num_ckpt_keep)
                    
                    checkpoint_path = "{}/do_{:08d}".format(train_config.checkpoint_path,
                                                            steps)
                    save_checkpoint(
                        checkpoint_path, {
                            'mpd': 
                                ((mpd.module if train_config.num_gpus > 1 else mpd).state_dict()) 
                                if not use_fsdp else fsdp_state_dict(mpd),
                            'msd': 
                                ((msd.module if train_config.num_gpus > 1 else msd).state_dict()) 
                                if not use_fsdp else fsdp_state_dict(msd),
                            'mstftd': 
                                ((mstftd.module if train_config.num_gpus > 1 else mstftd).state_dict()) 
                                if not use_fsdp else fsdp_state_dict(mstftd),
                            'optim_g': optim_g.state_dict(),
                            'optim_d': optim_d.state_dict(),
                            'steps': steps,
                            'epoch': epoch,
                            "min_mel_error": min_mel_error
                        },
                        num_ckpt_keep=train_config.num_ckpt_keep)
                # Tensorboard summary logging
                if steps % train_config.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)

            # Validation
            if steps % train_config.validation_interval == 0 and steps != 0:
                if world_rank == 0:
                    valid_sampler.set_epoch(epoch)

                print_rank_0(f'Validation step-{steps // train_config.validation_interval}')
                generator.eval()
                encoder.eval()
                quantizer.eval()
                torch.cuda.empty_cache()
                val_err_tot = 0
                valid_begin = time.time()
                all_codes = []
                unique_indices = {}
                with torch.no_grad():
                    for j, batch in enumerate(validation_loader):
                        x, y, _, y_mel = batch
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=is_amp):
                            encoded_feature = encoder(y_mel.to(device))
                            rvq_result = quantizer(encoded_feature)
                            quantized = rvq_result.quantized
                            y_g_hat = generator(quantized)

                        code_per_codebook, ngroups, nresiduals = split_group_and_residual(rvq_result.codes)
                        all_codes.append(code_per_codebook)

                        y_mel = torch.autograd.Variable(y_mel.to(device))
                        y_g_hat_mel = mel_sepc_trans1(y_g_hat.squeeze(1))
                        y_g_hat_mel = y_g_hat_mel.to(y_mel.device)
                        i_size = min(y_mel.size(2), y_g_hat_mel.size(2))
                        val_err_tot += F.l1_loss(y_mel[:, :, :i_size], y_g_hat_mel[:, :, :i_size])

                        if j < 40 and world_rank == 0:
                            if not plot_gt_once:
                                sw.add_audio(f'gt/y_{j}', y[0], steps, mel_config['sampling_rate'])
                                sw.add_figure(f'gt/y_spec_{j}', plot_spectrogram(x[0]), steps)
                            sw.add_audio(f'generated/y_hat_{j}', y_g_hat[0], steps, mel_config['sampling_rate'])
                            y_hat_spec = mel_sepc_trans1(y_g_hat.squeeze(1))
                            mel_fig = plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy())
                            sw.add_figure(f'generated/y_hat_spec_{j}', mel_fig, steps)

                    for ci, codes_z in enumerate(zip(*all_codes)):
                        codes_m = torch.cat(codes_z)
                        cur_group, cur_residual = ci // ngroups, ci % ngroups
                        codebook_ppl, codebook_usage, uni_indices = calc_codebook_ppl_usage(indices=codes_m, 
                                                                                            codebook_size=model_config['quantizer']['codebook_size'], 
                                                                                            only_calc_usage=True)
                        unique_indices[f'g{cur_group}r{cur_residual}'] = uni_indices
                    unique_list = [None] * dist.get_world_size()
                    dist.gather_object(
                        unique_indices,
                        unique_list if dist.get_rank() == 0 else None,
                        dst=0)

                    val_err = val_err_tot / (j + 1)
                    print_rank_0(f'Befor reduce sum valid mel error: {val_err}')
                    dist.reduce(val_err, dst=0, op=dist.ReduceOp.SUM)
                    print_rank_0(f'After reduce sum valid mel error: {val_err}')
                    reduced_val_err = val_err / dist.get_world_size()
                    print_rank_0(f'Mean valid mel error: {reduced_val_err}')

                    if world_rank == 0:
                        cb2usage = {}
                        usage = []
                        unique_indices_all = {}
                        all_keys = unique_indices.keys()
                        for key in all_keys:
                            all_uni_inds_list = [torch.tensor(ele[key], dtype=torch.int64) for ele in unique_list]
                            all_uni_inds = torch.cat(all_uni_inds_list).unique().tolist()
                            usage_t = len(all_uni_inds) / model_config['quantizer']['codebook_size']
                            usage.append(usage_t)
                            cb2usage[key] = usage_t
                            unique_indices_all[key] = all_uni_inds
                        mean_usage = sum(usage) / len(usage)
                        usage_path = os.path.join(train_config.logger_path, f'unique_indices_step{steps}.json')
                        with open(usage_path, 'w') as f:
                            json.dump(unique_indices_all, f, ensure_ascii=False, indent=4)

                        sw.add_scalar("validation/mel_spec_error", reduced_val_err.item(), steps)
                        sw.add_scalar("validation/mean_usage", mean_usage, steps)
                        wandb_valid_key_values = {
                            "Validation": {
                                "mel_spec_error": reduced_val_err.item(),
                                "mean_codebook_usage": mean_usage
                            }
                        }
                        wandb_log(wandb_valid_key_values, wandb_step)
                        print(f'Validation mel error: {reduced_val_err}\nMean Codebook usage: {mean_usage}')
                    
                    if not plot_gt_once:
                        plot_gt_once = True
                dist.barrier()
                valid_end = time.time()
                print_rank_0(f'Validation takes: {valid_end - valid_begin: .2f}s')

                generator.train()
                encoder.train()
                quantizer.train()

            """if steps % train_config.reset_codebook_interval == 0 and world_rank == 0:
                codec = AIPalCodec(configs=model_config, only_quantizer=True)
                codec.quantizer.load_state_dict((quantizer.module if train_config.num_gpus > 1 else quantizer).state_dict())
                codec.reset_codebook(unique_indice=unique_indices,
                                     save_path=None,
                                     topK=1)
                quantizer.load_state_dict(codec.quantizer.state_dict())
                checkpoint_path = "{}/g_{:08d}".format(train_config.checkpoint_path, steps)
                save_checkpoint(
                    checkpoint_path, {
                        'generator': 
                            ((generator.module if train_config.num_gpus > 1 else generator).state_dict()) 
                            if not use_fsdp else fsdp_state_dict(generator),
                        'encoder': 
                            ((encoder.module if train_config.num_gpus > 1 else encoder).state_dict()) 
                            if not use_fsdp else fsdp_state_dict(encoder),
                        'quantizer': 
                            ((quantizer.module if train_config.num_gpus > 1 else quantizer).state_dict()) 
                            if not use_fsdp else fsdp_state_dict(quantizer)
                    },
                    num_ckpt_keep=train_config.num_ckpt_keep)
                checkpoint_path = "{}/do_{:08d}".format(train_config.checkpoint_path, steps)
                save_checkpoint(
                        checkpoint_path, {
                            'mpd': 
                                ((mpd.module if train_config.num_gpus > 1 else mpd).state_dict()) 
                                if not use_fsdp else fsdp_state_dict(mpd),
                            'msd': 
                                ((msd.module if train_config.num_gpus > 1 else msd).state_dict()) 
                                if not use_fsdp else fsdp_state_dict(msd),
                            'mstftd': 
                                ((mstftd.module if train_config.num_gpus > 1 else mstftd).state_dict()) 
                                if not use_fsdp else fsdp_state_dict(mstftd),
                            'optim_g': optim_g.state_dict(),
                            'optim_d': optim_d.state_dict(),
                            'steps': steps,
                            'epoch': epoch,
                            "min_mel_error": min_mel_error
                        },
                        num_ckpt_keep=train_config.num_ckpt_keep)
            if steps % train_config.reset_codebook_interval == 0:
                g_stat_dict = AIPalCodec.scan_checkpoint(train_config.load_path, 'g_')
                stu_state_dict_g = load_checkpoint(g_stat_dict, device)
                (generator.module if train_config.num_gpus > 1 else generator).load_state_dict(stu_state_dict_g['generator'])
                (encoder.module if train_config.num_gpus > 1 else encoder).load_state_dict(stu_state_dict_g["encoder"])
                (quantizer.module if train_config.num_gpus > 1 else quantizer).load_state_dict(stu_state_dict_g["quantizer"])
            dist.barrier()"""

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if world_rank == 0:
            print(f'Time taken for epoch {epoch + 1} is {int(time.time() - start)} sec\n')


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='')
    parser.add_argument('--train_config', default='')
    parser.add_argument("--fsdp", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()
    
    setup()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    pid = os.getpid()
    print(f'current pid: {pid}, rank: {rank}, local rank: {local_rank}')
    
    with open(args.model_config) as f:
        model_config = json.loads(f.read())
        model_config = AttrDict(model_config)
        
    with open(args.train_config) as f:
        train_config = json.loads(f.read())
        train_config = AttrDict(train_config)

    torch.manual_seed(train_config.seed)
    if torch.cuda.is_available():
        print_rank_0(f'Total Cuda Devices is {dist.get_world_size()}')
        torch.cuda.manual_seed(train_config.seed)
        train_config.num_gpus = torch.cuda.device_count()
        train_config.batch_size = int(train_config.batch_size / train_config.num_gpus)
        print_rank_0(f'Batch size per GPU :{train_config.batch_size}')

    if args.amp:
        print(f'Rank{rank}: Auto mixed precision enabled')

    train(local_rank, 
          rank, 
          train_config, 
          model_config, 
          is_debug=args.debug, 
          use_fsdp=args.fsdp,
          is_amp=args.amp)
        
    cleanup()


if __name__ == '__main__':
    main()
