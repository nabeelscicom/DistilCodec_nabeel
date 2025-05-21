import os
import json

import torch
from tqdm import tqdm

from aipal_codec import AIPalCodec
from evaluation.si_snr import compute_sisnr
from evaluation.stoi import compute_stoi
from evaluation.mcd import compute_mcd
from evaluation.pesq import compute_pesq
from evaluation.utmos import compute_utmos
from evaluation.codebook_analysis import calc_codebook_ppl_usage, split_group_and_residual
from models import get_validation_files


def split_list(lst, size):
    return [lst[i:i+size] for i in range(0, len(lst), size)]
    

def reconstruct_audio(valid_config: str, 
                      ckpt_path: str, 
                      ckpt_config: str, 
                      batch_size: int = 1, 
                      skip_decoding: bool = False,
                      enable_bfloat16: bool = False):
    
    n_valid_files = valid_config.n_valid_files
    only_calc_usage = valid_config.only_calc_usage
    evaluation_name = None if 'eval_name' not in valid_config else valid_config['eval_name'] 
    audio_list = get_validation_files(valid_config, 
                                      total_validation_number=n_valid_files,
                                      is_shuffle=False)
    batch_audio = split_list(audio_list, batch_size)
    codec = AIPalCodec.from_pretrained(config_path=ckpt_config,
                                       model_path=ckpt_path,
                                       load_steps=valid_config.ckpt_steps,
                                       is_debug=False,
                                       use_generator=True).eval()
    # special_audio_tokens = codec.get_codebook()['special_audio_tokens']
    # print(f'Special Tokens:\n{json.dumps(special_audio_tokens, indent=4)}')

    dir_path = valid_config.audio_path
    print(f'Evaluation directory: {dir_path}')
    
    eval_path = os.path.join(dir_path, 'evaluation')
    if not os.path.exists(eval_path):
        os.mkdir(eval_path)
    audio_files_path = os.path.join(eval_path, 'valid_files.json')
    with open(audio_files_path, 'w') as f:
        json.dump(audio_list, f, ensure_ascii=False, indent=4)
    if 'eval_name' not in valid_config:
        dir_name = f'gen_audios_{n_valid_files}'
    else:
        dir_name = f'gen_audios_{evaluation_name}'
    audio_path = os.path.join(eval_path, dir_name)
    if not os.path.exists(audio_path):
        os.mkdir(audio_path)
    gen_audio_pathes = []
    all_codes = []
    for batch in tqdm(batch_audio, desc='Audio reconstruction'):
        with torch.no_grad():
            quantized_ret, gen_time_lengths, _ = codec.encode(batch, enable_bfloat16)
            print(quantized_ret.codes.size(), quantized_ret.codes.dtype,  quantized_ret.quantized.size())

            # codec.quantizer.decode()
            code_per_codebook, ngroups, nresiduals = split_group_and_residual(quantized_ret.codes)
            all_codes.append(code_per_codebook)

            # new_codes = quantized_ret.codes.squeeze().unsqueeze(0).unsqueeze(0).unsqueeze(0)
            # print(new_codes.size())
            re_features = codec.quantizer.decode(indices=quantized_ret.codes)
            # print(re_features.size())
            audio_names = [os.path.split(p)[1] for p in batch]
            if skip_decoding:
                for name_t in audio_names:
                    audio_path_t = os.path.join(audio_path, name_t)
                    gen_audio_pathes.append(audio_path_t)
            else:
                y_gen = codec.decode_from_features(quantized_features=re_features, # quantized_ret.quantized,
                                     enable_bfloat16=enable_bfloat16)
                gen_audio_pathes_t = codec.save_wav(
                    audio_gen_batch=y_gen, 
                    nhop_lengths=gen_time_lengths, 
                    audio_names=audio_names, 
                    save_path=audio_path
                )
                gen_audio_pathes.extend(gen_audio_pathes_t)
    
    perplexities = []
    usage = []
    codebook_size = codec.quantizer_config.codebook_size
    unique_indices = {}
    for ci, codes_z in enumerate(zip(*all_codes)):
        codes_m = torch.cat(codes_z)
        cur_group, cur_residual = ci // ngroups, ci % ngroups
        codebook_ppl, codebook_usage, uni_indices = calc_codebook_ppl_usage(indices=codes_m, codebook_size=codebook_size, only_calc_usage=only_calc_usage)
        perplexities.append(codebook_ppl)
        usage.append(codebook_usage)
        unique_indices[f'g{cur_group}r{cur_residual}'] = uni_indices
    codebook_eval = {
        "Perplexity": {
            "mean": sum(perplexities) / len(perplexities),
            "max": max(perplexities),
            "min": min(perplexities)
        },
        "Utilization": {
            "mean": sum(usage) / len(usage),
            "max": max(usage),
            "min": min(usage)
        }
    }
    print(f'Codebook PPL: mean-{sum(perplexities) / len(perplexities): .4f} max-{max(perplexities): .4f} min-{min(perplexities): .4f}')
    print(f'Codebook Utilization: mean-{sum(usage) / len(usage): .4f} max-{max(usage): .4f} min-{min(usage): .4f}')
            
    real2gen_pathes = list(zip(audio_list, gen_audio_pathes))
    
    return real2gen_pathes, eval_path, codec.ckpt_step, codebook_eval, evaluation_name, unique_indices


def write_eval_result(eval_path, ckpt_step, eval_results, task_name: str = None):
    eval_name = f'eval_step{ckpt_step}.json' if task_name is None else f'{task_name}_{ckpt_step}.json'
    eval_result_path = os.path.join(eval_path, eval_name)
    with open(eval_result_path, 'w') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=4)
        print(f'Writed evaluation results to: {eval_result_path}')


def evaluation(valid_config: str, 
               ckpt_path: str, 
               config_path, 
               skip_decoding: bool = False,
               enable_bfloat16: bool = False):
    
    eval_results = dict()
    real2gen_pathes, eval_path, ckpt_step, codebook_eval, task_name, unique_indices = reconstruct_audio(
        valid_config, 
        ckpt_path, 
        config_path,
        skip_decoding=skip_decoding,
        enable_bfloat16=enable_bfloat16
    )
    eval_results['codebook_eval'] = codebook_eval
    write_eval_result(eval_path, ckpt_step, unique_indices, task_name='unique_indices')
    print(ckpt_step)
    
    # 评估SI-SNR
    snr_result = compute_sisnr(real2gen_pathes)
    eval_results["SI-SNR"] = snr_result
    print(f'SI-SNR: {snr_result}')
    
    # 评估STOI
    stoi_result = compute_stoi(real2gen_pathes)
    eval_results['STOI'] = stoi_result
    print(f'STOI: {stoi_result}')

    # 评估UTMOS
    utmos_result = compute_utmos(real2gen_pathes)
    eval_results['UTMOS'] = utmos_result
    print(f'UTMOS: {utmos_result}')
    write_eval_result(eval_path, ckpt_step, eval_results, task_name=task_name)

    # 评估MCD
    mcd_result = compute_mcd(real2gen_pathes)
    eval_results['MCD'] = mcd_result
    print(f'MCD: {mcd_result}')
    write_eval_result(eval_path, ckpt_step, eval_results, task_name=task_name)

    # 评估PESQ
    pesq_result = compute_pesq(real2gen_pathes)
    eval_results['PESQ'] = pesq_result
    print(f'PESQ: {pesq_result}')
    write_eval_result(eval_path, ckpt_step, eval_results, task_name=task_name)
