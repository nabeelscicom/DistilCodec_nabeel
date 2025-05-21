import math

import torch
from tqdm import tqdm

from evaluation.utils import load_audio


def si_snr(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """
    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    
    references_projection = (x_zm**2).sum(dim=-1) + eps
    references_on_estimates = (s_zm * x_zm).sum(dim=-1) + eps
    
    scale = references_on_estimates / references_projection
    e_true = scale * x
    e_res = s - e_true

    signal = (e_true**2).sum(dim=-1)
    noise = (e_res**2).sum(dim=-1)
    sdr = -10 * torch.log10(signal / noise + eps)
    sdr = sdr.mean()

    return sdr


def compute_sisnr(real2gen_pathes: list):
    ret = []
    for real_path, gen_path in tqdm(real2gen_pathes, desc='Calculating SI-SNR'):
        real_audio = load_audio(real_path)
        gen_audio = load_audio(gen_path)
        min_length = min(real_audio.shape[1], gen_audio.shape[1])
        cal_sisnr = si_snr(x=real_audio[:, :min_length], 
                           s=gen_audio[:, :min_length])
        ret.append(cal_sisnr.tolist())
    
    sisnr_score = [num for num in ret if (num is not None) and (not math.isnan(num))]
    mean_sisnr = sum(sisnr_score) / len(sisnr_score)
    
    return mean_sisnr