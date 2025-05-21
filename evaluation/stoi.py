import torch
from torch import nn
from torch_stoi import NegSTOILoss
from tqdm import tqdm
import math

from evaluation.utils import load_audio


def compute_stoi(real2gen_pathes: list):
    sample_rate = 24000
    loss_func = NegSTOILoss(sample_rate=sample_rate)
    
    ret = []
    for real_path, gen_path in tqdm(real2gen_pathes, desc='Calculating STOI'):
        try:
            real_audio = load_audio(real_path)
            gen_audio = load_audio(gen_path)
            min_length = min(real_audio.shape[1], gen_audio.shape[1])
            if min_length < 2:
                continue
            stoi_loss = loss_func(gen_audio[:, :min_length], 
                                real_audio[:, :min_length])
            ret.append(stoi_loss.tolist()[0])
        except: 
            print(f'Data error: {gen_path}')
            continue
    
    ret = [num for num in ret if not math.isnan(num)]
    mean_stoi = sum(ret) / len(ret)
    
    return -mean_stoi
    

if __name__ == '__main__':
    sample_rate = 16000
    loss_func = NegSTOILoss(sample_rate=sample_rate)
    # Your nnet and optimizer definition here
    nnet = nn.Module()

    noisy_speech = torch.randn(2, 16000)
    clean_speech = torch.randn(2, 16000)
    # Estimate clean speech
    est_speech = nnet(noisy_speech)
    # Compute loss and backward (then step etc...)
    loss_batch = loss_func(est_speech, clean_speech)
    loss_batch.mean().backward()
