import multiprocessing as mp
import time
import math

import librosa
import numpy as np
import torch
import torchaudio
import torchcrepe
from torchcrepe.loudness import REF_DB
from tqdm import tqdm

from evaluation.utils import load_audio

SILENCE_THRESHOLD = -60
UNVOICED_THRESHOLD = 0.21

"""
Periodicity metrics adapted from https://github.com/descriptinc/cargan
"""


def predict_pitch(
    audio: torch.Tensor, silence_threshold: float = SILENCE_THRESHOLD, unvoiced_treshold: float = UNVOICED_THRESHOLD
):
    """
    Predicts pitch and periodicity for the given audio.

    Args:
        audio (Tensor): The audio waveform.
        silence_threshold (float): The threshold for silence detection.
        unvoiced_treshold (float): The threshold for unvoiced detection.

    Returns:
        pitch (ndarray): The predicted pitch.
        periodicity (ndarray): The predicted periodicity.
    """
    # torchcrepe inference
    pitch, periodicity = torchcrepe.predict(
        audio,
        fmin=50.0,
        fmax=550,
        sample_rate=torchcrepe.SAMPLE_RATE,
        model="full",
        return_periodicity=True,
        device=audio.device,
        pad=False,
    )
    pitch = pitch.cpu().numpy()
    periodicity = periodicity.cpu().numpy()

    # Calculate dB-scaled spectrogram and set low energy frames to unvoiced
    hop_length = torchcrepe.SAMPLE_RATE // 100  # default CREPE
    stft = torchaudio.functional.spectrogram(
        audio,
        window=torch.hann_window(torchcrepe.WINDOW_SIZE, device=audio.device),
        n_fft=torchcrepe.WINDOW_SIZE,
        hop_length=hop_length,
        win_length=torchcrepe.WINDOW_SIZE,
        power=2,
        normalized=False,
        pad=0,
        center=False,
    )

    # Perceptual weighting
    freqs = librosa.fft_frequencies(sr=torchcrepe.SAMPLE_RATE, n_fft=torchcrepe.WINDOW_SIZE)
    perceptual_stft = librosa.perceptual_weighting(stft.cpu().numpy(), freqs) - REF_DB
    silence = perceptual_stft.mean(axis=1) < silence_threshold

    periodicity[silence] = 0
    pitch[periodicity < unvoiced_treshold] = torchcrepe.UNVOICED

    return pitch, periodicity


def calculate_periodicity_metrics(y: torch.Tensor, y_hat: torch.Tensor):
    """
    Calculates periodicity metrics for the predicted and true audio data.

    Args:
        y (Tensor): The true audio data.
        y_hat (Tensor): The predicted audio data.

    Returns:
        periodicity_loss (float): The periodicity loss.
        pitch_loss (float): The pitch loss.
        f1 (float): The F1 score for voiced/unvoiced classification
    """
    true_pitch, true_periodicity = predict_pitch(y)
    pred_pitch, pred_periodicity = predict_pitch(y_hat)

    true_voiced = ~np.isnan(true_pitch)
    pred_voiced = ~np.isnan(pred_pitch)

    periodicity_loss = np.sqrt(((pred_periodicity - true_periodicity) ** 2).mean(axis=1)).mean()

    # Update pitch rmse
    voiced = true_voiced & pred_voiced
    difference_cents = 1200 * (np.log2(true_pitch[voiced]) - np.log2(pred_pitch[voiced]))
    pitch_loss = np.sqrt((difference_cents ** 2).mean())

    # voiced/unvoiced precision and recall
    true_positives = (true_voiced & pred_voiced).sum()
    false_positives = (~true_voiced & pred_voiced).sum()
    false_negatives = (true_voiced & ~pred_voiced).sum()

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * precision * recall / (precision + recall)
    results = {
        'periodicity_loss': periodicity_loss,
        'pitch_loss': pitch_loss,
        'vu_f1': f1
    }

    return results


def calc_vuf1_mp(params: dict):
    result = calculate_periodicity_metrics(**params)

    return result


def compute_vuf1(real2gen_pathes: list):    
    '''n_cpus = 20
    print(f'Number of CPUs: {n_cpus}')
    pool = mp.Pool(processes=n_cpus)'''

    job_list = []
    vuf1_scores = []
    for real_path, gen_path in tqdm(real2gen_pathes, desc='Calculating V/U F1'):
        real_audio = load_audio(real_path).cuda()
        gen_audio = load_audio(gen_path).cuda()
        min_length = min(real_audio.shape[1], gen_audio.shape[1])
        params = {
            'y': real_audio[:, :min_length],
            'y_hat': gen_audio[:, :min_length]
        }
        # job_list.append(params)
        result = calculate_periodicity_metrics(**params)
        vuf1_scores.append(result)
    '''begin = time.time()
    vuf1_scores = pool.map(calc_vuf1_mp, job_list)
    elapsed = time.time() - begin
    print(f'Calculating VUF1 scores takes {elapsed: .2f}s')'''

    list_len = len(vuf1_scores)
    mean_periodicity_loss = sum([ele['periodicity_loss'] for ele in vuf1_scores]) / list_len
    mean_pitch_loss = sum([ele['pitch_loss'] for ele in vuf1_scores]) / list_len
    mean_vuf1 = sum([ele['pitch_loss'] for ele in vuf1_scores]) / list_len
    results = {
        'periodicity_loss': mean_periodicity_loss,
        'pitch_loss': mean_pitch_loss,
        'vu_f1': mean_vuf1
    }

    return results

