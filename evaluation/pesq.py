import math
import time
import multiprocessing as mp

from pesq import pesq
from tqdm import tqdm
import torchaudio

from evaluation.utils import load_audio


def pesq_mp(params: dict):
    try:
        pesq_score = pesq(params['sr'], params['ref'], params['deg'])
    except:
        return None

    return pesq_score


def compute_pesq(real2gen_pathes: list):
    n_cpus = 20
    print(f'Number of CPUs: {n_cpus}')
    pool = mp.Pool(processes=n_cpus)

    sample_rate = 24000
    job_list = []
    for real_path, gen_path in tqdm(real2gen_pathes, desc='Calculating PESQ'):
        real_audio = load_audio(real_path)
        gen_audio = load_audio(gen_path)
        min_length = min(real_audio.shape[1], gen_audio.shape[1])
        if min_length < 2:
            continue
        real_audio = real_audio[:, :min_length]
        gen_audio = gen_audio[:, :min_length]

        sr_16k = 16000
        ref_audio_16k = torchaudio.functional.resample(
            real_audio, sr_16k, sample_rate
        ).squeeze().cpu().numpy()
        gen_audio_16k = torchaudio.functional.resample(
            gen_audio, sr_16k, sample_rate
        ).squeeze().cpu().numpy()
        params = {
            'sr': sr_16k,
            'ref': ref_audio_16k,
            'deg': gen_audio_16k
        }
        job_list.append(params)
    begin = time.time()
    pesq_scores = pool.map(pesq_mp, job_list)
    elapsed = time.time() - begin
    print(f'Calculating PESQ scores takes {elapsed: .2f}s')
    pesq_scores = [num for num in pesq_scores if (num is not None) and (not math.isnan(num))]
    mean_wb_pesq = sum(pesq_scores) / len(pesq_scores)
    
    return mean_wb_pesq

