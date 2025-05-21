import multiprocessing as mp
import time

from pymcd.mcd import Calculate_MCD
from tqdm import tqdm


# instance of MCD class
# three different modes "plain", "dtw" and "dtw_sl" for the above three MCD metrics
mcd_toolbox = Calculate_MCD(MCD_mode="plain")


def mcd_mp(params: dict):
    mcd_score = mcd_toolbox.calculate_mcd(params['real_path'], params['gen_path'])

    return mcd_score


def compute_mcd(real2gen_pathes: list):    
    n_cpus = 20
    print(f'Number of CPUs: {n_cpus}')
    pool = mp.Pool(processes=n_cpus)

    job_list = []
    for real_path, gen_path in tqdm(real2gen_pathes, desc='Calculating MCD'):
        # two inputs w.r.t. reference (ground-truth) and synthesized speeches, respectively
        # 同样的元语音和生成语音对比
        # mcd_value = mcd_toolbox.calculate_mcd(real_path, gen_path)
        params = {
            'real_path': real_path,
            'gen_path': gen_path
        }
        job_list.append(params)
    begin = time.time()
    mcd_scores = pool.map(mcd_mp, job_list)
    elapsed = time.time() - begin
    print(f'Calculating MCD scores takes {elapsed: .2f}s')
    mean_mcd = sum(mcd_scores) / len(mcd_scores)
    
    return mean_mcd