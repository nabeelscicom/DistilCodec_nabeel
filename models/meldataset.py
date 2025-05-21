# code based on https://github.com/b04901014/MQTTS
import math
import os
import random
import time
from multiprocessing import Pool
from itertools import zip_longest

import librosa
import numpy as np
import torch.utils.data

from models.mel_spec import LogMelSpectrogram
from models.utils import get_third_level_directories, get_files_in_directory
from utils.env import AttrDict


def load_wav(full_path, sr):
    wav, sr = librosa.load(full_path, sr=sr)
    return wav, sr


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


def get_all_files_mp(directory):
    # Step 1: 获取所有第三级目录
    begin = time.time()
    third_level_dirs = get_third_level_directories(directory)
    print(f'Number of third level directory is: {len(third_level_dirs)}, takes {time.time() - begin}')
    
    # Step 2: 使用多进程遍历第三级目录及其子目录中的文件
    nprocess = 4
    with Pool(nprocess) as pool:
        results = pool.map(get_files_in_directory, third_level_dirs)
    
    # Step 3: 获取根目录及一级、二级目录中的文件
    root_files = []
    for root, _, files in os.walk(directory):
        depth = root[len(directory):].count(os.sep)
        if depth < 3:  # 只处理根目录及一二级目录
            for file in files:
                root_files.append(os.path.join(root, file))
    
    # Step 4: 合并所有文件列表
    all_files = []
    all_files.extend(root_files)
    for filelist in results:
        all_files.extend(filelist)
    
    return all_files


def get_all_files_path(directory, desc, is_mp=False, suffix: str = '.wav'):
    print(desc)
    start = time.time()
    if is_mp:
        file_paths = get_all_files_mp(directory)
    else:
        file_paths = []  # 存储所有文件路径的列表
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(suffix):
                    file_paths.append(os.path.join(root, file))  # 将文件路径添加到列表中
    elapsed = time.time() - start 
    print(f'Scanning files of 【{directory}】 takes {elapsed: .2f}s')
        
    return file_paths


def get_training_files(training_config, suffix: str = '.wav'):
    training_files = []
    training_files_pathes = training_config['training_files_path']
    if isinstance(training_files_pathes, dict):
        replay_training_file_pathes = training_files_pathes['replay_training_file_pathes']
        for i, ele in enumerate(replay_training_file_pathes):
            path_t = ele['path']
            replay_rate_t = ele['replay_rate']
            training_files_t = get_all_files_path(path_t,
                                                  desc=f'Scanning replay path: {path_t}, Replay-Rate: {replay_rate_t}',
                                                  suffix=suffix)
            random.shuffle(training_files_t)
            random.shuffle(training_files_t)
            sample_len = int(len(training_files_t) * replay_rate_t)
            training_files.extend(training_files_t[:sample_len])
            print(f'After replay step{i + 1} number of replay files: {sample_len}, total number: {len(training_files)}')

        current_training_file_pathes = training_files_pathes['current_training_file_pathes']
        if isinstance(current_training_file_pathes, list):
            training_file_pathes = current_training_file_pathes
        elif isinstance(current_training_file_pathes, str):
            training_file_pathes = [current_training_file_pathes]
        else:
            raise ValueError('Config "current_training_file_pathes" should be one of [string, list of string]')
        for p in training_file_pathes:
            training_files_t = get_all_files_path(p,
                                                  desc=f"Scanning Training Data: {p}",
                                                  suffix=suffix)
            training_files.extend(training_files_t)
            print(f'After scanning {p}, Number of files: {len(training_files_t)}')
    elif isinstance(training_file_pathes, list):
        for p in training_file_pathes:
            training_files_t = get_all_files_path(p,
                                                  desc=f"Scanning Training Data: {path_t}",
                                                  suffix=suffix)
            training_files.extend(training_files_t)
            print(f'After scanning {p} number of files: {len(training_files_t)}, total number: {len(training_files)}')
    elif isinstance(training_file_pathes, str):
        training_files_t = get_all_files_path(p,
                                              desc=f"Scanning Training Data: {path_t}",
                                              suffix=suffix)
        training_files.extend(training_files_t)
    else:
        raise ValueError(f'Training pathes configuration error')
    
    random.shuffle(training_files)
            
    return training_files


def interleave_arrays(*arrays):
    # 使用 zip_longest 处理不等长的数组
    interleaved = []
    for elements in zip_longest(*arrays, fillvalue=None):
        # 过滤掉 None 值
        interleaved.extend([e for e in elements if e is not None])
        
    return interleaved


def get_validation_files(training_config, 
                         total_validation_number=500,
                         is_shuffle=True,
                         suffix: str = '.wav'):
    validation_files = []
    valid_files_path = training_config['validation_files_path']
    if isinstance(valid_files_path, list):
        files_list = []
        for i, p in enumerate(valid_files_path):
            valid_files_t = get_all_files_path(p,
                                               desc=f'Scanning validation file path: {p}',
                                               suffix=suffix)
            if i > 0 and is_shuffle:
                random.shuffle(valid_files_t)
            files_list.append(valid_files_t)
        validation_files = interleave_arrays(*files_list)
    elif isinstance(valid_files_path, str):
        validation_files = get_all_files_path(valid_files_path,
                                              desc=f'Scanning validation file path: {valid_files_path}',
                                              suffix=suffix)
    else:
        raise ValueError(f'Validation pathes configuration error')
    
    return validation_files[:total_validation_number]


def get_dataset_filelist(training_config, suffix: str = '.wav'):
    training_files = get_training_files(training_config, suffix=suffix)
    print(f'Training files: {len(training_files)}')
    
    validation_files = get_validation_files(training_config, suffix=suffix)
    print(f'Validation files: {len(validation_files)}')
        
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 training_files,
                 segment_size,
                 n_fft,
                 num_mels,
                 hop_size,
                 win_size,
                 sampling_rate,
                 fmin,
                 fmax,
                 split=True,
                 shuffle=True,
                 n_cache_reuse=1,
                 device=None,
                 fmax_loss=None,
                 fine_tuning=False,
                 base_mels_path=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path
        self.mel_transform = LogMelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=self.n_fft,
            win_length=self.win_size,
            hop_length=self.hop_size,
            n_mels=self.num_mels,
            f_min=self.fmin,
            f_max=self.fmax)

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            try:
                # Note by yuantian: load with the sample_rate of config
                audio, sampling_rate = load_wav(filename, sr=self.sampling_rate)
            except Exception as e:
                print(f"Error on audio: {filename}")
                audio = np.random.normal(size=(self.sampling_rate, )) * 0.05
                sampling_rate = self.sampling_rate
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start +
                                  self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (
                        0, self.segment_size - audio.size(1)), 'constant')

            mel = self.mel_transform(audio)
        else:
            mel = np.load(
                os.path.join(self.base_mels_path,
                             os.path.splitext(os.path.split(filename)[-1])[0] +
                             '.npy'))
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0,
                                               mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(
                        mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (
                        0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (
                        0, self.segment_size - audio.size(1)), 'constant')

        mel_loss = self.mel_transform(audio)

        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)
    
    
def init_multi_mel_transforms(mel_config: dict, device):
    mel_config = AttrDict(mel_config)
    
    n_ffts = [mel_config.n_fft, mel_config.n_fft * 2, mel_config.n_fft // 2, mel_config.n_fft // 4]
    hop_lens = [mel_config.hop_size, mel_config.hop_size * 2, mel_config.hop_size // 2, mel_config.hop_size // 4]
    win_lens = [mel_config.win_size, mel_config.win_size * 2, mel_config.win_size // 2, mel_config.win_size // 4]
    
    mel_transforms = []
    for n_fft, hop_length, win_length in zip(n_ffts, hop_lens, win_lens):
        mel_transform = LogMelSpectrogram(
                sample_rate=mel_config.sampling_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=mel_config.num_mels,
                f_min=mel_config.fmin,
                f_max=mel_config.fmax) # .to(device)
        mel_transforms.append(mel_transform)
        
    return mel_transforms
