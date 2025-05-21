import torch

from models import load_wav


def load_audio(audio_path: str, sampling_rate: int = 24000):
    audio, _ = load_wav(audio_path, 
                        sr=sampling_rate)
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
        
    return audio