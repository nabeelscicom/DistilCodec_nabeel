# DistilCodec
The Joint Laboratory of International Digital Economy Academy (IDEA) and Emdoor, in collaboration with Emdoor Information Technology Co., Ltd., has launched DistilCodec - A Single-Codebook Neural Audio Codec with 32768 codes trained on uniersal audio.



[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2408.16532)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20DistilCodec-Models-blue)](https://huggingface.co/IDEA-Emdoor/DistilCodec-v1.0)

# 🔥 News
- *2025.05.27*: We release DistilCodec-v1.0 checkpoint on [huggingface](https://huggingface.co/IDEA-Emdoor/DistilCodec-v1.0).
- *2025.05.26*: We release the code of DistilCodec-v1.0, including training and inference.
- *2025.05.22*: We release UniTTS and DistilCodec on [arxiv](https://arxiv.org/abs/2408.16532).

## Introduction of DistilCodec
### Training Schema

### Evaluation
The second row of the table demonstrates the codebook utilization and perplexity (PPL) of DistilCodec evaluated on LibriSpeech-Test-Clean. Given DistilCodec's capability to process universal audio, we have constructed an integrated test set comprising speech, audiobook, and music samples for evaluating codebook utilization and PPL in universal audio scenarios. As shown in the table, DistilCodec achieves near-optimal codebook utilization (approaching 100%) across both datasets, accompanied by notably high PPL values (the theoretical maximum PPL equals the codebook size, which is 32,768). These results substantiate DistilCodec's superior audio reconstruction capabilities in universal audio applications.
| Dataset              | Codebook Usage(%)↑ | Codebook PPL↑ |
|-----------------------|---------------------|---------------|
| LibriSpeech-Clean-Test| 98.2                | 21660.5       |
| Universal-Audio-Test  | 99.9                | 26999.0       |
Additionally, we conducted a comprehensive comparative analysis of DistilCodec’s speech reconstruction capabilities using the LibriSpeech-Clean-Test benchmark. 
| Model             | Codebook Size | Nq | Token Rate (TPS) | Bandwidth (bps) | STOI ↑ | PESQ ↑ | UTMOS ↑ |
|-------------------|---------------|----|------------------|----------------|--------|--------|--------|
| Encodec           | 1024          | 8  | 600              | 6000           | 0.94   | 2.75   | 3.07   |
| DAC               | 1024          | 12 | 600              | 6000           | 0.95   | 4.01   | 4.00   |
| Encodec           | 1024          | 2  | 150              | 1500           | 0.84   | 1.56   | 1.58   |
| Mimi              | 2048          | 8  | 100              | 1100           | 0.91   | 2.25   | 3.56   |
| BigCodec          | 8192          | 1  | 80               | 1040           | 0.94   | 2.68   | 4.11   |
| DAC               | 1024          | 2  | 100              | 1000           | 0.73   | 1.14   | 1.29   |
| SpeechTokenizer   | 1024          | 2  | 100              | 1000           | 0.77   | 1.25   | 2.28   |
| X-codec           | 1024          | 2  | 100              | 1000           | 0.86   | 2.33   | 4.21   |
| WavTokenizer      | 4096          | 1  | 75               | 900            | 0.89   | 2.14   | 3.94   |
| X-codec2          | 65536         | 1  | 50               | 800            | 0.92   | 2.43   | 4.13   |
| StableCodec       | 15625         | 2  | 50               | 697            | 0.91   | 2.24   | 4.23   |
| Single-Codec      | 8192          | 1  | 23.4             | 304            | 0.86   | 1.88   | 3.72   |
| BiCodec           | 8192          | 1  | 50               | 650            | 0.92   | 2.51   | 4.18   |
| DistilCodec       | 32768         | 1  | 93               | 1300           | 0.93   | 2.02   | 3.75   |

### Demonstraion of reconstruction samples

## Installation of DistilCodec
-*Step1*: Create conda environment for DistilCodec.
```bash
conda create -n distilcodec python=3.10
conda activate distilcodec
```
-*Step2*: install requirements.
```bash
pip install requirements.txt
```


## Infer

### Part1: Reconstruct audio from raw wav

```python

from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer


device=torch.device('cpu')

config_path = "./configs/xxx.yaml"
model_path = "./xxx.ckpt"
audio_outpath = "xxx"

wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device)


wav, sr = torchaudio.load(audio_path)
wav = convert_audio(wav, sr, 24000, 1) 
bandwidth_id = torch.tensor([0])
wav=wav.to(device)
features,discrete_code= wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id) 
torchaudio.save(audio_outpath, audio_out, sample_rate=24000, encoding='PCM_S', bits_per_sample=16)
```


### Part2: Generating discrete codecs
```python

from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer

device=torch.device('cpu')

config_path = "./configs/xxx.yaml"
model_path = "./xxx.ckpt"

wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device)

wav, sr = torchaudio.load(audio_path)
wav = convert_audio(wav, sr, 24000, 1) 
bandwidth_id = torch.tensor([0])
wav=wav.to(device)
_,discrete_code= wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
print(discrete_code)
```



### Part3: Audio reconstruction through codecs
```python
# audio_tokens [n_q,1,t]/[n_q,t]
features = wavtokenizer.codes_to_features(audio_tokens)
bandwidth_id = torch.tensor([0])  
audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
```

## Available models
🤗 links to the Huggingface model hub.

| Model name                                                          |                                                                                                            HuggingFace                                                                                                             |  Corpus  |  Token/s  | Domain | Open-Source |
|:--------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------:|:---------:|:----------:|:------:|
| WavTokenizer-small-600-24k-4096             |             [🤗](https://huggingface.co/novateur/WavTokenizer/blob/main/WavTokenizer_small_600_24k_4096.ckpt)    | LibriTTS  | 40  |  Speech  | √ |
| WavTokenizer-small-320-24k-4096             |             [🤗](https://huggingface.co/novateur/WavTokenizer/blob/main/WavTokenizer_small_320_24k_4096.ckpt)     | LibriTTS  | 75 |  Speech  | √|
| WavTokenizer-medium-320-24k-4096                 |               [🤗](https://huggingface.co/collections/novateur/wavtokenizer-medium-large-66de94b6fd7d68a2933e4fc0)         | 10000 Hours | 75 |  Speech, Audio, Music  | √ |
| WavTokenizer-large-600-24k-4096 | [🤗](https://huggingface.co/novateur/WavTokenizer-large-unify-40token) | 80000 Hours | 40 |   Speech, Audio, Music   | √|
| WavTokenizer-large-320-24k-4096   | [🤗](https://huggingface.co/novateur/WavTokenizer-large-speech-75token) | 80000 Hours | 75 |   Speech, Audio, Music   | √ |

      

## Training

### Step1: Prepare train dataset
```python
# Process the data into a form similar to ./data/demo.txt
```

### Step2: Modifying configuration files
```python
# ./configs/xxx.yaml
# Modify the values of parameters such as batch_size, filelist_path, save_dir, device
```

### Step3: Start training process
Refer to [Pytorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/) for details about customizing the
training pipeline.

```bash
cd ./WavTokenizer
python train.py fit --config ./configs/xxx.yaml
```


## Citation

If this code contributes to your research, please cite our work, Language-Codec and WavTokenizer:

```
@article{ji2024wavtokenizer,
  title={Wavtokenizer: an efficient acoustic discrete codec tokenizer for audio language modeling},
  author={Ji, Shengpeng and Jiang, Ziyue and Wang, Wen and Chen, Yifu and Fang, Minghui and Zuo, Jialong and Yang, Qian and Cheng, Xize and Wang, Zehan and Li, Ruiqi and others},
  journal={arXiv preprint arXiv:2408.16532},
  year={2024}
}

@article{ji2024language,
  title={Language-codec: Reducing the gaps between discrete codec representation and speech language models},
  author={Ji, Shengpeng and Fang, Minghui and Jiang, Ziyue and Huang, Rongjie and Zuo, Jialung and Wang, Shulei and Zhao, Zhou},
  journal={arXiv preprint arXiv:2402.12208},
  year={2024}
}
```