import os
import json
import re
import pathlib
import glob
import shutil
from functools import reduce

import torch
from torch import nn
import numpy as np
import soundfile as sf
from tqdm import tqdm
import librosa

from .models import load_wav
from .models import HiFiGANGenerator
from .models import ConvNeXtEncoder
from .models import LogMelSpectrogram

from .vector_quantization import DownsampleGRVQ, GRVQResult


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class DistilCodec(nn.Module):
    def __init__(
        self,
        configs: dict,
        is_debug=False,
        only_quantizer: bool = False
    ):
        super().__init__()
        
        self.is_debug = is_debug
        self.device = None
        self.ckpt_step = 0
        self.codec_config = configs
        self.ngroups = configs['quantizer']['n_groups']
        self.nresiduals = configs['quantizer']['n_codebooks']
        self.g_ckpt_path = ''
        
        self.encoder_config = AttrDict(configs['encoder'])
        self.decoder_config = AttrDict(configs['decoder'])
        self.quantizer_config = AttrDict(configs['quantizer'])
        self.quantizer_config.pop('quantizer_type')
        self.spec_config = AttrDict(configs['spec_transform'])

        self.encoder = ConvNeXtEncoder(**self.encoder_config) if not only_quantizer else None
        self.quantizer = DownsampleGRVQ(**self.quantizer_config)
        self.generator = HiFiGANGenerator(**self.decoder_config) if not only_quantizer else None
        
        self.spec_transform = LogMelSpectrogram(
            sample_rate=self.spec_config.sampling_rate,
            n_fft=self.spec_config.n_fft,
            win_length=self.spec_config.win_size,
            hop_length=self.spec_config.hop_size,
            n_mels=self.spec_config.num_mels,
            f_min=self.spec_config.fmin,
            f_max=self.spec_config.fmax)
        
        self.hop_size = self.spec_config.hop_size
        self.ds_factor = reduce(lambda x, y: x * y, self.quantizer_config.downsample_factor)

        self.tokens_id_offset = configs['token_id_offset'] if 'token_id_offset' in configs else 0
        base_model = configs['base_model'] if 'base_model' in configs else ''
        self.gr_audio_code2token = self.construct_audio_code(self.tokens_id_offset)

    def move_to_cuda(self):
        self.encoder = self.encoder.to(self.device)
        self.generator = self.generator.to(self.device)
        self.quantizer = self.quantizer.to(self.device)
        
    @classmethod
    def from_pretrained(cls, config_path, model_path, load_steps=-1, is_debug=False, use_generator=False, local_rank=0):
        with open(config_path) as f:
            model_config = json.loads(f.read())

        cp_g = model_path
        
        codec = cls(model_config)
        codec.device = torch.device('cuda:{:d}'.format(local_rank))
        codec.is_debug = is_debug
        codec.ckpt_step = -1
        codec.g_ckpt_path = -1
        
        state_dict_g = cls.load_checkpoint(cp_g, codec.device)
        if use_generator:
            codec.generator.load_state_dict(state_dict_g['generator'])
        codec.encoder.load_state_dict(state_dict_g['encoder'])
        codec.quantizer.load_state_dict(state_dict_g['quantizer'])
        codec.move_to_cuda()
        
        return codec
    
    def preprocess_raw_audio_batch(self, audio_data_info_list: list[str]):
        audio_list = []
        audio_lengths = []
        n_hop_lengths = []
        gen_audio_time_lengths = []
        new_files = []
        max_length = total_time = 0
        for p in audio_data_info_list:
            audio, sampling_rate = p
            if sampling_rate != self.spec_config.sampling_rate:
                audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=self.spec_config.sampling_rate)
                sampling_rate = self.spec_config.sampling_rate
       
            audio = torch.FloatTensor(audio)
            audio = audio.unsqueeze(0)
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
             
            total_time += audio.shape[1] / self.spec_config.sampling_rate
            max_length = max(audio.shape[1], max_length)
            n_hop_length = audio.shape[1] // (self.hop_size * self.ds_factor)
            gen_time_length = (audio.shape[1] // self.hop_size) * (self.hop_size + 1)
            
            audio_list.append(audio)
            audio_lengths.append(audio.shape[1])
            n_hop_lengths.append(n_hop_length)
            gen_audio_time_lengths.append(gen_time_length)
            new_files.append(p)
        
        if self.is_debug:
            print(f'Max lengths: {max_length}')
            print(f'Total time: {total_time:.2f}s')
        
        # Pad to max length
        for i, audio in enumerate(audio_list):
            audio_list[i] = torch.nn.functional.pad(audio, 
                                                    (1, max_length - audio_lengths[i]), 
                                                    "constant")
        audios = torch.stack(audio_list, dim=0).to(self.device)
        mel_specs = self.spec_transform(audios).to(self.device)
        
        if self.is_debug:
            print(f'Audios shape: {audios.shape}')
            for rl, nhl in zip(audio_lengths, gen_audio_time_lengths):
                print(f'Real length: {rl}, NHop length: {nhl}')

        return audios, mel_specs, gen_audio_time_lengths, n_hop_lengths
    
    def preprocess_audio_batch(self, audio_pathes: list[str]):
        audio_list = []
        audio_lengths = []
        n_hop_lengths = []
        gen_audio_time_lengths = []
        new_files = []
        max_length = total_time = 0
        for p in audio_pathes:
            try:
                audio, sampling_rate = load_wav(p, sr=self.spec_config.sampling_rate)
            except Exception as e:
                print(f"Error on audio: {p}")
                audio = np.random.normal(size=(self.spec_config.sampling_rate, )) * 0.05
                sampling_rate = self.spec_config.sampling_rate
            if sampling_rate != self.spec_config.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(sampling_rate, 
                                                                           self.spec_config.sampling_rate))
            
            audio = torch.FloatTensor(audio)
            audio = audio.unsqueeze(0)
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
             
            total_time += audio.shape[1] / self.spec_config.sampling_rate
            max_length = max(audio.shape[1], max_length)
            n_hop_length = audio.shape[1] // (self.hop_size * self.ds_factor)
            gen_time_length = (audio.shape[1] // self.hop_size) * (self.hop_size + 1)
            
            audio_list.append(audio)
            audio_lengths.append(audio.shape[1])
            n_hop_lengths.append(n_hop_length)
            gen_audio_time_lengths.append(gen_time_length)
            new_files.append(p)
        
        if self.is_debug:
            print(f'Max lengths: {max_length}')
            print(f'Total time: {total_time:.2f}s')
        
        # Pad to max length
        for i, audio in enumerate(audio_list):
            audio_list[i] = torch.nn.functional.pad(audio, 
                                                    (1, max_length - audio_lengths[i]), 
                                                    "constant")
        audios = torch.stack(audio_list, dim=0).to(self.device)
        mel_specs = self.spec_transform(audios).to(self.device)
        
        if self.is_debug:
            print(f'Audios shape: {audios.shape}')
            for rl, nhl in zip(audio_lengths, gen_audio_time_lengths):
                print(f'Real length: {rl}, NHop length: {nhl}')

        return audios, mel_specs, gen_audio_time_lengths, n_hop_lengths
    
    def construct_audio_code(self, tokens_id_offset: int = 0):
        stacked_codebooks = self.quantizer.grvq.codebooks.cpu()
        gr_audio_code2token = {}
        code_index_diff = tokens_id_offset
        for g_number, g_codebook in enumerate(stacked_codebooks.split(split_size=1, dim=0)):
            g_codebook = g_codebook.squeeze(0)
            for r_number, r_codebook in enumerate(g_codebook.split(split_size=1, dim=0)):
                r_codebook = r_codebook.squeeze(0)
                codebook_size = r_codebook.shape[0]
                code_numbers_str = {}
                for n in range(0, codebook_size):
                    code_numbers_str[str(n)] = {
                        'content': f'<|g{g_number}r{r_number}_{str(n + code_index_diff)}|>',
                        'absolute_token_id': n + code_index_diff,
                        'in_codebook_id': n
                    }
                gr_audio_code2token[f'g{g_number}r{r_number}'] = {
                    "codebook_size": codebook_size,
                    "audio_code_token": code_numbers_str
                }
            code_index_diff += codebook_size

        gr_audio_code2token['special_audio_tokens'] = {
            str(code_index_diff + 0): {
                "content": "<|beginofaudio|>",
                "description": "Audio output mode begin descriptor",
                'absolute_token_id': code_index_diff + 0
            },
            str(code_index_diff + 1): {
                "content": "<|endofaudio|>",
                "description": "Audio output mode end descriptor",
                'absolute_token_id': code_index_diff + 1
            },
            str(code_index_diff + 2): {
                "content": "<|sil|>",
                "description": "Audio silence descriptor",
                'absolute_token_id': code_index_diff + 2
            },
            str(code_index_diff + 3): {
                "content": "<|inter_audio_begin|>",
                "description": "Interleave Audio output mode begin descriptor",
                'absolute_token_id': code_index_diff + 3
            },
            str(code_index_diff + 4): {
                "content": "<|inter_audio_end|>",
                "description": "Interleave Audio output mode end descriptor",
                'absolute_token_id': code_index_diff + 4
            },
            str(code_index_diff + 5): {
                "content": "<|cot_begin|>",
                "description": "Cot begin descriptor",
                'absolute_token_id': code_index_diff + 7
            },
            str(code_index_diff + 6): {
                "content": "<|cot_end|>",
                "description": "Cot end descriptor",
                'absolute_token_id': code_index_diff + 8
            },
            str(code_index_diff + 7): {
                "content": "<|unused600|>",
                "description": "unused end descriptor",
                'absolute_token_id': code_index_diff + 9
            }
        }

        return gr_audio_code2token

    def get_codebook(self, is_one_codebook: bool = False):
        stacked_codebooks = self.quantizer.grvq.codebooks.cpu()
        code_index_diff = 0
        codebooks = []
        audio_tokens = []
        for g_number, g_codebook in enumerate(stacked_codebooks.split(split_size=1, dim=0)):
            g_codebook = g_codebook.squeeze(0)
            for r_number, r_codebook in enumerate(g_codebook.split(split_size=1, dim=0)):
                r_codebook = r_codebook.squeeze(0)
                codebook_size = r_codebook.shape[0]
                if is_one_codebook:
                    codebooks.append(r_codebook)
                    audio_tokens.extend([self.gr_audio_code2token[f'g{g_number}r{r_number}']['audio_code_token'][x]['content'] for x in 
                                         self.gr_audio_code2token[f'g{g_number}r{r_number}']['audio_code_token'].keys()])
                else:
                    self.gr_audio_code2token[f'g{g_number}r{r_number}']['codebook'] = r_codebook
            code_index_diff += codebook_size
        self.gr_audio_code2token['audio_tokens_all'] = audio_tokens

        if is_one_codebook:
            codebook_cat = torch.cat(codebooks, dim=0)
            print(f'Codebook size: {codebook_cat.shape}')
        else:
            codebook_cat = None

        return self.gr_audio_code2token, codebook_cat

    @staticmethod
    def embedding_analysis(embeddings: torch.Tensor, desp: str = ''):
        embeddings = embeddings.float()
        mean_audio_number = embeddings.abs().mean().item()
        max_audio_number = embeddings.max().item()
        min_audio_number = embeddings.min().item()
        print(f'''{desp} mean: {mean_audio_number}\n{desp} max: {max_audio_number}\n{desp} min: {min_audio_number}''')
    
    def llm_token_expanding(self, 
                            llm_path='/cognitive_comp/sunqianguo/pretrained/open_source/qwen/Qwen2.5-7B',
                            saved_path='/cognitive_comp/wangrui/data/qwen_models/qwen2.5-7b-ate',
                            is_test: bool = True,
                            is_random_init: bool = False,
                            audio_scale_factor: float = 100.0):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load LLM
        text_model = AutoModelForCausalLM.from_pretrained(llm_path, device_map="auto", torch_dtype=torch.bfloat16)
        print(text_model)

        # Get audio embeddings
        audio_tokens_info, audio_embeddings = self.get_codebook(is_one_codebook=True)
        n_audio_tokens = audio_embeddings.shape[0]
        self.embedding_analysis(audio_embeddings, 'Audio embds')
        norm_audio_embedding = audio_embeddings / audio_scale_factor # (audio_embeddings - mean_audio_embedding)
        print(f'{"*"*20}\nPartial norm audio embed:\n{norm_audio_embedding[20000: 20050, 200: 250]}\n{"*"*20}' )

        special_tokens_info = audio_tokens_info['special_audio_tokens']
        n_special_tokens = len(special_tokens_info)
        special_embeddings = torch.nn.Embedding(num_embeddings=n_special_tokens, 
                                                embedding_dim=self.quantizer_config['codebook_dim'],
                                                dtype=torch.bfloat16)

        # Get text embeddings
        text_embeddings = text_model.get_input_embeddings()
        self.embedding_analysis(text_embeddings.weight.data, 'Text embds')
        mean_text_embedding = text_embeddings.weight.data.mean(dim=0)
        print(f'Text embedding: {text_embeddings.weight.data.shape} {text_embeddings.weight.data.type()}')

        # Merge embeddings
        n_text_tokens = text_embeddings.weight.data.shape[0]
        assert n_text_tokens == self.tokens_id_offset
        embd_dim = text_embeddings.weight.data.shape[1]
        assert embd_dim == self.quantizer_config['codebook_dim']
        new_embedding_number = n_text_tokens + n_audio_tokens + n_special_tokens
        new_embeddings = torch.nn.Embedding(new_embedding_number, embd_dim, dtype=torch.bfloat16)
        new_embeddings.weight.data[:n_text_tokens, :] = text_embeddings.weight.data
        new_audio_embd = mean_text_embedding.repeat(audio_embeddings.shape[0], 1)
        new_embeddings.weight.data[n_text_tokens: n_text_tokens+n_audio_tokens, :] = norm_audio_embedding.bfloat16() if not is_random_init else (torch.zeros_like(audio_embeddings, dtype=torch.bfloat16) + 0.00)
        expand_mean_text = mean_text_embedding.repeat(special_embeddings.weight.data.shape[0], 1)
        new_embeddings.weight.data[n_text_tokens+n_audio_tokens:, :] = expand_mean_text if not is_random_init else (torch.zeros_like(special_embeddings.weight.data, dtype=torch.bfloat16) + 0.00)
        print('Set text-audio embedding to LLM...')
        text_model.set_input_embeddings(new_embeddings)
        at_embd = text_model.get_input_embeddings()
        print(f'New text-audio embedding: {at_embd.weight.data.shape} {at_embd.weight.data.type()}')

        # Merge lm_head
        lm_head = text_model.lm_head
        print(f'lm_head: {lm_head.weight.data.size()} {lm_head.weight.data.type()}')
        new_lm_head = torch.nn.Linear(in_features=embd_dim, out_features=new_embedding_number, bias=False, dtype=torch.bfloat16)
        new_lm_head.weight.data[:n_text_tokens, :] = lm_head.weight.data[:n_text_tokens, :]
        mean_lm_head = norm_audio_embedding.mean(dim=0)
        expand_audio_head = mean_lm_head.repeat(audio_embeddings.shape[0], 1)
        new_lm_head.weight.data[n_text_tokens: n_text_tokens+n_audio_tokens, :] = norm_audio_embedding.bfloat16() if not is_random_init else (torch.zeros_like(audio_embeddings, dtype=torch.bfloat16) + 0.00)
        mean_special_head = lm_head.weight.data.mean(dim=0)
        expand_mean_head = mean_special_head.repeat(special_embeddings.weight.data.shape[0], 1)
        new_lm_head.weight.data[n_text_tokens+n_audio_tokens:, :] = expand_mean_head if not is_random_init else (torch.zeros_like(special_embeddings.weight.data, dtype=torch.bfloat16) + 0.00)
        print('Set text-audio lm_head to LLM...')
        text_model.lm_head = new_lm_head
        at_head = text_model.lm_head
        print(f'New text-audio lm_head: {at_head.weight.data.shape} {at_head.weight.data.type()}')

        # Saving model and model config
        print(f'Saving new audio-text LLM to {saved_path}')
        if not os.path.exists(saved_path):
            os.mkdir(saved_path)
        text_model.save_pretrained(saved_path)
        config_path = os.path.join(saved_path, 'config.json')
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        model_config['vocab_size'] = new_embedding_number
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=4)
        print('Completed.')
        
        # Copying codec ckpt to saved_path
        new_codec_save_path = saved_path
        codec_config_path = os.path.join(new_codec_save_path, 'codec_config.json')
        print(f'Save codec config to {codec_config_path}')
        with open(codec_config_path, 'w') as f:
            json.dump(self.codec_config, f, indent=4)
        print(f'Saving codec ckpt to {new_codec_save_path}')
        shutil.copy(self.g_ckpt_path, new_codec_save_path)
        print('Completed.')

        # Merge tokenizer
        old_tokenizer = AutoTokenizer.from_pretrained('/cognitive_comp/sunqianguo/pretrained/open_source/qwen/Qwen2.5-7B')
        real_vocab_size = old_tokenizer.vocab_size + len(old_tokenizer.added_tokens_decoder)
        print(f'Original tokenizer vocabulary size: nopadding-{real_vocab_size} padding-{n_text_tokens}')
        unused_vocab_size = n_text_tokens - real_vocab_size
        print(f'Unused token size: {unused_vocab_size}')
        unused_tokens = [f'<|unused{i}|>' for i in range(unused_vocab_size)]
        audio_tokens_all = audio_tokens_info['audio_tokens_all']
        print(f'Original Audio tokens:\n{audio_tokens_all[:10]}\n{audio_tokens_all[35000: 35010]}')
        print('Add audio tokens to LLM...')
        new_tokens = unused_tokens + audio_tokens_all
        old_tokenizer.add_tokens(new_tokens=new_tokens)
        special_tokens = [special_tokens_info[key]['content'] for key in special_tokens_info.keys()]
        print(f'Special tokens: {special_tokens}')
        old_tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        print('Saving new tokenizer...')
        old_tokenizer.save_pretrained(saved_path)

        if is_test:
            print('New text-audio model testing...')
            new_model = AutoModelForCausalLM.from_pretrained(saved_path, device_map="auto", torch_dtype=torch.bfloat16)
            print(new_model.lm_head.weight.data[160000: 180000, :])
            print(f'Test Model:\n{new_model}')
            new_tokenizer = AutoTokenizer.from_pretrained(saved_path)
            print(f'New Tokenizer vocabulary size: {new_tokenizer.vocab_size + len(new_tokenizer.added_tokens_decoder)}')
            print('\nTest Case1:')
            print(new_tokenizer.tokenize(text=''.join(special_tokens)))
            print(new_tokenizer.encode(text=''.join(special_tokens), add_special_tokens=True))
            print('\nTest Case2:')
            print(new_tokenizer.tokenize(text=''.join(audio_tokens_all[:5])))
            print(new_tokenizer.encode(text=''.join(audio_tokens_all[:5]), add_special_tokens=True))
            # print('\nTest Case3:')
            # print(new_tokenizer.tokenize(text=''.join(audio_tokens_all[35000: 35005])))
            # print(new_tokenizer.encode(text=''.join(audio_tokens_all[35000: 35005]), add_special_tokens=True))

    @staticmethod
    def _pairwise_distance(unvalid_embd: torch.Tensor, valid_embd: torch.Tensor) -> torch.tensor:
        dot_product = torch.matmul(unvalid_embd, valid_embd.transpose(1, 0))
        unvalid_norm = torch.norm(unvalid_embd, p=2, dim=1, keepdim=True)
        valid_norm = torch.norm(valid_embd, p=2, dim=1, keepdim=True).transpose(1, 0)
        distances = unvalid_norm - 2 * dot_product + valid_norm
        print(f'Distances size: {distances.shape}')

        return distances

    def reset_codebook(self, 
                       unique_indice,
                       save_path: str,
                       topK: int = 6):
        print(f'Top-K: {topK}')

        if isinstance(unique_indice, str):
            with open(unique_indice, 'r') as f:
                unique_codes: dict = json.load(f)
        elif isinstance(unique_indice, dict):
            unique_codes = unique_indice
        else:
            raise ValueError('param unique_indice is not [str|list]')

        audio_tokens_info, _ = self.get_codebook(is_one_codebook=False)
        for i, gr_name in enumerate(unique_codes.keys()):
            valid_codes_t = unique_codes[gr_name]
            valid_embd = audio_tokens_info[gr_name]['codebook'][valid_codes_t]
            n_codes = audio_tokens_info[gr_name]['codebook'].shape[0]
            print(f"Original codebook size: {audio_tokens_info[gr_name]['codebook'].shape}")
            all_codes = set(list(range(n_codes)))
            unvalid_codes_t = list(all_codes - set(valid_codes_t))
            print(f'Total codes: {n_codes}\nNumber valid codes: {len(valid_codes_t)}\nNumber unvalid codes: {len(unvalid_codes_t)}')
            unvalid_embd = audio_tokens_info[gr_name]['codebook'][unvalid_codes_t]
            distances = self._pairwise_distance(unvalid_embd.cuda(), valid_embd.cuda())
            _, sorted_indices = torch.sort(distances, dim=-1, descending=True)
            topK_indices = sorted_indices[:, :topK]
            print(f'Sorted index: {topK_indices.shape}\nSome TopK examples: {topK_indices[:5, :]}')
            topk_embds = []
            for topk in tqdm(torch.split(topK_indices, 1, 0), desc='Merge TopK embds'):
                topk_embd = valid_embd[topk.tolist()].mean(dim=0)
                topk_embds.append(topk_embd)
            new_unvalid_embd = torch.stack(topk_embds)
            print(f'New unvalid embedding: {new_unvalid_embd.shape}')
            new_codebook = torch.cat([valid_embd, new_unvalid_embd]).unsqueeze(dim=0)
            print(f'New Unvalid embedding size: {new_codebook.shape}')
            cur_group, cur_residual = i // self.ngroups, i % self.nresiduals
            print(f'Group: {cur_group}, Residual: {cur_residual}')
            self.quantizer.grvq.rvqs[cur_group].layers[cur_residual]._codebook.embed.data = new_codebook
        
        if save_path is None:
            return
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        checkpoint_path = "{}/g_{:08d}".format(save_path, self.ckpt_step)
        self.save_checkpoint(
            checkpoint_path, {
                'generator': self.generator.state_dict(),
                'encoder': self.encoder.state_dict(),
                'quantizer': self.quantizer.state_dict()
            },
            num_ckpt_keep=5
        )

    @staticmethod
    def load_checkpoint(filepath, device):
        assert os.path.isfile(filepath)
        checkpoint_dict = torch.load(filepath, map_location=torch.device('cpu'))
        return checkpoint_dict

    @staticmethod
    def save_checkpoint(filepath, obj, num_ckpt_keep=5):
        name = re.match(r'(do|g)_\d+', pathlib.Path(filepath).name).group(1)
        ckpts = sorted(pathlib.Path(filepath).parent.glob(f'{name}_*'))
        if len(ckpts) > num_ckpt_keep:
            [os.remove(c) for c in ckpts[:-num_ckpt_keep]]
        print("Saving checkpoint to {}".format(filepath))
        torch.save(obj, filepath)
        print("Complete.")

    @staticmethod
    def scan_checkpoint(cp_dir, prefix, target_steps: int = -1):
        pattern = os.path.join(cp_dir, prefix + '????????')
        cp_list = glob.glob(pattern)
        if len(cp_list) == 0:
            return None
        
        if target_steps != -1:
            for p in cp_list:
                if p.endswith(str(target_steps)):
                    return p
        
        return sorted(cp_list)[-1]

    def forward(self, audio_pathes: list):
        audios, mel_specs, gen_time_lengths, n_hop_lengths = self.preprocess_audio_batch(audio_pathes=audio_pathes)
        encoded_mel = self.encoder(mel_specs)
        if self.is_debug:
            print(f'Mel spectrums: {mel_specs.shape}')
            print(f'Encoded Mel spectrums: {encoded_mel.shape}')
            
        rvq_result = self.quantizer(encoded_mel)
        quantized = rvq_result.quantized
        
        y_g_hat = self.generator(quantized)
        
        return y_g_hat, audios, gen_time_lengths, n_hop_lengths
    
    def audio_tokenize(self, codes: list, n_groups: int, n_residual: int):
        n_gr = n_groups * n_residual
        gr_codes = [codes[i: i + n_gr] for i in range(0, len(codes), n_gr)]
        new_codes = []
        for gr in gr_codes:
            group_codes = [gr[i: i + n_residual] for i in range(0, len(gr), n_residual)]
            for g, gr_code in enumerate(group_codes):
                for r, re_code in enumerate(gr_code):
                    audio_token = self.gr_audio_code2token[f'g{g}r{r}']['audio_code_token'][str(re_code)]
                    new_codes.append(audio_token)

        return new_codes

    def encode(self, audio_pathes: list, enable_bfloat16: bool = False, raw_audio=False) -> GRVQResult:
        if raw_audio:
            _, mel_specs, gen_time_lengths, n_hop_lengths = self.preprocess_raw_audio_batch(audio_pathes)
        else:
            _, mel_specs, gen_time_lengths, n_hop_lengths = self.preprocess_audio_batch(audio_pathes=audio_pathes)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enable_bfloat16):
            encoded_mel = self.encoder(mel_specs)
            if self.is_debug:
                print(f'Mel spectrums: {mel_specs.shape}')
                print(f'Encoded Mel spectrums: {encoded_mel.shape}')
            quantized_ret = self.quantizer(encoded_mel)
            for split_codes, split_pjt_in, split_fup, hop_len in zip(quantized_ret.codes.split(split_size=1, dim=1),
                                                                     quantized_ret.x_pjt_in.split(split_size=1, dim=0),
                                                                     quantized_ret.quantized_fup.split(split_size=1, dim=0),
                                                                     n_hop_lengths):
                codes_t = split_codes[:, :, :hop_len, :]
                n_groups, n_residual, seq_len = codes_t.shape[0], codes_t.shape[-1], codes_t.shape[2]
                codes_t = codes_t.squeeze(dim=1).transpose(1, 0).reshape(seq_len, n_groups * n_residual).flatten().cpu().tolist()
                codes_with_audio_info = self.audio_tokenize(codes=codes_t, n_groups=n_groups, n_residual=n_residual)
                quantized_ret.codes_list.append(codes_with_audio_info)
                # print(split_codes, codes_with_audio_info[:10])
                pjt_in_t = split_pjt_in[:, :hop_len, :].squeeze(0).reshape(hop_len, 2, -1).reshape(hop_len * 2, -1).cpu()
                quantized_ret.x_pjt_in_list.append(pjt_in_t)

                fup_t = split_fup[:, :hop_len, :].squeeze(0).reshape(hop_len, 2, -1).reshape(hop_len * 2, -1).cpu()
                quantized_ret.quantized_fup_list.append(fup_t)
                # print(hop_len, pjt_in_t.shape, fup_t.shape, len(codes_t))

        return quantized_ret, gen_time_lengths, n_hop_lengths

    def decode_from_features(self, quantized_features, enable_bfloat16: bool = False) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enable_bfloat16):
            y_g_hat = self.generator(quantized_features)

        return y_g_hat
    
    def decode_from_codes(self, codes: list, minus_token_offset: bool = True, enable_bfloat16: bool = False) -> torch.Tensor:

        if minus_token_offset:
            for c in codes:
                if c - self.tokens_id_offset < 0:
                    print(f'c is :{c}', flush=True)
            codes = [c - self.tokens_id_offset for c in codes]
        codes = torch.tensor(codes, dtype=torch.int64).unsqueeze(0).unsqueeze(0).unsqueeze(-1).cuda()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enable_bfloat16):
                re_features = self.quantizer.decode(indices=codes)
                y_g_hat = self.generator(re_features)

        return y_g_hat.detach()
    # Add this method to your DistilCodec class in distil_codec.py
    # Insert it right after your existing decode_from_codes method (around line 400+)

    def decode_from_codes_batch(self, codes_list: list, minus_token_offset: bool = True, enable_bfloat16: bool = False) -> list:
        """
        FINAL CORRECT VERSION - 3D tensor as required by conv_transpose1d
        Input shape: [batch_size, channels, sequence_length] = 3D
        """
        if not codes_list:
            return []
        
        # Process token offset for all sequences
        if minus_token_offset:
            processed_codes_list = []
            for codes in codes_list:
                for c in codes:
                    if c - self.tokens_id_offset < 0:
                        print(f'c is :{c}', flush=True)
                processed_codes = [c - self.tokens_id_offset for c in codes]
                processed_codes_list.append(processed_codes)
            codes_list = processed_codes_list
        
        # Handle variable sequence lengths with padding
        max_length = max(len(codes) for codes in codes_list)
        batch_size = len(codes_list)
        
        # Create 3D tensor: [batch_size, channels, sequence_length]
        # This is what conv_transpose1d expects for batched input
        batched_codes = torch.zeros(batch_size, 1, max_length, dtype=torch.int64).cuda()
        
        # Fill the batch tensor
        for i, codes in enumerate(codes_list):
            codes_tensor = torch.tensor(codes, dtype=torch.int64)
            batched_codes[i, 0, :len(codes)] = codes_tensor
        
        # Process entire batch
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enable_bfloat16):
                re_features = self.quantizer.decode(indices=batched_codes)
                y_g_hat_batch = self.generator(re_features)
        
        # Split batch results back to individual tensors
        results = []
        for i in range(batch_size):
            individual_result = y_g_hat_batch[i:i+1].detach()
            results.append(individual_result)
        
        return results
    def save_wav(self, audio_gen_batch: torch.Tensor, nhop_lengths, audio_names=None, save_path='./log', name_tag='default'):
        if audio_names is not None and len(audio_names) == len(nhop_lengths):
            use_org_name = True
        else:
            use_org_name = False
        
        gen_audio_pathes = []
        for i in range(audio_gen_batch.shape[0]):
            audio_gen = audio_gen_batch[i, 0, :nhop_lengths[i]].float().cpu().numpy()
            audio_name = f'{name_tag}.wav' if not use_org_name else f'{audio_names[i]}'
            audio_path_t = os.path.join(save_path, audio_name)
            gen_audio_pathes.append(audio_path_t)
            sf.write(audio_path_t, audio_gen, self.spec_config.sampling_rate)
            
        return gen_audio_pathes
    
    #test
def load_and_resample_audio(file_path, target_sr, mono=True, limited=None):
    """
    读取说话人语音文件，修改采样率，并合并多声道（如果需要）。

    :param file_path: 说话人语音文件的路径
    :param target_sr: 目标采样率
    :param mono: 是否将说话人语音转换为单声道（True）或保持多声道（False）
    :return: 处理后的说话人语音信号和目标采样率
    """
    # 读取说话人语音文件
    y, orig_sr = librosa.load(file_path, sr=None, mono=False)
    audio_duration = len(y) / orig_sr
    # 如果音频长度超过最大时长，随机截取一段音频
    if limited is not None and audio_duration > limited and len(y) - int(orig_sr * limited) > 1000:
        start = np.random.randint(0, len(y) - int(orig_sr * limited))
        y = y[start:start + int(orig_sr * limited)]
        # y = y[:int(limited * orig_sr)]

    # 修改采样率
    y_resampled = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)

    # 合并多声道（如果需要）
    if mono and len(y_resampled.shape) > 1:
        y_resampled = np.mean(y_resampled, axis=0, keepdims=True)
    if len(y_resampled.shape) == 1:
        y_resampled = np.expand_dims(y_resampled, axis=0)

    return y_resampled, target_sr, audio_duration


def decode_audio(codec: DistilCodec, audio_tsr, target_sr=24000, plus_offset: bool = True):
    """A demo method for decoding audio token int audio wave.

    Args:
        codec (DistilCodec): An instance of DisilCodec
        audio_tsr (_type_): Audio with target sampling rate.
        target_sr (int, optional): Target Sampling Rate. Defaults to 24000.
        plus_offset (bool, optional): Weather plus LLM's token offset, if set to False, then the audio token will be in [0, 32767]; If set to True, then the audio token will be in [offset, offset + 32767]. Defaults to True.

    Returns:
        List of int: Audio tokens list.
    """
    
    with torch.no_grad():
        quantized_ret = codec.encode([[audio_tsr.tolist()[0], target_sr]], enable_bfloat16=True, raw_audio=True)[0]
        if plus_offset:
            audio_tokens = [code + codec.tokens_id_offset for code in quantized_ret.codes.squeeze().cpu().tolist()]
        else:
            audio_tokens = quantized_ret.codes.squeeze().cpu().tolist()
        # print(f'Audio tokens: {audio_tokens}')

    return audio_tokens


def demo_for_generate_audio_codes(codec: DistilCodec, audio_path, target_sr=24000, plus_llm_offset=True):
    """A demo method for generate audio from audio tokens.

    Args:
        codec (DistilCodec): An instance of DisilCodec
        audio_path (_type_): The input audio file path.
        target_sr (int, optional): Target sampling rate. Defaults to 24000.
        plus_llm_offset (bool, optional):  Weather plus LLM's token offset, if set to False, then the audio token will be in [0, 32767]; If set to True, then the audio token will be in [offset, offset + 32767]. Defaults to True.

    Returns:
        List of Int: Audio tokens list.
    """
    
    audio_tsr, _, _ = load_and_resample_audio(file_path=audio_path, target_sr=target_sr)
    audio_tokens = decode_audio(codec, audio_tsr=audio_tsr, plus_offset=plus_llm_offset)
    
    return audio_tokens
