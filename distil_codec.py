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
import torch.amp
from tqdm import tqdm

from utils import AttrDict

from models import load_wav
from models import HiFiGANGenerator
from models import ConvNeXtEncoder
from models import LogMelSpectrogram

from vector_quantization import DownsampleGRVQ, GRVQResult


class AudioBPE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
        if self.tokens_id_offset > 0 and base_model != '':
            print(f'Base Model: {base_model}, Token offset: {self.tokens_id_offset}')
        self.gr_audio_code2token = self.construct_audio_code(self.tokens_id_offset)

    def move_to_cuda(self):
        self.encoder = self.encoder.to(self.device)
        self.generator = self.generator.to(self.device)
        self.quantizer = self.quantizer.to(self.device)
        
    @classmethod
    def from_pretrained(cls, config_path, model_path, load_steps=-1, is_debug=False, use_generator=False, local_rank=0):
        with open(config_path) as f:
            model_config = json.loads(f.read())
        
        if os.path.isdir(model_path):
            cp_g = cls.scan_checkpoint(model_path, 'g_', target_steps=load_steps)
            ckpt_step = int(cp_g[-8:])
        else:
            ckpt_step = 0
            raise ValueError('model_path is not a directory')
        
        codec = cls(model_config)
        codec.device = torch.device('cuda:{:d}'.format(local_rank))
        codec.is_debug = is_debug
        codec.ckpt_step = ckpt_step
        codec.g_ckpt_path = cp_g
        
        state_dict_g = cls.load_checkpoint(cp_g, codec.device)
        if use_generator:
            codec.generator.load_state_dict(state_dict_g['generator'])
        codec.encoder.load_state_dict(state_dict_g['encoder'])
        codec.quantizer.load_state_dict(state_dict_g['quantizer'])
        codec.move_to_cuda()
        
        return codec
    
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
                            is_llm_mean: bool = True,
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
        if is_llm_mean:
            audio_init_embd = new_audio_embd
        else:
            audio_init_embd = norm_audio_embedding
        new_embeddings.weight.data[n_text_tokens: n_text_tokens+n_audio_tokens, :] = audio_init_embd.bfloat16() if not is_random_init else (torch.zeros_like(audio_embeddings, dtype=torch.bfloat16) + 0.00)
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
        if is_llm_mean:
            audio_head_init = expand_audio_head
        else:
            audio_head_init = norm_audio_embedding
        new_lm_head.weight.data[n_text_tokens: n_text_tokens+n_audio_tokens, :] = audio_head_init.bfloat16() if not is_random_init else (torch.zeros_like(audio_embeddings, dtype=torch.bfloat16) + 0.00)
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
        print("Loading '{}'".format(filepath))
        checkpoint_dict = torch.load(filepath, map_location=torch.device('cpu'))
        print("Complete.")
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

    def encode(self, audio_pathes: list, enable_bfloat16: bool = False) -> GRVQResult:
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
            codes = [c - self.tokens_id_offset for c in codes]
            print(codes)
        codes = torch.tensor(codes, dtype=torch.int64).unsqueeze(0).unsqueeze(0).unsqueeze(-1).cuda()
        print(codes.size())
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enable_bfloat16):
            re_features = self.quantizer.decode(indices=codes)
            y_g_hat = self.generator(re_features)

        return y_g_hat
    
    def save_wav(self, audio_gen_batch: torch.Tensor, nhop_lengths, audio_names=None, save_path='./log'):
        if audio_names is not None and len(audio_names) == len(nhop_lengths):
            use_org_name = True
        else:
            use_org_name = False
        
        gen_audio_pathes = []
        for i in range(audio_gen_batch.shape[0]):
            audio_gen = audio_gen_batch[i, 0, :nhop_lengths[i]].float().cpu().numpy()
            audio_name = f'output{i}.wav' if not use_org_name else f'{audio_names[i]}'
            audio_path_t = os.path.join(save_path, audio_name)
            gen_audio_pathes.append(audio_path_t)
            sf.write(audio_path_t, audio_gen, self.spec_config.sampling_rate)
            
        return gen_audio_pathes
    

def tokenizer_expanding_qwen():
    import os

    load_steps = 151000
    codec_date = '0927'
    codec = DistilCodec.from_pretrained(config_path=f'/cognitive_comp/wangrui/codes/audio_codec/scripts/workspace/{codec_date}_24k_3s/model_config.json',
                                       model_path=f'/cognitive_comp/wangrui/data/lam{codec_date}/ckpt',
                                       load_steps=load_steps,
                                       is_debug=False).eval()
    divided_factor = 50.0
    llm_path='/cognitive_comp/sunqianguo/pretrained/open_source/qwen/Qwen2.5-7B'
    llm_name = os.path.split(llm_path)[-1]
    saved_path = f'/cognitive_comp/wangrui/data/qwen_models/{llm_name}-Codec{codec_date}-S{load_steps}-AEdivided{int(divided_factor)}'
    codec.llm_token_expanding(saved_path=saved_path,
                              llm_path=llm_path,
                              is_random_init=False,
                              audio_scale_factor=divided_factor)


def reset_codebook():
    codec = DistilCodec.from_pretrained(config_path='/cognitive_comp/wangrui/codes/audio_codec/scripts/workspace/0930_24k_3s/model_config.json',
                                       model_path='/cognitive_comp/wangrui/data/lam0930/ckpt',
                                       load_steps=181000,
                                       is_debug=False).eval()
    codec.reset_codebook(unique_indice='/cognitive_comp/wangrui/data/lam0930/evaluation/unique_indices_182000.json',
                         save_path='/cognitive_comp/wangrui/data/lam0930/reset_ckpt',
                         topK=1)


def codec_test():
    audio_pathes = ["/cognitive_comp/common_data/audio/output/24k_data/dev/stage3/part22/7330f158-004a-4a52-8f55-bdd8462d335d_0174.wav",
                    "/cognitive_comp/common_data/audio/output/dev/df85d640-46bf-4a3d-b642-9ae558fc9966_0012.wav"]
    model_config = "/cognitive_comp/wangrui/codes/audio_codec/scripts/workspace/0827_24k_3s/model_config.json"
    ckpt_config = "/cognitive_comp/wangrui/data/audio_codec_ckpts/0821"
    codec = DistilCodec.from_pretrained(config_path=model_config,
                                       model_path=ckpt_config,
                                       is_debug=False).eval()
    codec.preprocess_audio_batch(audio_pathes=audio_pathes)
    with torch.no_grad():
        quantized_ret, nhop_lengths, _ = codec.encode(audio_pathes)
        print(f'Quantized feature shape: {quantized_ret.quantized.shape}')
        print(f'Quantized feature shape: {quantized_ret.quantized_fup.shape}')
        print(f'Quantized Flatten Codes: {len(quantized_ret.codes_list[0])}')
        y_gen = codec.decode_from_features(quantized_features=quantized_ret.quantized)
        print(y_gen.shape)
        #codec.save_wav(audio_gen_batch=y_gen,
        #               nhop_lengths=nhop_lengths)
        codec.save_wav(audio_gen_batch=y_gen,
                       nhop_lengths=nhop_lengths)
        

def codec_test():
    audio_pathes = ["/cognitive_comp/common_data/audio/output/24k_data/dev/stage3/part22/7330f158-004a-4a52-8f55-bdd8462d335d_0174.wav",
                    "/cognitive_comp/common_data/audio/output/dev/df85d640-46bf-4a3d-b642-9ae558fc9966_0012.wav"]
    model_config = "/cognitive_comp/wangrui/codes/audio_codec/scripts/workspace/0827_24k_3s/model_config.json"
    ckpt_config = "/cognitive_comp/wangrui/data/audio_codec_ckpts/0821"
    codec = DistilCodec.from_pretrained(config_path=model_config,
                                       model_path=ckpt_config,
                                       is_debug=False).eval()
    codec.preprocess_audio_batch(audio_pathes=audio_pathes)
    with torch.no_grad():
        quantized_ret, nhop_lengths, _ = codec.encode(audio_pathes)
        print(f'Quantized feature shape: {quantized_ret.quantized.shape}')
        print(f'Quantized feature shape: {quantized_ret.quantized_fup.shape}')
        print(f'Quantized Flatten Codes: {len(quantized_ret.codes_list[0])}')
        y_gen = codec.decode_from_features(quantized_features=quantized_ret.quantized)
        print(y_gen.shape)
        #codec.save_wav(audio_gen_batch=y_gen,
        #               nhop_lengths=nhop_lengths)
        codec.save_wav(audio_gen_batch=y_gen,
                       nhop_lengths=nhop_lengths)
        

def codec_test2():
    audio_pathes = ["/cognitive_comp/common_data/audio/output/dev/df85d640-46bf-4a3d-b642-9ae558fc9966_0012.wav"]
    model_config = "/cognitive_comp/wangrui/codes/audio_codec/scripts/workspace/0927_24k_3s/model_config.json"
    ckpt_config = "/cognitive_comp/wangrui/data/lam0927/ckpt"
    codec = DistilCodec.from_pretrained(config_path=model_config,
                                       model_path=ckpt_config,
                                       load_steps=204000,
                                       use_generator=True,
                                       is_debug=False).eval()
    codec.preprocess_audio_batch(audio_pathes=audio_pathes)
    with torch.no_grad():
        quantized_ret, nhop_lengths, _ = codec.encode(audio_pathes)
        print(f'Quantized feature shape: {quantized_ret.quantized.shape}')
        print(f'Quantized feature shape: {quantized_ret.quantized_fup.shape}')
        print(f'Quantized Flatten Codes: {len(quantized_ret.codes_list[0])}')
        code_list = quantized_ret.codes.squeeze().cpu().tolist()
        print(code_list)
        y_gen = codec.decode_from_codes(codes=code_list)
        print(y_gen.shape)
        codec.save_wav(audio_gen_batch=y_gen,
                       nhop_lengths=nhop_lengths)
        

def codec_test3():
    from transformers import AutoTokenizer
    audio_pathes = './log/codes.json'
    model_config = "/cognitive_comp/wangrui/codes/audio_codec/scripts/workspace/0927_24k_3s/model_config.json"
    ckpt_config = "/cognitive_comp/wangrui/data/lam0927/ckpt"
    codec = DistilCodec.from_pretrained(config_path=model_config,
                                       model_path=ckpt_config,
                                       load_steps=204000,
                                       use_generator=True,
                                       is_debug=False).eval()
    with torch.no_grad():
        with open(audio_pathes, 'r') as f:
            codes = json.load(f)['audio_token']
            codes = []
        code_in_str = '<|g0r0_157815|><|g0r0_163862|><|g0r0_160034|><|g0r0_155890|><|g0r0_155834|><|g0r0_159785|><|g0r0_167700|><|g0r0_158844|><|g0r0_168475|><|g0r0_156415|><|g0r0_155236|><|g0r0_154583|><|g0r0_154007|><|g0r0_181990|><|g0r0_170305|><|g0r0_156821|><|g0r0_170216|><|g0r0_159466|><|g0r0_174469|><|g0r0_170707|><|g0r0_174337|><|g0r0_180287|><|g0r0_175087|><|g0r0_180991|><|g0r0_177685|><|g0r0_174908|><|g0r0_158248|><|g0r0_175281|><|g0r0_157311|><|g0r0_184140|><|g0r0_159974|><|g0r0_165979|><|g0r0_156815|><|g0r0_157850|><|g0r0_155785|><|g0r0_184046|><|g0r0_167890|><|g0r0_163432|><|g0r0_173041|><|g0r0_156178|><|g0r0_168165|><|g0r0_167469|><|g0r0_174342|><|g0r0_153290|><|g0r0_172394|><|g0r0_173658|><|g0r0_171057|><|g0r0_162436|><|g0r0_167485|><|g0r0_168098|><|g0r0_178756|><|g0r0_163534|><|g0r0_183447|><|g0r0_173772|><|g0r0_163517|><|g0r0_165189|><|g0r0_159811|><|g0r0_156508|><|g0r0_164764|><|g0r0_164792|><|g0r0_176297|><|g0r0_153692|><|g0r0_180491|><|g0r0_161067|><|g0r0_182762|><|g0r0_165866|><|g0r0_172214|><|g0r0_180582|><|g0r0_179811|><|g0r0_167920|><|g0r0_167855|><|g0r0_182100|><|g0r0_181488|><|g0r0_170258|><|g0r0_161049|><|g0r0_161285|><|g0r0_159437|><|g0r0_177028|><|g0r0_183933|><|g0r0_178215|><|g0r0_155778|><|g0r0_166534|><|g0r0_155137|><|g0r0_172690|><|g0r0_171536|><|g0r0_174361|><|g0r0_169222|><|g0r0_183609|><|g0r0_166138|><|g0r0_174602|><|g0r0_166426|><|g0r0_180847|><|g0r0_157100|><|g0r0_171762|><|g0r0_178848|><|g0r0_164785|><|g0r0_154833|><|g0r0_160962|><|g0r0_175667|><|g0r0_156849|><|g0r0_171932|><|g0r0_171201|><|g0r0_170609|><|g0r0_160494|><|g0r0_174761|><|g0r0_168980|><|g0r0_154129|><|g0r0_166259|><|g0r0_180191|><|g0r0_168171|><|g0r0_152186|><|g0r0_176752|><|g0r0_170297|><|g0r0_180590|><|g0r0_183165|><|g0r0_163407|><|g0r0_180708|><|g0r0_154321|><|g0r0_161279|><|g0r0_158274|><|g0r0_162719|><|g0r0_160263|><|g0r0_160202|><|g0r0_180274|><|g0r0_156694|><|g0r0_154725|><|g0r0_164957|><|g0r0_166550|><|g0r0_182433|><|g0r0_170488|><|g0r0_158783|><|g0r0_164724|><|g0r0_172897|><|g0r0_164679|><|g0r0_184380|><|g0r0_166021|><|g0r0_168697|><|g0r0_171166|><|g0r0_162009|><|g0r0_154844|><|g0r0_155692|><|g0r0_171710|><|g0r0_165934|><|g0r0_158553|><|g0r0_169979|><|g0r0_179784|><|g0r0_184472|><|g0r0_169085|><|g0r0_171824|><|g0r0_162685|><|g0r0_164228|><|g0r0_167320|><|g0r0_169824|><|g0r0_180333|><|g0r0_153021|><|g0r0_177718|><|g0r0_158474|><|g0r0_159715|><|g0r0_175196|><|g0r0_183049|><|g0r0_170015|><|g0r0_156741|><|g0r0_167391|><|g0r0_156832|><|g0r0_183303|><|g0r0_168916|><|g0r0_164926|><|g0r0_158492|><|g0r0_171096|><|g0r0_179249|><|g0r0_162821|><|g0r0_152531|><|g0r0_171023|><|g0r0_177592|><|g0r0_177592|><|g0r0_156955|><|g0r0_171849|><|g0r0_168435|><|g0r0_157631|><|g0r0_169760|><|g0r0_162991|><|g0r0_165724|><|g0r0_176326|><|g0r0_176516|><|g0r0_160725|><|g0r0_156989|><|g0r0_181405|><|g0r0_156289|><|g0r0_170554|><|g0r0_152159|><|g0r0_179445|><|g0r0_167485|><|g0r0_165265|><|g0r0_160757|><|g0r0_171661|><|g0r0_165689|><|g0r0_181916|><|g0r0_161402|><|g0r0_152478|><|g0r0_162282|><|g0r0_171179|><|g0r0_152115|><|g0r0_158391|><|g0r0_177792|><|g0r0_173494|><|g0r0_171926|><|g0r0_162289|><|g0r0_176522|><|g0r0_177153|><|g0r0_154567|><|g0r0_170274|><|g0r0_181977|><|g0r0_173329|><|g0r0_161432|><|g0r0_170095|><|g0r0_181625|><|g0r0_157642|><|g0r0_184806|><|g0r0_184320|><|g0r0_154581|><|g0r0_173329|><|g0r0_177710|><|g0r0_161300|><|g0r0_154113|><|g0r0_181551|><|g0r0_166982|><|g0r0_178772|><|g0r0_178995|><|g0r0_169957|><|g0r0_170095|><|g0r0_178983|><|g0r0_180523|><|g0r0_165182|><|g0r0_181252|><|g0r0_181875|><|g0r0_183609|><|g0r0_162388|><|g0r0_166426|><|g0r0_158416|>'
        tokenizer_path = '/cognitive_comp/ccnl_common_data/wangrui/audio-text-models/qwen_models/Qwen2.5-7B-Codec0927-S204000-AEdivided100'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        codes = tokenizer.encode(code_in_str)
        codes = [160696, 176479, 167509, 182039, 161377,
        167063, 177177, 182323, 152499, 157423, 180893, 171956, 160442, 161683,
        169686, 174873, 182036, 169422, 160996, 174559, 173466, 156135, 175562,
        183604, 183865, 155058, 156207, 168959, 173597, 176299, 176777, 161744,
        153346, 176133, 152279, 159455, 156060, 172142, 168620, 152279, 173761,
        173761, 173753, 182900, 180904, 180622, 173906, 154597, 160453, 155027,
        159700, 159941, 168436, 155054, 159247, 156958, 166093, 178492, 161658,
        177739, 174587, 159711, 162600, 163470, 152568, 173069, 168287, 156776,
        160596, 165953, 159482, 168551, 153442, 153439, 154692, 153958, 182714,
        164985, 168344, 174487, 165089, 153739, 169078, 159495, 180241, 179662,
        184622, 157898, 153597, 156110, 171521, 160239, 172469, 156290, 160379,
        161498, 172053, 166119, 167627, 154127, 168048, 171665, 175638, 164250,
        173540, 181507, 162071, 173087, 176121, 153264, 183935, 170002, 180521,
        182064, 171399, 181704, 171898, 168989, 156907, 182078, 153387, 182060,
        163519, 172395, 158826, 167289, 177484, 158752, 177607, 155874, 183946,
        152554, 158902, 153039, 174227, 174910, 177002, 181906, 183374, 176863,
        173971, 152747, 179212, 165543, 155802, 181038, 167664, 164408, 159236,
        155713, 173790, 177123, 163837, 167770, 171162, 172076, 164763, 162462,
        171983, 171873, 163763, 174231, 176596, 161777, 159747, 159547, 155062,
        156935, 171059, 174227, 170400, 166328, 175452, 154332, 179059, 171634,
        158740, 164901, 157470, 165322, 182078, 160337, 159765, 154431, 184325,
        163182, 174745, 173688, 180674, 173558, 164364, 177314, 166203, 180641,
        165634, 153219, 158638, 163310, 175255, 180070, 154750, 153160, 163227,
        166567, 155978, 175044, 159237, 174196, 159346, 164882, 156868, 170400,
        180026, 172897, 175602, 160378, 165952, 158896, 160554, 167455, 166524,
        169252, 152106, 179713, 172791, 179984, 156038, 162530, 176629, 176141,
        179885, 161745, 182595, 171765, 161043, 159455, 178913, 155432, 174053,
        175417, 160114, 160233, 156741, 170361, 178302, 152414, 169382, 168165,
        178515, 153954, 179183, 153441, 178784, 153197, 161502, 173269, 177720,
        178282, 168620, 158207, 177025, 175494, 176802, 165308, 165419, 157129,
        168076, 169707, 157374, 174906, 163691, 160681, 184103, 178419, 175418,
        170234, 163612, 165540, 169486, 184385, 171573, 179967, 152553, 164883,
        166654, 175265, 152111, 176327, 182182, 174160, 168556, 156861, 167424,
        152602, 177979, 181102, 165721, 172017, 161800, 155775, 152437, 160188,
        153242, 164730, 172503, 154938, 170421, 171493, 167993, 172053, 155830,
        152993, 153550, 162374, 156664, 156110, 152597, 156664, 168858, 180144,
        170179, 157898, 169994, 162056, 160341, 162266, 152169, 162593, 156051,
        178660, 175913, 162162, 173012, 177176, 178948, 155752, 158717, 165184,
        159956, 174417, 167229, 175944, 155859, 166281, 171628, 182320, 182568,
        155173, 164811, 154750, 167229, 184256, 161643, 175944, 158324, 163388,
        178383, 161853, 162432, 163412, 153037, 162432, 168155, 171475, 170321,
        163788, 169108, 176886, 168901, 160678, 160060, 173713, 161597, 180617,
        173873, 178013, 175097, 165886]
        y_gen = codec.decode_from_codes(codes=codes)
        print(y_gen.shape)
        codec.save_wav(audio_gen_batch=y_gen,
                       nhop_lengths=[y_gen.shape[-1]])
    

if __name__ == "__main__":
    codec_test3()
