from aipal_codec import AIPalCodec
import argparse
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
import traceback
import subprocess
import time

from utils.file import list_files

class TextDataset(Dataset):
    def __init__(self, input_path):
        self.data = self.load_files(input_path=input_path)
        
         
    def load_files(self, input_path):
        data_list = []
        AUDIO_EXTENSIONS = {
            ".wav",
        }
        print(AUDIO_EXTENSIONS)

        files = list_files(input_path, extensions=AUDIO_EXTENSIONS, recursive=True) 
        for file in files:
            res = {}
            res['audio_path'] = str(file)
            data_list.append(res)
        return data_list


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class CodecClient:
    def __init__(self) -> None:
        pass

    def __init__(self, args):
        self.args = args
        self.local_rank = args.local_rank
    
    def load_asr_instructions(self, config_path: str):
        """
        Args:
             config: asr 的instruction路径地址

        """
        asr_instruction_list = []
        with open(config_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                asr_instruction_list.append(line)

        return asr_instruction_list

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_path", type=str)
        parser.add_argument("--model_config", type=str)
        parser.add_argument("--ckpt_config", type=str)
        parser.add_argument("--output_dir", default='', type=str)
        parser.add_argument("--local_rank", type=int, default=-1)
        args = parser.parse_args()
        return args
    
    def load_model(self):
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = int(os.environ['LOCAL_RANK'])
        
    
        codec = AIPalCodec.from_pretrained(config_path=self.args.model_config,
                                       model_path=self.args.ckpt_config,
                                       is_debug=False, local_rank=local_rank).eval()
        if local_rank == 0:
            self.run_nvidia_smi()
        return codec
    
    def post_result(self, audio_token_list):
        """_summary_

        Args:
            audio_token_dict (_type_): _description_
        """
        audio_tokens = []

        for token_dict in audio_token_list:
            audio_tokens.append(token_dict['absolute_token_id'])
        if len(audio_tokens) == len(audio_token_list):
            return audio_tokens
        else:
            return []

    def save_data(self, audio_path, audio_token):
        """_summary_

        Args:
            audio_path (_type_): _description_
            audio_token (_type_): _description_
        """
        save_file_path = audio_path.replace('.wav', '.audio')
        audio_tokens = self.post_result(audio_token_list=audio_token)
        if len(audio_tokens) == 0:
            return

        res = {'audio_path': audio_path,
               'audio_token': audio_tokens}
        
        print(f'save_file_path:{save_file_path}')
        with open(save_file_path, 'w', encoding='utf-8') as fout:
            fout.write(json.dumps(res, ensure_ascii=False))
    
    def run_nvidia_smi(self):
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print(result.stdout.decode('utf-8'))
        else:
            print(f"Error: {result.stderr.decode('utf-8')}")

    
    def inference(self):
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        
        codec = self.load_model()
        

        self.dataset = TextDataset(self.args.input_path)
        self.sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        self.dataloader = DataLoader(self.dataset, batch_size=4, sampler=self.sampler)
        
        # for audio_path in audio_path_list:
        for audio_path_batch in tqdm(self.dataloader, desc="Processing"):
            try:
                st_time = time.time()
                res_batch, _, _ = codec.encode(audio_path_batch['audio_path'], enable_bfloat16=True)
                end_time = time.time()
                #print(f'spend time is:{end_time-st_time}')
                
                
                for audio_token, audio_path in zip(res_batch.codes_list, audio_path_batch['audio_path']):
                    self.save_data(audio_path=audio_path, audio_token=audio_token)
            except Exception as e:
                traceback.print_exc()
                print(f' the exception is :{str(e)}')
                continue

def main():
    args = CodecClient.get_args()
    inference_instance = CodecClient(args)
    inference_instance.inference()

if __name__=='__main__':
    device_count = torch.cuda.device_count()

    print(f"当前可见的 GPU 设备数量: {device_count}", flush=True) 
    main()