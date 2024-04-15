import glob
import json
import os
import pdb
import logging
import string
import random
from os.path import join as pj
from typing import List
from itertools import product
from distutils.dir_util import copy_tree

from .trainer import Trainer, to_list
from .automatic_evaluation import evaluate
from .data import DEFAULT_CACHE_DIR

import easydict 

import shutil
import wandb
import time
import datetime
import pandas as pd

def main():
    
    args = easydict.EasyDict({"checkpoint_dir": "tmp_instruction_flan_t5_small_squad_qg_explanation_sample", 
                              
                              "project": "lmqg_qg_squad",
                              
                              "entity": "minseok0809",
                              
                              "name": "instruction_flan_t5_small_squad_qg_explanation_sample",
                              
                              "dataset_path": "data_explanation", 
                              
                              "dataset_name": "default", 
                              
                              "input_types": "paragraph_answer", 
                              
                              "output_types": "sentence", 
                              
                              "prefix_types": "qg", 
                              
                              "instruction_types": "instruction", 
                              
                              "model": "google/flan-t5-small", 
                              
                              "model_adapter": False,   
                              
                              "lora_rank": 128, 
                              
                              "lora_alpha": 32, 

                              "max_length": 512, 
                              
                              "max_length_output": 32, 
                              
                              "epoch": 20,
                              
                              "epoch_partial": 2, 
                              
                              "batch": 32, 
                              
                              "batch_eval": 32,
                              
                              "n_max_config": 5, 
                              
                              "lr": 5e-05, 
                              
                              "fp16": False, 
                              
                              "random_seed": 1, 
                              
                              "gradient_accumulation_steps": 32, 
                              
                              "label_smoothing": 0.15,
                              
                              "language": "en"})


    __all__ = 'GridSearcher'

    def get_random_string(length: int = 6, exclude: List = None):
        tmp = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
        if exclude:
            while tmp in exclude:
                tmp = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
        return tmp


    class GridSearcher:
        """ Grid search (epoch, batch, lr, random_seed, label_smoothing) """

        def __init__(self, checkpoint_dir: str = args.checkpoint_dir,
                     project: str = args.project, entity: str = args.entity, name: str = args.name,
                     dataset_path: str = args.dataset_path, dataset_name: str = args.dataset_name, 
                    input_types: List or str = args.input_types, output_types: List or str = args.output_types, 
                    prefix_types: List or str = args.prefix_types, instruction_types: List or str = args.instruction_types,
                    model: str = args.model, 
                    model_adapter: bool = args.model_adapter, lora_rank: int = args.lora_rank,
                    lora_alpha: int = args.lora_alpha,
                    fp16: bool = args.fp16,
                    gradient_accumulation_steps: list or int = args.gradient_accumulation_steps, 
                    metric: str = 'validation/Bleu_4',
                    epoch: int = args.epoch, epoch_partial: int = args.epoch_partial, 
                    n_max_config: int = args.n_max_config, max_length: int = args.max_length,
                    max_length_eval: int = None, max_length_output: int = args.max_length_output, 
                    max_length_output_eval: int = None,
                    prediction_aggregation: str = 'first', prediction_level: str = 'sentence',
                    batch: int = args.batch, batch_eval: int = args.batch_eval, n_beams_eval: int = 4, 
                    lr: list or float = args.lr, label_smoothing: list or float = args.label_smoothing, 
                    random_seed: List or int = args.random_seed, language: str = args.language,
                    normalize: bool = True, use_auth_token: bool = False, torch_dtype=None, device_map: str = None,
                    low_cpu_mem_usage: bool = False):

            wandb.init(project=project, entity=entity, name=name)
            
            # evaluation configs
            max_length_eval = max_length if max_length_eval is None else max_length_eval
            max_length_output_eval = max_length_output if max_length_output_eval is None else max_length_output_eval
            self.eval_config = {
                'max_length_eval': max_length_eval, 'max_length_output_eval': max_length_output_eval,
                'n_beams_eval': n_beams_eval, 'prediction_aggregation': prediction_aggregation,
                'prediction_level': prediction_level, 'language': language, 'normalize': normalize
            }

            # static configs
            self.static_config = {
                'dataset_path': dataset_path, 'dataset_name': dataset_name, 'input_types': input_types,
                'output_types': output_types, 'model': model, 'model_adapter': model_adapter, 
                'lora_rank': lora_rank, 'lora_alpha': lora_alpha,
                'fp16': fp16, 'batch': batch, 'epoch': epoch,
                'max_length': max_length, 'max_length_output': max_length_output, 'prefix_types': prefix_types,
                'instruction_types': instruction_types
            }
            
            # dynamic config
            self.epoch = epoch
            self.epoch_partial = epoch_partial
            self.batch_eval = batch_eval
            self.checkpoint_dir = checkpoint_dir
            self.n_max_config = n_max_config
            self.use_auth_token = use_auth_token
            self.torch_dtype = torch_dtype
            self.device_map = device_map
            self.low_cpu_mem_usage = low_cpu_mem_usage
            
            self.split, self.metric = metric.split('/')

            self.dynamic_config = {
                'lr': lr,
                'label_smoothing': label_smoothing,
                'random_seed': random_seed,
                'gradient_accumulation_steps': gradient_accumulation_steps,
            }
            
            self.dynamic_configs = list(self.dynamic_config.values())


            """
            self.all_dynamic_configs = list(product(
                self.dynamic_config['lr'],
                self.dynamic_config['label_smoothing'],
                self.dynamic_config['random_seed'],
                self.dynamic_config['gradient_accumulation_steps'],
            ))
            """
            
        def initialize_searcher(self):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            path_to_config = pj(self.checkpoint_dir, 'config_static.json')
            if os.path.exists(path_to_config):
                with open(path_to_config) as f:
                    tmp = json.load(f)
                tmp_v = [tmp[k] for k in sorted(tmp.keys())]
                static_tmp_v = [self.static_config[k] for k in sorted(tmp.keys())]
                assert tmp_v == static_tmp_v, f'{str(tmp_v)}\n not matched \n{str(static_tmp_v)}'
            path_to_d_config = pj(self.checkpoint_dir, 'config_dynamic.json')
            if os.path.exists(path_to_d_config):
                with open(path_to_d_config) as f:
                    tmp = json.load(f)

                tmp_v = [tmp[k] for k in sorted(tmp.keys())]
                dynamic_tmp_v = [self.dynamic_config[k] for k in sorted(tmp.keys())]

                assert tmp_v == dynamic_tmp_v
            path_to_e_config = pj(self.checkpoint_dir, 'config_eval.json')
            if os.path.exists(path_to_e_config):
                with open(path_to_e_config) as f:
                    tmp = json.load(f)
                tmp_v = [tmp[k] for k in sorted(tmp.keys())]
                eval_tmp_v = [self.eval_config[k] for k in sorted(tmp.keys())]
                assert tmp_v == eval_tmp_v, f'{str(tmp_v)}\n not matched \n{str(eval_tmp_v)}'

            with open(path_to_config, 'w') as f:
                json.dump(self.static_config, f)
            with open(path_to_d_config, 'w') as f:
                json.dump(self.dynamic_config, f)
            with open(path_to_e_config, 'w') as f:
                json.dump(self.eval_config, f)

            # add file handler
            logger = logging.getLogger()
            file_handler = logging.FileHandler(pj(self.checkpoint_dir, 'grid_search.log'))
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
            logger.addHandler(file_handler)
            logging.info(f'INITIALIZE GRID SEARCHER: {len(self.dynamic_config)} configs to try')

        def get_evaluator(self, overwrite: bool):
            # configure evaluation data typeseval
            input_types = to_list(self.static_config['input_types'], sorting=False)
            output_types = to_list(self.static_config['output_types'], sorting=False)
            instruction_types = to_list(self.static_config['instruction_types'], sorting=False)
            assert len(input_types) == len(output_types), f"{len(input_types)} != {len(output_types)}"
            if self.static_config['prefix_types'] is None:
                prefix_types = [None] * len(input_types)
            else:
                prefix_types = to_list(self.static_config['prefix_types'], sorting=False)
                
                
            tmp = [(i, o, p, instruct) for i, o, p, instruct in zip(input_types, output_types, prefix_types, instruction_types)]
            if len(tmp) > 1:
                tmp = [(i, o, p, instruct) for i, o, p, instruct in tmp if o in ['question', 'questions_answers']]
            assert len(tmp) == 1, tmp
            i, o, p, instruct= tmp[0]
            prefix = pj(
                DEFAULT_CACHE_DIR,
                "encoded_feature",
                f"{self.static_config['dataset_path']}{'.' + self.static_config['dataset_name'] if self.static_config['dataset_name'] != 'default' else ''}"
                f"{self.static_config['model']}.{self.eval_config['max_length_eval']}.{self.eval_config['max_length_output_eval']}.{i}.{o}"
            )
            data_cache_paths = {split: f"{prefix}.{split}.{p}.pkl" for split in ['test', 'validation']}

            def get_evaluation(_checkpoint_dir_model):
                return evaluate(
                    export_dir=pj(_checkpoint_dir_model, 'eval'),
                    batch_size=self.batch_eval,
                    n_beams=self.eval_config['n_beams_eval'],
                    model=_checkpoint_dir_model,
                    model_adapter=self.static_config['model_adapter'],
                    lora_rank=self.static_config['lora_rank'], 
                    lora_alpha=self.static_config['lora_alpha'], 
                    max_length=self.eval_config['max_length_eval'],
                    overwrite=overwrite,
                    max_length_output=self.eval_config['max_length_output_eval'],
                    bleu_only=True, 
                    dataset_path=self.static_config['dataset_path'],
                    dataset_name=self.static_config['dataset_name'],
                    input_type=i,
                    output_type=o,
                    instruction_type=instruct,
                    prediction_aggregation=self.eval_config['prediction_aggregation'],
                    prediction_level=self.eval_config['prediction_level'],
                    language=self.eval_config['language'],
                    use_auth_token=self.use_auth_token,
                    data_caches=data_cache_paths)
            return get_evaluation

        def run(self, interval: int = 25, overwrite: bool = False):

            self.initialize_searcher()

            # instantiate evaluator
            evaluator = self.get_evaluator(overwrite)

            checkpoints = []
            ckpt_exist = {}
            for trainer_config in glob.glob(pj(self.checkpoint_dir, 'model_*', 'trainer_config.json')):
                with open(trainer_config, 'r') as f:
                    ckpt_exist[os.path.dirname(trainer_config)] = json.load(f)

            logging.info(f'## 1st RUN: Configuration ##')
            config = self.static_config.copy()
        
            tmp_dynamic_config = {
                'lr': self.dynamic_configs[0],
                'label_smoothing': self.dynamic_configs[1],
                'random_seed': self.dynamic_configs[2],
                'gradient_accumulation_steps': self.dynamic_configs[3]
            }
            
            config.update(tmp_dynamic_config)
            
            # pdb.set_trace()
            
            ex_dynamic_config = [(k_, [v[k] for k in sorted(tmp_dynamic_config.keys())]) for k_, v in ckpt_exist.items()]
            tmp_dynamic_config = [tmp_dynamic_config[k] for k in sorted(tmp_dynamic_config.keys())]
            duplicated_ckpt = [k for k, v in ex_dynamic_config if v == tmp_dynamic_config]
            if len(duplicated_ckpt) == 1:
                logging.info(f'skip as the config exists at {duplicated_ckpt} \n{config}')
                checkpoint_dir = duplicated_ckpt[0]
                model_dir = self.checkpoint_dir
            elif len(duplicated_ckpt) == 0:
                ckpt_name_exist = [os.path.basename(k).replace('model_', '') for k in ckpt_exist.keys()]
                ckpt_name_made = [os.path.basename(c).replace('model_', '') for c in checkpoints]
                model_ckpt = get_random_string(exclude=ckpt_name_exist + ckpt_name_made)
                model_dir = self.checkpoint_dir
                checkpoint_dir = pj(self.checkpoint_dir, f'model_{model_ckpt}')
                
            else:
                raise ValueError(f'duplicated checkpoints are found: \n {duplicated_ckpt}')

            
            if not os.path.exists(pj(checkpoint_dir, f'epoch_{self.epoch}')):
                trainer = Trainer(
                    checkpoint_dir=checkpoint_dir, model_dir=model_dir, disable_log=True, use_auth_token=self.use_auth_token,
                    device_map=self.device_map, low_cpu_mem_usage=self.low_cpu_mem_usage, torch_dtype=self.torch_dtype,
                    **config)
                trainer.train(model_dir=model_dir, epoch_save=1, interval=interval)

            checkpoints.append(checkpoint_dir)

                
    trainer = GridSearcher(
        checkpoint_dir=args.checkpoint_dir,
        project=args.project,
        entity=args.entity,
        name=args.name,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        input_types=args.input_types,
        output_types=args.output_types,
        prefix_types=args.prefix_types,
        instruction_types=args.instruction_types,
        model=args.model,
        model_adapter=args.model_adapter, 
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_length=args.max_length,
        max_length_output=args.max_length_output,
        epoch=args.epoch,
        epoch_partial=args.epoch_partial,
        batch=args.batch,
        batch_eval=args.batch_eval,
        n_max_config=args.n_max_config,
        fp16=args.fp16,
        random_seed=args.random_seed,
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        lr=args.lr,
        label_smoothing=args.label_smoothing,
        language=args.language
    )
    
    trainer.run()
    
    
    wandb.finish()
    
if __name__ == "__main__":
    main()
