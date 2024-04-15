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
    
    args = easydict.EasyDict({"checkpoint_dir": "tmp_ckpt_flan_t5_large_squad_qg_lora_r8_al32", 
                              
                              "project": "lmqg_qg_squad",
                              
                              "entity": "minseok0809",
                              
                              "name": "flan_t5_large_squad_qg_lora_r8_al32",
                              
                              "dataset_path": "lmqg/qg_squad", 
                              
                              "dataset_name": "default", 
                              
                              "input_types": "paragraph_answer", 
                              
                              "output_types": "question", 
                              
                              "prefix_types": "qg", 
                              
                              "model": "google/flan-t5-large", 
                              
                              "model_adapter": True,   
                              
                              "lora_rank": 8, 
                              
                              "lora_alpha": 32, 

                              "max_length": 512, 
                              
                              "max_length_output": 32, 
                              
                              "epoch": 20,
                              
                              "epoch_partial": 2, 
                              
                              "batch": 4, 
                              
                              "batch_eval": 4,
                              
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
                    prefix_types: List or str = args.prefix_types, model: str = args.model, 
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
                'max_length': max_length, 'max_length_output': max_length_output, 'prefix_types': prefix_types
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
            path_to_config = pj(self.checkpoint_dir, 'config_static.json')
            path_to_d_config = pj(self.checkpoint_dir, 'config_dynamic.json')
            path_to_e_config = pj(self.checkpoint_dir, 'config_eval.json')

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
            assert len(input_types) == len(output_types), f"{len(input_types)} != {len(output_types)}"
            if self.static_config['prefix_types'] is None:
                prefix_types = [None] * len(input_types)
            else:
                prefix_types = to_list(self.static_config['prefix_types'], sorting=False)
            tmp = [(i, o, p) for i, o, p in zip(input_types, output_types, prefix_types)]
            if len(tmp) > 1:
                tmp = [(i, o, p) for i, o, p in tmp if o in ['question', 'questions_answers']]
            assert len(tmp) == 1, tmp
            i, o, p = tmp[0]
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
            elif len(duplicated_ckpt) == 0:
                ckpt_name_exist = [os.path.basename(k).replace('model_', '') for k in ckpt_exist.keys()]
                ckpt_name_made = [os.path.basename(c).replace('model_', '') for c in checkpoints]
                model_ckpt = get_random_string(exclude=ckpt_name_exist + ckpt_name_made)
                model_dir = self.checkpoint_dir
                checkpoint_dir = pj(self.checkpoint_dir, f'model_{model_ckpt}')
                
            else:
                raise ValueError(f'duplicated checkpoints are found: \n {duplicated_ckpt}')

            checkpoints.append(checkpoint_dir)

            metrics = {}
            evaluation_log_df = pd.DataFrame({'Evaluation Time':[0],
                                              'Valid Bleu 1':[0], 'Valid Bleu 2':[0],'Valid Bleu 3':[0], 'Valid Bleu 4':[0], 
                                              'Test Bleu 1':[0], 'Test Bleu 2':[0], 'Test Bleu 3':[0], 'Test Bleu 4':[0]})
            
            for n, checkpoint_dir in enumerate(checkpoints):
                for i in range(1, self.epoch+1):
                    if i % self.epoch_partial == 0:
                        logging.info(f'## 1st RUN (EVAL): Configuration {n}/{len(checkpoints)} ##')
                        
                        
                        checkpoint_dir_model = pj(checkpoint_dir, f'epoch_{i}')

                        start = time.time()
                        try:
                            metric = evaluator(checkpoint_dir_model)
                            
                            bleu_metric = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]
                            valid_metric = metric["validation"]
                            test_metric = metric["test"]
                            valid_bleu_1 = valid_metric[bleu_metric[0]]; valid_bleu_2 = valid_metric[bleu_metric[1]]
                            valid_bleu_3 = valid_metric[bleu_metric[2]]; valid_bleu_4 = valid_metric[bleu_metric[3]]
                            test_bleu_1 = test_metric[bleu_metric[0]]; test_bleu_2 = test_metric[bleu_metric[1]]
                            test_bleu_3 = test_metric[bleu_metric[2]]; test_bleu_4 = test_metric[bleu_metric[3]]
                            
                            wandb.log({"valid_bleu_1": valid_bleu_1, "valid_bleu_2": valid_bleu_2,
                                       "valid_bleu_3": valid_bleu_3, "valid_bleu_4": valid_bleu_4,
                                       "test_bleu_1": test_bleu_1, "test_bleu_2": test_bleu_2,
                                       "test_bleu_3": test_bleu_3, "test_bleu_4": test_bleu_4})

                            
                            # except Exception:
                            # metrics[checkpoint_dir_model] = metric[self.split][self.metric] (revised)
                        except Exception as e:
                            logging.exception("Error while computing metric")
                            print(e)
                            
                        end = time.time()
                        runtime = end - start
                        runtime = str(datetime.timedelta(seconds = runtime)).split(".")[0]
                        print("\n")
                        print("Evaluation Time: ", runtime)
                        print("Valid Bleu 1: ", round(valid_bleu_1, 4), "   ", "Valid Bleu 2:", round(valid_bleu_2, 4))
                        print("Valid Bleu 3: ", round(valid_bleu_3, 4), "   ", "Valid Bleu 4: ", round(valid_bleu_4, 4))
                        print("Test Bleu 1: ", round(test_bleu_1, 4), "   ", "Test Bleu 2: ", round(test_bleu_2, 4))
                        print("Test Bleu 3: ", round(test_bleu_3, 4), "   ", "Test Bleu 4: ", round(test_bleu_4, 4))
                        print("\n")
                        
                        epoch_num = int(checkpoint_dir_model.split("_")[-1])
                        evaluation_log_df.loc[epoch_num] = [runtime, valid_bleu_1, valid_bleu_2, valid_bleu_3, valid_bleu_4,
                                                            test_bleu_1, test_bleu_2, test_bleu_3, test_bleu_4,]
            
                        # print("Metrics: ", metric) (revised)
                        metrics = [[checkpoint_dir_model, metric['validation']['Bleu_4']]]
                        # metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True) (revised)
                        
                        with open(pj(self.checkpoint_dir, f"metric.{self.eval_config['prediction_level']}.{i // self.epoch_partial}.json"), 'w') as f:
                            json.dump(metrics, f)

                        logging.info(f'{i // self.epoch_partial} RUN RESULTS ({self.split}/{self.metric})')
                        for n, (k, v) in enumerate(metrics):
                            logging.info(f'\t * rank: {n} | metric: {round(v, 3)} | model: {k} |')

                        if self.epoch_partial == self.epoch:
                            logging.info('No 2nd phase as epoch_partial == epoch')
                            return
                        
            model_dir = self.checkpoint_dir
            training_log_path = model_dir +"/lmqg_log.xlsx" 
            training_log_df = pd.read_excel(training_log_path, engine='openpyxl') 
            training_log_df = pd.concat([training_log_df, evaluation_log_df], axis = 1)
            training_log_df = training_log_df.drop([0], axis = 0)
            training_log_df.to_excel(training_log_path, index=False)

                
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
