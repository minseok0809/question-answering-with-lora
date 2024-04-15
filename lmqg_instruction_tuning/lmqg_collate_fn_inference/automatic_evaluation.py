""" SQuAD QG evaluation (sentence/answer level) """
import re
import json
import logging
import os
import pandas as pd
from collections import defaultdict
from os.path import join as pj
from typing import Dict, List
import torch
from .automatic_evaluation_tool.bleu.bleu import Bleu
from .automatic_evaluation_tool.meteor.meteor import Meteor
from .automatic_evaluation_tool.rouge import Rouge
from .automatic_evaluation_tool.bertscore import BERTScore
from .automatic_evaluation_tool.moverscore import MoverScore
from .automatic_evaluation_tool.qa_aligned_f1_score import QAAlignedF1Score
from .spacy_module import SpacyPipeline
from .language_model import TransformersQG
from .data import get_reference_files, get_dataset

import pdb

LANG_NEED_TOKENIZATION = ['ja', 'zh']

def compute_metrics(out_file,
                    tgt_file,
                    src_file: str = None,
                    prediction_aggregation: str = 'first',
                    bleu_only: bool = True,
                    language: str = 'en',
                    skip: List = None,
                    qag_model: bool = False):
    """ compute automatic metric """
    spacy_model = SpacyPipeline(language=language) if language in LANG_NEED_TOKENIZATION else None
    pairs = []
    pairs_raw = []
    
    with open(tgt_file, "r") as infile:
        for n, line in enumerate(infile):
            if line.endswith('\n'):
                line = line[:-1]
            pairs_raw.append({'tokenized_question': line.strip(), 'tokenized_sentence': n})
            if spacy_model is None:
                pairs.append({'tokenized_question': line.strip(), 'tokenized_sentence': n})
            else:
                pairs.append({'tokenized_question': ' '.join(spacy_model.token(line.strip())), 'tokenized_sentence': n})
        
    if src_file is not None and prediction_aggregation is not None:
        # group by the source (sentence where the question are produced); this is used for grouping but not evaluation
        with open(src_file, 'r') as infile:
            for n, line in enumerate(infile):
                if line.endswith('\n'):
                    line = line[:-1]
                pairs_raw[n]['tokenized_sentence'] = line.strip().lower()
                if spacy_model is None:
                    pairs[n]['tokenized_sentence'] = line.strip().lower()
                else:
                    pairs[n]['tokenized_sentence'] = ' '.join(spacy_model.token(line.strip().lower()))

    # fix prediction's tokenization: lower-casing and detaching sp characters
    with open(out_file, 'r') as infile:
        for n, line in enumerate(infile):
            if line.endswith('\n'):
                line = line[:-1]
            pairs_raw[n]['prediction'] = line.strip()
            if spacy_model is None:
                pairs[n]['prediction'] = line.strip()
            else:
                pairs[n]['prediction'] = ' '.join(spacy_model.token(line.strip()))

    # sprint("Pairs Length: ", len(pairs))
    # eval
    json.encoder.FLOAT_REPR = lambda o: format(o, '.4f')

    def format_pair(_pairs):
        res = defaultdict(lambda: [])
        gts = defaultdict(lambda: [])

        for idx, pair in enumerate(_pairs):
            # if idx <= 10: # (revised)
            # key is the sentence where the model generates the question
            key = pair['tokenized_sentence']

            # one generation per sentence
            if 'prediction' in pair:
                pred = pair['prediction'].encode('utf-8')
            else:
                logging.warning('prediction not found at the evaluation')
                pred = ''.encode('utf-8')
            res[key].append(pred)

            # multiple gold question per sentence
            gts[key].append(pair['tokenized_question'].encode('utf-8'))

        res_filtered = defaultdict(lambda: [])
        for k, v in res.items():
            # answer-level evaluation
            if prediction_aggregation is None:
                assert len(v) == 1
                res_filtered[k] = v
            elif prediction_aggregation == 'first':
                # the first one
                res_filtered[k] = [v[0]]
            elif prediction_aggregation == 'last':
                # the last one
                res_filtered[k] = [v[-1]]
            elif prediction_aggregation == 'long':
                # the longest generation
                res_filtered[k] = [v[v.index(sorted(v, key=len)[-1])]]
            elif prediction_aggregation == 'short':
                # the shortest generation
                res_filtered[k] = [v[v.index(sorted(v, key=len)[0])]]
            elif prediction_aggregation == 'middle':
                # middle length generation
                res_filtered[k] = [v[v.index(sorted(v, key=len)[int(len(v)/2)])]]
            else:
                raise ValueError(f'unknown aggregation method: {prediction_aggregation}')
        return gts, res_filtered

    output = {}
    
    scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])]
 
    """
    if not bleu_only:
        scorers_extra = [
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (BERTScore(language=language), "BERTScore"),
            (MoverScore(language=language), 'MoverScore')
        ]
        if qag_model:
            scorers_extra += [
                (QAAlignedF1Score(target_metric='f1', base_metric='bertscore', language=language),
                 "QAAlignedF1Score (BERTScore)"),
                (QAAlignedF1Score(target_metric='recall', base_metric='bertscore', language=language),
                 "QAAlignedRecall (BERTScore)"),
                (QAAlignedF1Score(target_metric='precision', base_metric='bertscore', language=language),
                 "QAAlignedPrecision (BERTScore)"),
                (QAAlignedF1Score(target_metric='f1', base_metric='moverscore', language=language),
                 "QAAlignedF1Score (MoverScore)"),
                (QAAlignedF1Score(target_metric='recall', base_metric='moverscore', language=language),
                 "QAAlignedRecall (MoverScore)"),
                (QAAlignedF1Score(target_metric='precision', base_metric='moverscore', language=language),
                 "QAAlignedPrecision (MoverScore)")
            ]
        if skip is not None:
            scorers_extra = [s for s in scorers_extra if s[1] not in skip]
        scorers += scorers_extra
    """
    """
    prediction_df_path = 'prediction/bleu_validation.xlsx'
    if os.path.isfile(prediction_df_path) == True:
        prediction_df_path = 'prediction/bleu_validation.xlsx'
        prediction_df = pd.DataFrame({"Valid Blue 1": [0.0], "Valid Blue 2": [0.0], "Valid Blue 3": [0.0], "Valid Blue 4": [0.0]})     
        
    elif os.path.isfile(prediction_df_path) == False: 
        prediction_df_path = 'prediction/bleu_test.xlsx'
        prediction_df = pd.DataFrame({"Test Blue 1": [0.0], "Test Blue 2": [0.0], "Test Blue 3": [0.0], "Test Blue 4": [0.0]})    
    """
    gts_, res_filtered_ = format_pair(pairs)
    gts_raw, res_filtered_raw = format_pair(pairs_raw)
    for scorer, method in scorers:
        if type(method) is str and method.startswith('QAAligned'):
            score, scores = scorer.compute_score(gts_raw, res_filtered_raw)
        else:
            score, scores = scorer.compute_score(gts_, res_filtered_)
            # prediction_df.loc[num] = [scores[0], scores[1], scores[2], scores[3]]
        torch.cuda.empty_cache()
        if type(method) is not list:
            score, scores, method = [score], [scores], [method]
        for sc, scs, m in zip(score, scores, method):
            logging.info(f"\t{m}: {sc}")
            output[m] = sc
    
    # prediction_df.to_excel(prediction_df_path, index=False)
    # prediction_df.to_csv(prediction_df_path.replace("xlsx", "csv"), index=False)       
    return output


def evaluate(export_dir: str = '.',
             prediction_file: str = None,
             batch_size: int = 32,
             n_beams: int = 4,
             hypothesis_file_dev: str = None,
             hypothesis_file_test: str = None,
             model: str = None,
             model_adapter: bool = True,
             lora_rank:  int = 8,
             lora_alpha:  int = 32,
             max_length: int = 512,
             max_length_output: int = 64,
             dataset_path: str = 'lmqg/qg_squad',
             dataset_name: str = 'default',
             input_type: List or str = 'paragraph_answer',
             output_type: List or str = 'answer',
             question_type: List or str = 'quesiton',
             prefix_type: List or str = 'prefix',
             task_type: List or str = 'task',
             instruction_type: List or str = 'instruction',             
             prediction_aggregation: str = 'first',
             prediction_level: str = None,
             data_caches: Dict = None,
             bleu_only: bool = True,
             overwrite: bool = False,
             use_auth_token: bool = False,
             torch_dtype=None,
             device_map: str = None,
             low_cpu_mem_usage: bool = False,
             language: str = 'en',
             test_split: str = 'test',
             validation_split: str = 'validation'):
    """ Evaluate question-generation model """
    
    # pdb.set_trace()
    
    reference_files = get_reference_files(dataset_path, dataset_name)
    # print("Reference Files")
    
    if prediction_level is None:
        valid_prediction_level = [k.split('-')[0] for k in reference_files.keys()]
        if 'sentence' in valid_prediction_level:
            prediction_level = 'sentence'
        elif 'answer' in valid_prediction_level:
            prediction_level = 'answer'
        elif "questions_answers" in valid_prediction_level:
            prediction_level = 'questions_answers'
        else:
            raise ValueError(f"unexpected error: {valid_prediction_level}")
    path_metric = pj(export_dir, f'metric.{prediction_aggregation}.{prediction_level}.{input_type}.{output_type}.{dataset_path.replace("/", "_")}.{dataset_name}.json')
    metric = {}
    if not overwrite and os.path.exists(path_metric):
        with open(path_metric, 'r') as f:
            metric = json.load(f)
            if bleu_only:
                return metric
    os.makedirs(export_dir, exist_ok=True)
    
    # print("Before TransformerQG")
    
    if model is not None:
        lm = TransformersQG(model,
                            model_adapter,
                            lora_rank,
                            lora_alpha,
                            max_length=max_length,
                            max_length_output=max_length_output,
                            drop_overflow_error_text=False,
                            skip_overflow_error=True,
                            language=language,
                            use_auth_token=use_auth_token,
                            torch_dtype=torch_dtype,
                            device_map=device_map,
                            low_cpu_mem_usage=low_cpu_mem_usage)
        lm.eval()
    
     
        def get_model_prediction_file(export_dir, prediction_file, batch_size, split, _input_type, _output_type, _question_type, _prefix_type, _task_type, _instruction_type):
            path = pj(
                export_dir,
                f'samples.{split}.hyp.{_input_type}.{_output_type}.{dataset_path.replace("/", "_")}.{dataset_name}.txt')
            """
            raw_input, text_output, question_output, task_output, instruct_output = get_dataset(
                dataset_path,
                dataset_name,
                split=split,
                input_type=_input_type,
                output_type=_output_type,
                question_type=_question_type,
                task_type=_task_type,
                instruction_type=_instruction_type,
                use_auth_token=use_auth_token)
           
            if os.path.exists(path) and not overwrite:
                with open(path) as f_read:
                    tmp = f_read.read().split('\n')
                if len(tmp) == len(raw_input):
                    return path
                logging.warning(f'recompute {path}: {len(tmp)} != {len(raw_input)}')
             """
            prefix_type = None
            if lm.add_prefix:
                if _output_type == 'questions_answers':
                    prefix_type = 'qag'
                elif _output_type == 'question':
                    prefix_type = 'qg'
                elif _output_type == 'answer':
                    prefix_type = 'ae'
                else:
                    raise ValueError(f"prefix type is not determined for the output_type {_output_type}")
            # print("Question:", question_output)
            epoch_dir = re.findall('epoch_[0-9]+', export_dir)[0]
            output = lm.generate_prediction(epoch_dir, prediction_file, batch_size, split, dataset_path=dataset_path, dataset_name=dataset_name,
                                            input_types=_input_type, output_types=_output_type, 
                                            task_types=_task_type, question_types=_question_type, instruction_types=_instruction_type,
                                            cache_path=None if data_caches is None else data_caches[split])
            with open(path, 'w') as f_writer:
                f_writer.write('\n'.join(output))
            return path


        try:
            hypothesis_file_test = get_model_prediction_file(export_dir, prediction_file, batch_size, test_split, input_type, output_type, prefix_type, question_type, task_type, instruction_type)
        except ValueError:
            hypothesis_file_test = None
        try:
            hypothesis_file_dev = get_model_prediction_file(export_dir, prediction_file, batch_size, validation_split, input_type, output_type, prefix_type, question_type, task_type, instruction_type)
        except ValueError:
            hypothesis_file_dev = None

    assert hypothesis_file_dev is not None or hypothesis_file_test is not None,\
        f'model ({model}) or file path ({hypothesis_file_dev}, {hypothesis_file_test}) is needed'

    def get_metric(split, hypothesis_file, skip_metrics):
        assert prediction_level in ['sentence', 'paragraph', 'answer'], prediction_level
        if prediction_level == 'answer':
            src_file = None
        else:
            src_file = reference_files[f'{prediction_level}-{split}']

        __metric = compute_metrics(
            out_file=hypothesis_file,
            tgt_file=reference_files[f'{output_type}-{split}'],
            src_file=src_file,
            prediction_aggregation=prediction_aggregation,
            language=language,
            bleu_only=bleu_only,
            skip=skip_metrics,
            qag_model=output_type == 'questions_answers'
        )
        return __metric

    for _hypothesis_file, _split in zip([hypothesis_file_dev, hypothesis_file_test], [validation_split, test_split]):
        if _hypothesis_file is None:
            continue
        if _split in metric:
            # print("Before get_metric")
            _metric = get_metric(_split, _hypothesis_file, list(metric[_split].keys()))
            metric[_split].update(_metric)
        else:
            metric[_split] = get_metric(_split, _hypothesis_file, None)
    # print("After get_metric")
    with open(path_metric, 'w') as f:
        json.dump(metric, f)

    return metric

