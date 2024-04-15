""" Data utility """
import os
import requests
from os.path import join as pj

from datasets import load_dataset
from .language_model_for_training import internet_connection


__all__ = ('get_dataset', 'get_reference_files', 'DEFAULT_CACHE_DIR')

DEFAULT_CACHE_DIR = pj(os.path.expanduser('~'), '.cache', 'lmqg')
# dataset requires custom reference file
DATA_NEED_CUSTOM_REFERENCE = ['lmqg/qg_squad']


def get_dataset(path: str = 'lmqg/qg_squad',
                name: str = 'default',
                split: str = 'train',
                input_type: str = 'paragraph_answer',
                output_type: str = 'answer',
                instruction_type: str = 'instruction',
                use_auth_token: bool = False):
    """ Get question generation input/output list of texts. """
    name = None if name == 'default' else name
    if 'half' not in path:
        data_files = {"train": "train_instruction.csv",  "test": "test_instruction.csv", "validation": "dev_instruction.csv"}
    elif 'half' in path:
        data_files = {"train": "train_half_instruction.csv", "test": "test_half_instruction.csv", "validation": "dev_half_instruction.csv"}
    dataset = load_dataset(path, data_files=data_files, split=split, use_auth_token=use_auth_token)
    # dataset = load_dataset(path, name, split=split, use_auth_token=use_auth_token)
    return dataset[input_type], dataset[output_type], dataset[instruction_type]


def get_reference_files(path: str = 'lmqg/qg_squad', name: str = 'default', cache_dir: str = None,
                        instruction_type: str = None):
    """ Get reference files for automatic evaluation """
    name = None if name == 'default' else name
    local_files_only = not internet_connection()
    cache_dir = pj(DEFAULT_CACHE_DIR, 'reference_files', 'lmqg/qg_squad') if cache_dir is None else cache_dir
    output = {} 
    instruction_type = 'instruction'
    if instruction_type == 'instruction':
        url = f'https://huggingface.co/datasets/{path}/raw/main/reference_files'
        for split in ['test', 'validation']:

            if 'half' not in path:
                data_files = {"test": "test_instruction.csv", "validation": "dev_instruction.csv"}
            elif 'half' in path:
                data_files = {"test": "test_half_instruction.csv", "validation": "dev_half_instruction.csv"}

            dataset = load_dataset(path, data_files=data_files, split=split)          
            for feature in ['answer', 'question', 'paragraph', 'sentence', 'questions_answers']:
                if feature not in dataset.features:
                    continue
                filename = f'{feature}-{split}.txt' if name is None else f'{feature}-{split}.{name}.txt'
                ref_path = pj(cache_dir, filename)
                os.makedirs(os.path.dirname(ref_path), exist_ok=True)
                """
                if 'lmqg/qg_squad' in DATA_NEED_CTOM_REFERENCE:
                    if not os.path.exists(ref_path):
                        assert not local_files_only, f'network is not reachable, could not download the file from {url}/{filename}'
                        r = requests.get(f'{url}/{filename}')
                        content = r.content
                        assert "Entry not found" not in str(content) and content != b'', content
                        with open(ref_path, "wb") as f:
                            f.write(content)
                        with open(ref_path) as f:
                            assert len(f.read().split('\n')) > 20, f"invalid file {ref_path}"
                else:
                """
                with open(ref_path, 'w') as f:
                    f.write('\n'.join([i.replace('\n', '.') for i in dataset[feature]]))
                assert os.path.exists(ref_path)
                output[f'{feature}-{split}'] = ref_path
                
    elif instruction_type != 'instruction':
        url = f'https://huggingface.co/datasets/{path}/raw/main/reference_files'
        for split in ['test', 'validation']:
            dataset = load_dataset(path, name, split=split)
            for feature in ['answer', 'question', 'paragraph', 'sentence', 'questions_answers']:
                if feature not in dataset.features:
                    continue
                filename = f'{feature}-{split}.txt' if name is None else f'{feature}-{split}.{name}.txt'
                ref_path = pj(cache_dir, filename)
                os.makedirs(os.path.dirname(ref_path), exist_ok=True)
                if path in DATA_NEED_CUSTOM_REFERENCE:
                    if not os.path.exists(ref_path):
                        assert not local_files_only, f'network is not reachable, could not download the file from {url}/{filename}'
                        r = requests.get(f'{url}/{filename}')
                        content = r.content
                        assert "Entry not found" not in str(content) and content != b'', content
                        with open(ref_path, "wb") as f:
                            f.write(content)
                        with open(ref_path) as f:
                            assert len(f.read().split('\n')) > 20, f"invalid file {ref_path}"
                else:
                    with open(ref_path, 'w') as f:
                        f.write('\n'.join([i.replace('\n', '.') for i in dataset[feature]]))
                assert os.path.exists(ref_path)
                output[f'{feature}-{split}'] = ref_path
    return output
