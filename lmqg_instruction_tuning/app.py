""" API with huggingface inference API """
import os
import logging
import random
import traceback
from difflib import SequenceMatcher
from typing import List
from itertools import chain

from transformers import AutoConfig
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from lmppl import EncoderDecoderLM

from lmqg.inference_api import generate_qa
from lmqg.spacy_module import SpacyPipeline

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

API_TOKEN = os.getenv("API_TOKEN")
# local model for qa pair scoring
SCORER_MODEL = os.getenv("SCORER_MODEL", "lmqg/t5-small-squad-qag")
SCORER = EncoderDecoderLM(SCORER_MODEL, max_length_decoder=32, max_length_encoder=256)

##########
# CONFIG #
##########
# Default models for each qag type
DEFAULT_MODELS_E2E = {
    "en": ["lmqg/t5-small-squad-qag", None],
    "ja": ["lmqg/mt5-base-jaquad-qag", None],
    "de": ["lmqg/mbart-large-cc25-dequad-qag", None],
    "es": ["lmqg/mt5-small-esquad-qag", None],
    "it": ["lmqg/mt5-small-itquad-qag", None],
    "ko": ["lmqg/mbart-large-cc25-koquad-qag", None],
    "fr": ["lmqg/mt5-small-frquad-qag", None],
    "ru": ["lmqg/mt5-base-ruquad-qag", None]}
DEFAULT_MODELS_MULTITASK = {
    "en": ["lmqg/t5-small-squad-qg-ae", "lmqg/t5-small-squad-qg-ae"],
    "ja": ["lmqg/mt5-small-jaquad-qg-ae", "lmqg/mt5-small-jaquad-qg-ae"],
    "de": ["lmqg/mt5-small-dequad-qg-ae", "lmqg/mt5-small-dequad-qg-ae"],
    "es": ["lmqg/mt5-small-esquad-qg-ae", "lmqg/mt5-small-esquad-qg-ae"],
    "it": ["lmqg/mt5-small-itquad-qg-ae", "lmqg/mt5-small-itquad-qg-ae"],
    "ko": ["lmqg/mt5-small-koquad-qg-ae", "lmqg/mt5-small-koquad-qg-ae"],
    "fr": ["lmqg/mt5-small-frquad-qg-ae", "lmqg/mt5-small-frquad-qg-ae"],
    "ru": ["lmqg/mt5-small-ruquad-qg-ae", "lmqg/mt5-small-ruquad-qg-ae"]}
DEFAULT_MODELS_PIPELINE = {
    "en": ["lmqg/t5-small-squad-qg", "lmqg/t5-small-squad-ae"],
    "ja": ["lmqg/mt5-small-jaquad-qg", "lmqg/mt5-small-jaquad-ae"],
    "de": ["lmqg/mt5-small-dequad-qg", "lmqg/mt5-small-dequad-ae"],
    "es": ["lmqg/mt5-small-esquad-qg", "lmqg/mt5-small-esquad-ae"],
    "it": ["lmqg/mt5-small-itquad-qg", "lmqg/mt5-small-itquad-ae"],
    "ko": ["lmqg/mt5-small-koquad-qg", "lmqg/mt5-small-koquad-ae"],
    "fr": ["lmqg/mt5-small-frquad-qg", "lmqg/mt5-small-frquad-ae"],
    "ru": ["lmqg/mt5-small-ruquad-qg", "lmqg/mt5-small-ruquad-qg"]}
DEFAULT_MODELS = {"End2End": DEFAULT_MODELS_E2E, "Pipeline": DEFAULT_MODELS_PIPELINE, "Multitask": DEFAULT_MODELS_MULTITASK}
# Other configs
LANGUAGE_MAP = {
    "English": "en",
    "Japanese": "ja",
    'German': "de",
    'Spanish': 'es',
    'Italian': 'it',
    'Korean': 'ko',
    'Russian': "ru",
    'French': "fr"}
SPACY_PIPELINE = {i: SpacyPipeline(i) for i in LANGUAGE_MAP.values()}  # spacy for sentence splitting
# QAG model pretty names used in frontend
PRETTY_NAME = {
    "T5 SMALL": "lmqg/t5-small-squad",
    "T5 BASE": "lmqg/t5-base-squad",
    "T5 LARGE": "lmqg/t5-large-squad",
    "Flan-T5 SMALL": "lmqg/flan-t5-small-squad",
    "Flan-T5 BASE": "lmqg/flan-t5-base-squad",
    "Flan-T5 LARGE": "lmqg/flan-t5-large-squad",
    "BART BASE": "lmqg/bart-base-squad",
    "BART LARGE": "lmqg/bart-large-squad"}
PRETTY_NAME.update({f'mT5 SMALL ({i.upper()})': f'lmqg/mt5-small-{i}quad' for i in LANGUAGE_MAP.values() if i != 'en'})
PRETTY_NAME.update({f'mT5 BASE ({i.upper()})': f'lmqg/mt5-base-{i}quad' for i in LANGUAGE_MAP.values() if i != 'en'})
PRETTY_NAME.update({f'mBART LARGE ({i.upper()})': f'lmqg/mbart-large-cc25-{i}quad' for i in LANGUAGE_MAP.values() if i != 'en'})
# Prefix information for each model
PREFIX_INFO_QAG = {v: AutoConfig.from_pretrained(v).add_prefix for v in chain(*chain(*[v.values() for v in DEFAULT_MODELS.values()])) if v is not None}


########
# MAIN #
########
# App input
class ModelInput(BaseModel):
    input_text: str
    language: str = 'en'
    qag_type: str = None  # End2End/Pipeline/Multitask
    model: str = None
    highlight: str or List = None  # answer
    num_beams: int = 4
    use_gpu: bool = False
    do_sample: bool = True
    top_p: float = 0.9
    max_length: int = 64
    split: str = 'paragraph'


# Run app
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def validate_default(value, default=None):
    if value is not None and type(value) is not str:
        return value
    if value is None or value.lower() in ['default', '']:
        return default
    return value


############
# ENDPOINT #
############
# General info
@app.get("/")
def read_root():
    return {"About": "Automatic question & answer generation application. Send inquiry to https://asahiushio.com/."}


# Main endpoint
@app.post("/question_generation")
async def process(model_input: ModelInput):
    if len(model_input.input_text) == 0:
        raise HTTPException(status_code=404, detail='Input text is empty string.')
    try:
        # language validation
        if model_input.language in LANGUAGE_MAP:
            model_input.language = LANGUAGE_MAP[model_input.language]
        # qag_type validation
        elif model_input.qag_type == 'End2End' and model_input.highlight is not None:
            raise ValueError("End2End qag_type does not support answer-given question generation")

        model_input.highlight = validate_default(model_input.highlight, default=None)
        model_input.qag_type = validate_default(
            model_input.qag_type,
            default='Multitask'
            # default='End2End' if model_input.highlight is None and model_input.language == 'en' else 'Multitask'
        )
        model_input.model = validate_default(model_input.model, default=None)

        # model validation
        if model_input.model is None:
            model_qg, model_ae = DEFAULT_MODELS[model_input.qag_type][model_input.language]
        else:
            model_base = PRETTY_NAME[model_input.model]
            if model_input.qag_type == 'End2End':
                model_qg, model_ae = f"{model_base}-qag", None
            elif model_input.qag_type == 'Pipeline':
                model_qg, model_ae = f"{model_base}-qg", f"{model_base}-ae"
            elif model_input.qag_type == 'Multitask':
                model_qg, model_ae = f"{model_base}-qg-ae", f"{model_base}-qg-ae"
            else:
                raise ValueError(f"unknown qag_type: {model_input.qag_type}")
        qa_list = generate_qa(
            api_token=API_TOKEN,
            input_text=model_input.input_text,
            model_qg=model_qg,
            model_ae=model_ae,
            spacy=SPACY_PIPELINE[model_input.language],
            input_answer=model_input.highlight,
            top_p=model_input.top_p,
            use_gpu=model_input.use_gpu,
            do_sample=model_input.do_sample,
            max_length=model_input.max_length,
            num_beams=model_input.num_beams,
            is_qag=model_input.qag_type == 'End2End',
            add_prefix_qg=PREFIX_INFO_QAG[model_qg] if model_qg in PREFIX_INFO_QAG else None,
            add_prefix_answer=PREFIX_INFO_QAG[model_ae] if model_ae is not None and model_ae in PREFIX_INFO_QAG else None,
            split_level=model_input.split.lower()
        )
        if model_input.language == 'en':
            score = SCORER.get_perplexity(
                input_texts=[model_input.input_text] * len(qa_list),
                output_texts=[f"question: {x['question']}, answer: {x['answer']}" for x in qa_list]
            )
            # perplexity is positive value, so we transform it into unit interval with reverse order (higher is better)
            for s, qa in zip(score, qa_list):
                qa['score'] = 1 / (1 + s)
        else:
            for qa in qa_list:
                q = qa['question']
                c = model_input.input_text
                match = SequenceMatcher(None, c, q).find_longest_match(0, len(c), 0, len(q))
                qa['score'] = 1 - match.size / len(q)
        qa_list = sorted(qa_list, key=lambda x: x['score'], reverse=True)
        return {'qa': qa_list}
    except Exception:
        logging.exception('Error')
        raise HTTPException(status_code=404, detail=traceback.print_exc())


# Dummy endpoint
@app.post("/question_generation_dummy")
async def process(model_input: ModelInput):
    i = random.randint(0, 2)
    target = [{'question': "Who founded Nintendo Karuta?", 'answer': "Fusajiro Yamauchi", "score": 0.5},
              {'question': "When did Nintendo distribute its first video game console, the Color TV-Game?", 'answer': "1977", "score": 0.3},
              {'question': "When did Nintendo release Super Mario Bros?", 'answer': "1985", "score": 0.1}]
    return {"qa": target[:i]}
