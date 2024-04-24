# Question Answering with LoRA
`HuggingFace`  `Weight & Biases`  `Matplotlib` `Openpyxl`
<br></br><br><br><br>

## Introduction
This project is practice code for 한국소프트웨어종합학술대회(KSC) 2023 질문 생성 성능 향상을 위한 대규모 언어 모델 Post-training 적용 방법 [[Paper](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11705119)].

<br><br>

## LMQG + LoRA
[asahi417/lm-question-generation](https://github.com/asahi417/lm-question-generation)
<br>[LMQG](https://huggingface.co/lmqg)
<br>[LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora)
<br><br><br> Hyperparamter Tuning Plan (09/07 ~ 0913)

| Sever  | GPU      | Model           | Lora R          | Estimated  Runtime         | Estimated  Date         | 
| :------: | :--------: | :---------------: | :---------------: | :---------------: | :---------------: |
| 01  | device=0 | [google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl)   | 64 | 3 Day | 09/10 |
| 01 | device=1 | [google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl)   | 128      | 3 Day | 09/10 |
| 02 | device=0 | [google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl) | 4   | 5 Day | 09/12 |
| 02 | device=1 | [google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl) | 8   | 5 Day | 09/12 |
| 03  | device=0 | [google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl)   | 16               | 6 Day | 09/13 |
| 04  | device=0 | [google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl)    | 32               | 6 Day | 09/13 |

<br></br><br><br><br>

## LMQG + Insturction Tuning
[bigscience-workshop/promptsource](https://github.com/bigscience-workshop/promptsource)
<br>[allenai/natural-instructions](https://github.com/allenai/natural-instructions)

<b>Folder Information</b>
<br>lmqg:Offcial Github Folder
<br>lmqg_collate_fn: Instruction-Tuning + Collate_fn
<br>lmqg_collate_fn_inference: Instruction-Tuning Inference + Collate_fn
<br>lmqg_post_non_lora: Fine-Tuning
<br>lmqg_original: Fine-Tuning Evaluation
<br>lmqg_inference: Fine-Tuning Inference
<br></br><br><br><br>

## Reference
A Survey of Large Language Models
<br>Generative Language Models for Paragraph-Level Question Generation
<br>An Empirical Comparison of LM-based Question and Answer Generation Methods
<br>A Practical Toolkit for Multilingual Question and Answer Generation
<br>Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
<br>Finetuned Language Models Are Zero-Shot Learners
<br>LoRA: Low-Rank Adaptation of Large Language Models
<br>Alpaca: Intermittent Execution without Checkpoints
<br>Instruction Tuning with GPT-4
<br></br>
