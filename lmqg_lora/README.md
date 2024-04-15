# LMQG + LoRA
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

<br></br>

## Reference
A Survey of Large Language Models
<br>Generative Language Models for Paragraph-Level Question Generation
<br>An Empirical Comparison of LM-based Question and Answer Generation Methods
<br>A Practical Toolkit for Multilingual Question and Answer Generation
<br>Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
<br>Finetuned Language Models Are Zero-Shot Learners
<br>LoRA: Low-Rank Adaptation of Large Language Models
<br></br>