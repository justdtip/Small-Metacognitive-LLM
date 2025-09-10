---
license: apache-2.0
datasets:
- RUC-AIBOX/STILL-3-Preview-RL-Data
base_model:
- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
library_name: peft
language:
- en
- zh
pipeline_tag: question-answering
tags:
- reasoning
---

## Introduction

Tina (Tiny Reasoning Models via LoRA) models are all fine-tuned adapters on the base model [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B). 
This LoRA adapter in this repo is fine-tuned with the dataset [RUC-AIBOX/STILL-3-Preview-RL-Data](https://huggingface.co/datasets/RUC-AIBOX/STILL-3-Preview-RL-Data).
Please refer to our paper [Tina: Tiny Reasoning Models via LoRA](https://arxiv.org/abs/2504.15777) for more training details.


## Example Usage

The Tina model is meant to be used in combination with the base model as a standard adapter. Particularly, we release all checkpoints we have for each Tina model and one could select different checkpoint to use by specifying the `subfolder`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
  device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)

model = PeftModel.from_pretrained(
  base_model,
  "Tina-Yi/R1-Distill-Qwen-1.5B-STILL",
  subfolder="checkpoint-2000" # checkpoint 2000 is the best
)
```