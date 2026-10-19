---
library_name: peft
license: other
base_model: Qwen/Qwen3-8B
tags:
- base_model:adapter:Qwen/Qwen3-8B
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: qwen3-8b-jmed-lora
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen3-8b-jmed-lora

This model is a fine-tuned version of [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) on the jmed_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0034

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.03
- num_epochs: 2.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.0063        | 0.3854 | 100  | 0.0071          |
| 0.0051        | 0.7707 | 200  | 0.0042          |
| 0.0045        | 1.1541 | 300  | 0.0034          |
| 0.0065        | 1.5395 | 400  | 0.0036          |
| 0.0085        | 1.9249 | 500  | 0.0036          |
| 0.0101        | 2.0    | 520  | 0.0036          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.6.0
- Pytorch 2.11.0+cu130
- Datasets 4.0.0
- Tokenizers 0.22.2