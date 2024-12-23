---
library_name: peft
license: llama3.2
base_model: meta-llama/Llama-3.2-3B-Instruct
tags:
- generated_from_trainer
metrics:
- accuracy
- f1
model-index:
- name: llama-3b-offense
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# llama-3b-offense

This model is a fine-tuned version of [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.5348
- Accuracy: 0.7547
- F1: 0.4462

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 128
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 100
- num_epochs: 10

### Training results

| Training Loss | Epoch  | Step | Validation Loss | Accuracy | F1     |
|:-------------:|:------:|:----:|:---------------:|:--------:|:------:|
| No log        | 0.9662 | 100  | 0.6581          | 0.6802   | 0.3560 |
| No log        | 1.9275 | 200  | 0.5982          | 0.7105   | 0.4276 |
| No log        | 2.8889 | 300  | 0.5637          | 0.7349   | 0.4466 |
| No log        | 3.8502 | 400  | 0.5506          | 0.7349   | 0.4356 |
| 2.3869        | 4.8116 | 500  | 0.5452          | 0.75     | 0.4769 |
| 2.3869        | 5.7729 | 600  | 0.5400          | 0.7535   | 0.4619 |
| 2.3869        | 6.7343 | 700  | 0.5322          | 0.7593   | 0.5036 |
| 2.3869        | 7.6957 | 800  | 0.5339          | 0.7570   | 0.4709 |
| 2.3869        | 8.6570 | 900  | 0.5316          | 0.7616   | 0.4888 |
| 1.9359        | 9.6184 | 1000 | 0.5348          | 0.7547   | 0.4462 |


### Framework versions

- PEFT 0.14.0
- Transformers 4.47.1
- Pytorch 2.5.1+cu124
- Datasets 3.2.0
- Tokenizers 0.21.0