---
library_name: peft
license: llama3.2
base_model: meta-llama/Llama-3.2-3B-Instruct
tags:
- generated_from_trainer
metrics:
- accuracy
- precision
- recall
- f1
model-index:
- name: llama-3b-sst-5
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# llama-3b-sst-5

This model is a fine-tuned version of [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.2205
- Accuracy: 0.4623
- Precision: 0.4490
- Recall: 0.4360
- F1: 0.4354

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

| Training Loss | Epoch  | Step | Validation Loss | Accuracy | Precision | Recall | F1     |
|:-------------:|:------:|:----:|:---------------:|:--------:|:---------:|:------:|:------:|
| No log        | 1.4944 | 100  | 1.4847          | 0.3597   | 0.3413    | 0.3388 | 0.3306 |
| No log        | 2.9888 | 200  | 1.3224          | 0.4133   | 0.4129    | 0.3843 | 0.3901 |
| No log        | 4.4794 | 300  | 1.2652          | 0.4405   | 0.4313    | 0.4090 | 0.4121 |
| No log        | 5.9738 | 400  | 1.2515          | 0.4550   | 0.4465    | 0.4335 | 0.4248 |
| 5.2724        | 7.4644 | 500  | 1.2249          | 0.4659   | 0.4514    | 0.4367 | 0.4377 |
| 5.2724        | 8.9588 | 600  | 1.2205          | 0.4623   | 0.4490    | 0.4360 | 0.4354 |


### Framework versions

- PEFT 0.14.0
- Transformers 4.47.1
- Pytorch 2.5.1+cu124
- Datasets 3.2.0
- Tokenizers 0.21.0