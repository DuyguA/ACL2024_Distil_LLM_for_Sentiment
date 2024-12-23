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
- name: llama-3b-irony
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# llama-3b-irony

This model is a fine-tuned version of [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.5817
- Accuracy: 0.7105
- F1: 0.6146

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

| Training Loss | Epoch | Step | Validation Loss | Accuracy | F1     |
|:-------------:|:-----:|:----:|:---------------:|:--------:|:------:|
| No log        | 1.0   | 30   | 1.0633          | 0.5013   | 0.5155 |
| No log        | 2.0   | 60   | 0.7927          | 0.5982   | 0.5191 |
| No log        | 3.0   | 90   | 0.6772          | 0.6531   | 0.5763 |
| No log        | 4.0   | 120  | 0.6298          | 0.6786   | 0.5896 |
| No log        | 5.0   | 150  | 0.6055          | 0.6964   | 0.6222 |
| No log        | 6.0   | 180  | 0.5919          | 0.7041   | 0.5842 |
| No log        | 7.0   | 210  | 0.5895          | 0.7156   | 0.6455 |
| No log        | 8.0   | 240  | 0.5849          | 0.7066   | 0.6102 |
| No log        | 9.0   | 270  | 0.5831          | 0.7168   | 0.6172 |
| No log        | 10.0  | 300  | 0.5817          | 0.7105   | 0.6146 |


### Framework versions

- PEFT 0.14.0
- Transformers 4.47.1
- Pytorch 2.5.1+cu124
- Datasets 3.2.0
- Tokenizers 0.21.0