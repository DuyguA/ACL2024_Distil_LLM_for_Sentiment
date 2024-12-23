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
- name: llama-3b-sst-2
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# llama-3b-sst-2

This model is a fine-tuned version of [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2716
- Accuracy: 0.8865
- F1: 0.8899

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
- num_epochs: 3

### Training results

| Training Loss | Epoch  | Step | Validation Loss | Accuracy | F1     |
|:-------------:|:------:|:----:|:---------------:|:--------:|:------:|
| No log        | 0.1900 | 100  | 0.4869          | 0.7718   | 0.7863 |
| No log        | 0.3800 | 200  | 0.3686          | 0.8498   | 0.8479 |
| No log        | 0.5701 | 300  | 0.3440          | 0.8624   | 0.8684 |
| No log        | 0.7601 | 400  | 0.3106          | 0.8761   | 0.8797 |
| 1.6451        | 0.9501 | 500  | 0.3123          | 0.8739   | 0.8817 |
| 1.6451        | 1.1387 | 600  | 0.2887          | 0.8842   | 0.8889 |
| 1.6451        | 1.3287 | 700  | 0.2839          | 0.8911   | 0.8912 |
| 1.6451        | 1.5188 | 800  | 0.2787          | 0.8911   | 0.8931 |
| 1.6451        | 1.7088 | 900  | 0.2973          | 0.875    | 0.8829 |
| 1.0595        | 1.8988 | 1000 | 0.2712          | 0.8865   | 0.8884 |
| 1.0595        | 2.0874 | 1100 | 0.2727          | 0.8968   | 0.8968 |
| 1.0595        | 2.2774 | 1200 | 0.2701          | 0.8899   | 0.8919 |
| 1.0595        | 2.4675 | 1300 | 0.2692          | 0.8968   | 0.8977 |
| 1.0595        | 2.6575 | 1400 | 0.2682          | 0.8922   | 0.8944 |
| 0.9838        | 2.8475 | 1500 | 0.2716          | 0.8865   | 0.8899 |


### Framework versions

- PEFT 0.14.0
- Transformers 4.47.1
- Pytorch 2.5.1+cu124
- Datasets 3.2.0
- Tokenizers 0.21.0