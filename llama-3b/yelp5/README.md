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
- name: llama-3b-yelp-5
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# llama-3b-yelp-5

This model is a fine-tuned version of [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.1560
- Accuracy: 0.5031
- Precision: 0.4995
- Recall: 0.5007
- F1: 0.4992

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

| Training Loss | Epoch  | Step | Validation Loss | Accuracy | Precision | Recall | F1     |
|:-------------:|:------:|:----:|:---------------:|:--------:|:---------:|:------:|:------:|
| No log        | 0.2559 | 100  | 1.5388          | 0.3881   | 0.3781    | 0.3857 | 0.3795 |
| No log        | 0.5118 | 200  | 1.3305          | 0.4521   | 0.4572    | 0.4485 | 0.4488 |
| No log        | 0.7678 | 300  | 1.2792          | 0.4671   | 0.4605    | 0.4669 | 0.4569 |
| No log        | 1.0230 | 400  | 1.2323          | 0.4813   | 0.4753    | 0.4787 | 0.4749 |
| 5.6322        | 1.2790 | 500  | 1.2116          | 0.4912   | 0.4880    | 0.4897 | 0.4872 |
| 5.6322        | 1.5349 | 600  | 1.1980          | 0.4917   | 0.4883    | 0.4895 | 0.4876 |
| 5.6322        | 1.7908 | 700  | 1.1828          | 0.4979   | 0.4965    | 0.4953 | 0.4940 |
| 5.6322        | 2.0461 | 800  | 1.1738          | 0.498    | 0.4924    | 0.4963 | 0.4930 |
| 5.6322        | 2.3020 | 900  | 1.1682          | 0.4994   | 0.4991    | 0.4987 | 0.4980 |
| 4.4778        | 2.5579 | 1000 | 1.1581          | 0.5033   | 0.4979    | 0.5017 | 0.4993 |
| 4.4778        | 2.8138 | 1100 | 1.1560          | 0.5031   | 0.4995    | 0.5007 | 0.4992 |


### Framework versions

- PEFT 0.14.0
- Transformers 4.47.1
- Pytorch 2.5.1+cu124
- Datasets 3.2.0
- Tokenizers 0.21.0