import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, BitsAndBytesConfig, default_data_collator
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

import evaluate
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    return {"accuracy": acc, "f1": f1}

dataset = load_dataset("glue", "sst2")


model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=64)


tokenized_dataset = dataset.map(tokenize_function, batched=True)


id2label = {0: "NEGATIVE", 1: "POSITIVE"}

label2id = {"NEGATIVE": 0, "POSITIVE": 1}

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id=label2id, device_map="auto")
model.config.pad_token_id = model.config.eos_token_id[0]

peft_config = LoraConfig(
    r=16,  # Low-rank dimension
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Target attention layers for LoRA
    lora_dropout=0.01,
    task_type="SEQ_CLS",  # Task type
)

model = get_peft_model(model, peft_config)





training_args = TrainingArguments(
    output_dir="llama-1b-sst-2",
    learning_rate=2e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    warmup_steps=100,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    push_to_hub=True,
    report_to="none",
    push_to_hub_organization="BayanDuygu",
    bf16=True,
)

accuracy = evaluate.load("accuracy")


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("llama-1b-sst-2")


test_data = tokenized_dataset["validation"]
test_logits = trainer.predict(test_data).predictions
test_predictions = test_logits.argmax(axis=-1)

test_data = test_data.add_column("predicted_label", test_predictions)

# Convert the test dataset back to a Pandas DataFrame
test_df = pd.DataFrame({
    "text": test_data["sentence"],
    "true_label": test_data["label"],
    "predicted_label": test_data["predicted_label"]
})

test_df.to_csv("test_predictions_llama_sst2.csv", index=False)
print("Test predictions saved to 'test_predictions.csv'")

