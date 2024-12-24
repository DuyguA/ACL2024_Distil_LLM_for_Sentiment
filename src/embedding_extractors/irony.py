import torch
import json
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

import pandas as pd


model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(tokenizer.eos_token, tokenizer.eos_token_id)

def tokenize_function(example):
    sentences = [sentence + tokenizer.eos_token  for sentence in example["sentence"]]
    return tokenizer(sentences, truncation=True, max_length=64)


train_csv_path = "train.csv"  # Replace with the path to your train.csv file

train_df = pd.read_csv(train_csv_path, delimiter="\t")


# Rename columns for consistency (optional)
train_df.rename(columns={"Tweet text": "text", "Label": "label"}, inplace=True)

# Drop unnecessary columns (e.g., 'Tweet index') if needed
train_df = train_df[["text", "label"]]

# Convert pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)

# Create a DatasetDict with train and test splits
dataset = DatasetDict({
    "train": train_dataset,
})




#tokenized_dataset = dataset.map(tokenize_function, batched=True)


base_model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = PeftModel.from_pretrained(base_model, "BayanDuygu/llama-1b-irony")


model = model.to("cuda:0")

model.eval()

with torch.no_grad():
  with open("embeddings-llama1-B-irony.jsonl", "w") as ofile:
    for instance in dataset["train"]:
      inputs = tokenizer(instance["text"], return_tensors="pt")
      inputs = inputs.to("cuda:0")
      outputs = model(**inputs, return_dict=True, output_hidden_states=True)
      logits = outputs.logits.cpu()
      final_layer_output = outputs.hidden_states[-1].cpu()
      last_token_embedding = final_layer_output[:, -1, :]
      outjs = {"logit": logits.tolist(), "final_embed": last_token_embedding.tolist()}
      outjs = json.dumps(outjs)
      ofile.write(outjs+"\n")

