from torch.utils.data import Dataset, DataLoader
import torch, json
from datasets import load_dataset
from transformers import AutoTokenizer


class DistillDataset(Dataset):
    def __init__(self, text_dataset, tensor_file, text_key="text", mode="train"):
        self.dataset = text_dataset
        self.text_key = text_key
        self.mode = mode
        if mode == "train":
          self.tensors = []
          with open(tensor_file, "r") as tfile:
              for line in tfile:
                linejs = json.loads(line)
                logits = linejs["logit"]
                final_token_embed = linejs["final_embed"]
                logits = torch.tensor(logits, dtype=torch.float32)
                final_token_embed = torch.tensor(final_token_embed, dtype=torch.float32)
                self.tensors.append({"logits": logits, "final_embed": final_token_embed})

    def __getitem__(self, index):
        data_point = self.dataset[index]
        text = data_point[self.text_key]
        label = data_point["label"]
        if self.mode == "test":
          logits, final_embed_tokens = None, None
        else:
          tensor_dict = self.tensors[index]
          logits, final_embed_tokens = tensor_dict["logits"], tensor_dict["final_embed"] 

        item = {
            'text': text,
            'label': label,
            "logits": logits,
            "final_embed_token": final_embed_tokens
         }
        return item

    def __len__(self):
        return len(self.dataset)

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"




def variable_batcher(batch):
  all_texts = [item["text"] for item in batch]
  encodings = tokenizer(all_texts, padding="max_length", truncation=True, return_tensors="pt", max_length=128)
  input_ids = encodings['input_ids']
  attention_mask = encodings['attention_mask']

  labels = [item["label"] for item in batch]

  

  logits = [item["logits"] for item in batch]
  final_embed = [item["final_embed_token"] for item in batch]
  if logits[0] is None:
    all_logits, all_finals = None, None
  else:
    all_logits = torch.stack(logits)
    all_finals = torch.stack(final_embed)

  item = {
      'input_ids': torch.tensor(input_ids),
      'attention_mask': torch.tensor(attention_mask),
      'labels': torch.tensor(labels,  dtype=torch.long),
      'logits': all_logits,
      'final_embeds': all_finals,
      'text': all_texts,
    }
  return item

def distill_loader(text_dataset, tensor_file, mode, text_key, batch_size, shuffle=False):
  dataset = DistillDataset(text_dataset, tensor_file, mode=mode, text_key=text_key)
  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=variable_batcher)


'''
dset = load_dataset("SetFit/sst5")
#test_loader = distill_loader(dset["test"], None, mode="test", text_key="text", batch_size=4) 
fname =  "../llama-1b/sst-5/embeddings-llama1B-sst5.jsonl"
train_loader = distill_loader(dset["train"], fname, mode="train", text_key="text", batch_size=4) 

for batch in train_loader:
    print(batch)
    break

'''
