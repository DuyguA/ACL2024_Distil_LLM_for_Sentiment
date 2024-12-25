import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score
import numpy as np

import os, copy, json
from tqdm import tqdm

from data_loaders import  distill_loader
from dataset_makers import  make_dataset
from models import DistilledRoberta
from loss_functions import get_loss_class

class Trainer:
  def __init__(self, args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.mkdirs(args.output_dir)

    dataset = make_dataset(args.dataset)

    train_loader = distill_loader(dataset["train"], args.embeds_file,  mode='train', text_key=args.text_key, batch_size=args.batch_size, shuffle=True)
    test_loader = distill_loader(dataset["test"], None,  mode='test', text_key=args.text_key, batch_size=args.val_batch_size, shuffle=False)
    valid_loader = distill_loader(dataset["validation"], None,  mode='test', text_key=args.text_key, batch_size=args.val_batch_size, shuffle=False)
    print('Dataset prep done!\n')

    if torch.cuda.device_count() > 0:
      print(f"{torch.cuda.device_count()} GPUs found")

    print('Initializing model....')
    model = DistilledRoberta(args.num_labels)

    model = nn.DataParallel(model)
    model.to(device)
    params = model.parameters()
    self.scaler = GradScaler()

    optimizer = AdamW(params, lr=args.lr, weight_decay=0.01)
    self.device = device
    self.model = model
    self.optimizer = optimizer

    self.loss_function = get_loss_class(args.loss_type, args.alpha, args.beta, args.temperature)

    self.mode = args.loss_type

    self.args = args
    self.epoch_evals = []
    self.all_losses = []

  def train(self):
    best_epoch = 0
    print("First epoch will start soon")
    for epoch in range(self.args.epochs):
      print(f"{'*' * 20}Epoch: {epoch + 1}{'*' * 20}")
      loss = self.train_epoch()
      acc_val, f1_score  = self.eval()
      print(f'Binary Acc: {binary_acc:.3f}')
      self.epoch_accuracies.append((round(binary_acc, 3), round(f1_score, 3))) # append epoch accuracies
      print("Training finished here are epoch accuracy and f1s")
      with open("epoch_acc.txt", "w") as ofile:
        for eacc, ef1 in self.epoch_accuracies:
          ofile.write(str(eacc) + " " + str(ef1) + "\n")
      print("Now all losses")
      with open("losses.txt", "w") as ofile:
        for eloss in self.all_losses:
          ofile.write(str(eloss) + "\n")
    self.final_test()

  def train_epoch(self):
    self.model.train()
    epoch_loss = 0
    for i, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
        self.optimizer.zero_grad()
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        teacher_embeds = batch["final_embeds"].to(self.device) if "CLS" in self.mode else None

        labels = batch["labels"].to(self.device)  # target
        teacher_logits = batch["logits"].to(self.device) # target

        with autocast(device_type="cuda", dtype=torch.bfloat16):
          student_logits, student_hidden, teacher_hidden = self.model(input_ids, attention_mask, teacher_embeds)
          loss = self.loss_class(labels, student_logits, teacher_logits, student_hidden, teacher_hidden)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        interval = max(len(self.train_loader) // 20, 1)
        if i % interval == 0 or i == len(self.train_loader) - 1:
            lloss = round(loss.item(), 3)
            #print(f'Batch: {i + 1}/{len(self.train_loader)}\ttotal loss: {loss.item():.3f}\temotion loss:{emotion_loss.item():.3f}\tpair loss:{pair_loss.item():.3f}')
            print(f'Batch: {i + 1}/{len(self.train_loader)}\ttotal loss: {lloss:.3f}')
            self.all_losses.append(lloss) # append epoch losses
        epoch_loss += loss.item()
    return epoch_loss / len(self.train_loader)

  def predict_labels(self, logits):
    probs = F.softmax(logits, dim=1)
    predictions = torch.argmax(probs, dim=1)
    return predictions

  def eval(self):
    self.model.eval()
    label_pred = []
    label_true = []

    loader = self.valid_loader 

    with torch.no_grad():
      for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch['labels']
        student_logits, _, _  = self.model(input_ids, attention_mask, None)

        student_logits = student_logits.detach().cpu()
        predictions = self.predict_labels(student_logits)

        label_pred += predictions.tolist()
        label_true += labels.tolist()

                
    print(label_true, "true labels")
    print(label_pred, "pred labels")
    label_acc = accuracy_score(label_true, label_pred)
    if args.num_labels == 2:
      label_f1 = f1_score(label_true, label_pred, average="binary")
    else:
      label_f1 = f1_score(label_true, label_pred, average="macro")
    torch.save(self.model.state_dict(), args.output_dir+ "/model.pth")
    return label_acc, label_f1

  def final_test(self):
    loader = self.test_loader
    self.model.eval()
    all_preds = []
    all_labels = []
    all_texts = []

    with torch.no_grad():
      for batch in loader:
        texts = batch["texts"]
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"]
        student_logits, _, _ = self.model(input_ids, attention_mask, None)
        preds = predict_labels(student_logits)
        all_preds += preds.detach().cpu().tolist()
        all_labels += labels.detach().tolist()
        all_texts += texts

    output_csv = args.output_dir + "/results.csv"
    with open(output_csv, mode="w", newline="") as csv_file:
      writer = csv.write(csv_file)
      writer.writerow(["Text", "True Label", "Predicted Label"])
      for text, tlabel, plabel in zip(all_texts, all_labels, all_preds):
        writer.writerow([text, tlabel, plabel])
      


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=48, help='batch size of training')
    parser.add_argument('--val_batch_size', type=int, default=4, help='batch size of testing')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--text_key', type=str, default="text", help='Text key of the dataset')
    parser.add_argument('--dataset', type=str, default="sst-2", help='Name of the dataset')
    parser.add_argument('--num_labels', type=int, default="2", help='Nunber classes for classification')
    parser.add_argument('--embeds_file', type=str, default="tensors", help='File containing LLama embeddings')
    parser.add_argument('--temper', type=float, default="2.0", help='Temperature for smoothening logits')
    parser.add_argument('--loss_type', type=str, default="logits", help='Loss type for distillation. Options are logits or CLS')
    parser.add_argument('--alpha', type=float, default="0.5", help='Distillation coeff of logits, must be between 0 and 1. Must be nonzero if logits or logits+CLS chosen')
    parser.add_argument('--beta', type=float, default="0", help='Distillation coeff of CLS, must be between 0 and 1. Must be nonzero if CLS is chosen.')
    parser.add_argument('--output_dir', type=str, default="exp", help='Directory for saving the model')
    args = parser.parse_args()

    print(args)
    engine = Trainer(args)
    engine.train()
