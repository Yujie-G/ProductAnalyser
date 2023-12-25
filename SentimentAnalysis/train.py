import pandas as pd
import sys
import os
sys.path.append('/root/ProductAnalyser/SentimentAnalysis')
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

from dataset import ShoppingReviewDataset

from config import config
from utils import *

cfg = config()
cfg.epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
cfg.learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-5
cfg.train_batchsize = int(sys.argv[3]) if len(sys.argv) > 3 else 16
cfg.optimizer = sys.argv[4] if len(sys.argv) > 4 else "AdamW"
cfg.save_model = 1 if (len(sys.argv) > 5 and sys.argv[5] == "--save_model") else None

myLog(cfg, "------start train------")
device = torch.device("cuda:0")

df = pd.read_csv(cfg.dataset_path)
df = df[df['cat'].isin(['手机'])] # ,'平板','计算机'

work_dir = os.path.dirname(os.path.abspath(__file__))
def merge_dictionaries(dic):
    merged_dict = {}
    for sub_dict in dic.values():
        for key, value in sub_dict.items():
            if key in merged_dict:
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]
    return merged_dict

# 初始化BERT分词器
tokenizer = BertTokenizer.from_pretrained(cfg.model_path)# 'bert-base-chinese')

def preprocess(text):
    text = str(text)
    return tokenizer(text, padding='max_length', truncation=True, max_length=cfg.max_length)

# 应用预处理到每个评论
df['encoded'] = df['review'].apply(preprocess)


train_df, test_df = train_test_split(df, test_size=cfg.test_size)
val_size_relative = cfg.val_size / cfg.test_size
val_df, test_df = train_test_split(test_df, test_size=val_size_relative)

train_input = merge_dictionaries(train_df['encoded'].to_dict())
train_labels = train_df['label'].tolist()

test_input = merge_dictionaries(test_df['encoded'].to_dict())
test_labels = test_df['label'].tolist()

val_input = merge_dictionaries(val_df['encoded'].to_dict())
val_labels = val_df['label'].tolist()

# 创建数据集

train_dataset = ShoppingReviewDataset(train_input, train_labels)
val_dataset = ShoppingReviewDataset(val_input, val_labels)
test_dataset = ShoppingReviewDataset(test_input, test_labels)


# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=cfg.train_batchsize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=cfg.test_batchsize, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=cfg.test_batchsize, shuffle=False)



# 加载预训练的BERT模型
model = BertForSequenceClassification.from_pretrained(cfg.model_path, num_labels=2)
model = model.to(device)

# 优化器
optimizer = None
if cfg.optimizer == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)
elif cfg.optimizer == "AdamW":
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)

# 初始化用于保存loss值的列表
epoch_train_loss_values = []
epoch_val_loss_values = []
eval_val_accuracys = []
step_loss_values = []

for epoch in range(cfg.epochs):  # 迭代次数
    model.train()
    total_train_loss = 0
    # 使用tqdm显示进度条
    step = 0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}', unit='batch') as pbar:
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                step_loss_values.append(loss.item())
            pbar.set_description(f'Epoch {epoch + 1}/{cfg.epochs} - Loss: {loss.item():.4f}')
            pbar.update(1)
            step += 1

    
    # 计算平均loss
    avg_loss = total_train_loss / len(train_loader)
    epoch_train_loss_values.append(avg_loss)


    # validation
    model.eval()
    acc = 0
    total_eval_loss = 0
    with tqdm(total=len(val_loader), desc=f'Epoch {epoch + 1}', unit='batch') as pbar:
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=1)
            acc += torch.sum(predicted_labels == labels)
            total_eval_loss += loss.item()
            pbar.set_description(f'Val Loss: {loss.item():.4f}')
            pbar.update(1)
    avg_val_loss = total_eval_loss / len(val_loader)
    epoch_val_loss_values.append(avg_val_loss)
    acc = acc / len(val_loader.dataset)
    eval_val_accuracys.append(acc.item())

    myLog(cfg, f"[Epoch {epoch + 1}] Avg Train Loss: {avg_loss:.4f} Avg Val Loss: {avg_val_loss:.4f} Acc: {acc.item():.4f}\n")
    if cfg.save_model is not None:
        model_save_path = os.path.join(cfg.save_dir, str(epoch))
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        # 保存微调后的模型到本地
        model.save_pretrained(model_save_path)

        # 保存tokenizer到本地
        tokenizer.save_pretrained(model_save_path)

plt.figure(1)
plt.plot(epoch_train_loss_values, label='Training Loss')
plt.plot(epoch_val_loss_values, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss-Validation Loss per epoch Over Time')
plt.legend()
plt.savefig(os.path.join(cfg.save_dir, 'train_log','log_loss_epochs.png'))
plt.clf()
plt.plot(step_loss_values, label='Training Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Loss per step Over Time')
plt.legend()
plt.savefig(os.path.join(cfg.save_dir, 'train_log','log_loss_steps.png'))
plt.clf()
# set orange color
plt.plot(eval_val_accuracys, label='Validation Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy per epoch Over Time')
plt.legend()
plt.savefig(os.path.join(cfg.save_dir, 'train_log','log_acc_epochs.png'))
plt.clf()

model.eval()
total_eval_accuracy = 0
total_eval_loss = 0

for batch in test_loader:
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # 计算准确率
        predictions = torch.argmax(logits, dim=-1)
        correct_predictions = torch.eq(predictions, labels).sum().item()
        total_eval_accuracy += correct_predictions

# 计算整体准确率
accuracy = total_eval_accuracy / len(test_dataset)
myLog(cfg, f"Accuracy: {accuracy}")