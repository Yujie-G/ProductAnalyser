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
cfg.save_model = 1 if (len(sys.argv) > 4 and sys.argv[4] == "--save_model") else None

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

train_input = merge_dictionaries(train_df['encoded'].to_dict())
train_labels = train_df['label'].tolist()

test_input = merge_dictionaries(test_df['encoded'].to_dict())
test_labels = test_df['label'].tolist()

# print(merge_dictionaries(train_df['encoded'].to_dict()))
# exit(0)

# 创建数据集

train_dataset = ShoppingReviewDataset(train_input, train_labels)
test_dataset = ShoppingReviewDataset(test_input, test_labels)


# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=cfg.train_batchsize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=cfg.test_batchsize, shuffle=False)



# 加载预训练的BERT模型
model = BertForSequenceClassification.from_pretrained(cfg.model_path, num_labels=2)
model = model.to(device)

# 优化器
optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)

# 初始化用于保存loss值的列表
epoch_loss_values = []
step_loss_values = []

for epoch in range(cfg.epochs):  # 迭代次数
    model.train()
    total_loss = 0
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
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                step_loss_values.append(loss.item())
            pbar.set_description(f'Epoch {epoch + 1}/{cfg.epochs} - Loss: {loss.item():.4f}')
            pbar.update(1)
            step += 1

    
    # 计算平均loss
    avg_loss = total_loss / len(train_loader)
    epoch_loss_values.append(avg_loss)
    myLog(cfg, f"[Epoch {epoch + 1}] Avg Loss: {avg_loss:.4f}")
    if cfg.save_model is not None:
        model_save_path = os.path.join(cfg.save_dir, str(epoch))
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        # 保存微调后的模型到本地
        model.save_pretrained(model_save_path)

        # 保存tokenizer到本地
        tokenizer.save_pretrained(model_save_path)

plt.figure(1)
plt.plot(epoch_loss_values, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss per epoch Over Time')
plt.legend()
plt.savefig(os.path.join(cfg.save_dir, 'train_log','log_loss_epochs.png'))
plt.clf()
plt.plot(step_loss_values, label='Training Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Loss per step Over Time')
plt.legend()
plt.savefig(os.path.join(cfg.save_dir, 'train_log','log_loss_steps.png'))

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