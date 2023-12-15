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

model_path = "/root/autodl-tmp/chinese_wwm_pytorch"
model_save_dir = "/root/autodl-tmp/myChinese_wwm_ckps"
dataset_path = "/root/autodl-tmp/online_shopping_10_cats.csv"

device = torch.device("cuda:0")

df = pd.read_csv(dataset_path)
df = df[df['cat'] == '手机']

# 查看数据集结构
print(df.head())

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
tokenizer = BertTokenizer.from_pretrained(model_path)# 'bert-base-chinese')

def preprocess(text):
    text = str(text)
    return tokenizer(text, padding='max_length', truncation=True, max_length=512)

# 应用预处理到每个评论
df['encoded'] = df['review'].apply(preprocess)


train_df, test_df = train_test_split(df, test_size=0.2)

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
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



# 加载预训练的BERT模型
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
model = model.to(device)

# 优化器
optimizer = AdamW(model.parameters(), lr=1e-5)

# 初始化用于保存loss值的列表
epoch_loss_values = []

for epoch in range(3):  # 迭代次数
    model.train()
    total_loss = 0
    avg_loss = 1e10
    # 使用tqdm显示进度条
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}， loss: {avg_loss}"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    # 计算平均loss
    avg_loss = total_loss / len(train_loader)
    epoch_loss_values.append(avg_loss)
    with open('log.txt', 'a') as file:
        file.write(f"Epoch {epoch + 1} finished, Avg Loss: {avg_loss:.4f}")
    print(f"Epoch {epoch + 1} finished, Avg Loss: {avg_loss:.4f}")
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    torch.save(model.state_dict(), os.path.join(model_save_dir, f"epoch_{epoch+1}.pth"))

plt.plot(epoch_loss_values, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.savefig('loss.png')

