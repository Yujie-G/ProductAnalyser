import pandas as pd
import numpy as np
import os
import tqdm
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from dataset import VivoTestDataset

dir_path = '/root/autodl-tmp/ProductsComment'
file_name = 'mid.csv'
# file_name = sys.argv[1]
model_path = '/root/autodl-tmp/myChinese_wwm_ckps/2023-12-20 00:10/9'
device = torch.device("cuda:0")

df = pd.read_csv(os.path.join(dir_path, file_name))

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
model = model.to(device)
# model.load_state_dict(torch.load(model_path))

def preprocess(text):
    # 对文本进行预处理
    text = str(text)
    return tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

# 应用预处理
df['encoded'] = df['描述'].apply(lambda x: preprocess(str(x)))

def merge_dictionaries(dic):
    merged_dict = {}
    for sub_dict in dic.values():
        for key, value in sub_dict.items():
            if key in merged_dict:
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]
    return merged_dict

# 将 DataFrame 转换为 PyTorch Dataset
encoded_texts = merge_dictionaries(df['encoded'].to_dict())
dataset = VivoTestDataset(encoded_texts)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# 使用 BERT 进行情感预测
def predict_sentiment(data_loader):
    model.eval()
    sentiments = []
    scores = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="Processing data"):
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            score, predictions = torch.max(logits, dim=1)
            sentiments.extend(predictions.tolist())
            scores.extend(score.tolist())
    
    scores_array = np.array(scores)

    min_value = np.min(scores_array, axis=0)
    max_value = np.max(scores_array, axis=0)

    # 归一化scores到[0, 1]范围
    normalized_scores = (scores_array - min_value) / (max_value - min_value)
    return sentiments, normalized_scores.tolist()

# 获取预测结果
df['情感'], df['强度'] = predict_sentiment(dataloader)

df = df[['机型','类别','描述','情感','强度']]
df.to_csv(os.path.join(dir_path, 'result_'+file_name), index=False)


