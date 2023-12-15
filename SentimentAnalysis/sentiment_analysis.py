# from transformers import BertModel, BertTokenizer

# # 指定本地模型文件夹的路径
# model_path = '/root/autodl-tmp/chinese_wwm'

# # 初始化一个tokenizer和一个模型
# tokenizer = BertTokenizer.from_pretrained(model_path)
# model = BertModel.from_pretrained(model_path)

# # 输入文本
# text = "你好，这是一个BERT模型加载示例。"

# # 使用tokenizer将文本转换为模型可接受的输入格式
# inputs = tokenizer(text, return_tensors="pt")

# # 使用模型进行推断
# outputs = model(**inputs)

# # 获取模型的输出
# hidden_states = outputs.last_hidden_state

# # 打印输出
# print(hidden_states.shape)

model.eval()
total_eval_accuracy = 0
total_eval_loss = 0

for batch in test_loader:
    with torch.no_grad():
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # 计算准确率
        predictions = torch.argmax(logits, dim=-1)
        correct_predictions = torch.eq(predictions, labels).sum().item()
        total_eval_accuracy += correct_predictions

# 计算整体准确率
accuracy = total_eval_accuracy / len(test_dataset)
print(f"Accuracy: {accuracy}")
