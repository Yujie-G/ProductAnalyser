from datetime import datetime

# 获取当前时间
current_time = datetime.now()

# 格式化为精确到分钟的字符串
formatted_time = current_time.strftime("%Y-%m-%d %H:%M")

model_path = "/root/autodl-tmp/chinese_wwm_pytorch"
model_save_dir = f"/root/autodl-tmp/myChinese_wwm_ckps/{formatted_time}"
dataset_path = "/root/autodl-tmp/online_shopping_10_cats.csv"

EPOCHS = 10
LEARNING_RATE = 1e-5

TRAIN_BATCHSIZE = 12
TEST_BATCHSIZE = 8

MAX_LENGTH = 128

TEST_SIZE = 0.15
