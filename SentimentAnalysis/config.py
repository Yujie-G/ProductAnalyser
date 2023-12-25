from datetime import datetime

class config:
    def __init__(self):
        current_time = datetime.now()

        # 格式化为精确到分钟的字符串
        start_time = current_time.strftime("%Y-%m-%d-%H:%M:%S")
        print(start_time)
        self.model_path = "/root/autodl-tmp/chinese_wwm_pytorch"
        self.save_dir = f"/root/autodl-tmp/myChinese_wwm_ckps/{start_time}"
        self.dataset_path = "/root/autodl-tmp/online_shopping_10_cats.csv"
        self.save_model = None

        self.epochs = None
        self.learning_rate = None

        self.train_batchsize = None
        self.test_batchsize = 8

        self.max_length = 128
        self.optimizer = "AdamW"

        self.val_size = 0.1
        self.test_size = 0.2
