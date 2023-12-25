from datetime import datetime
import os

def get_timeString():
    # 获取当前时间
    current_time = datetime.now()
    # 格式化当前时间，使用 "-" 分隔年、月和日
    return current_time.strftime("%Y-%m-%d-%H:%M:%S")

start = True
def myLog(config, string):
    global start
    log_file_dir = os.path.join(config.save_dir,'train_log')
    log_file_name = os.path.join(log_file_dir, 'log.txt')
    if start:
        print('--------Logger start----------')
        print(f'Epochs:{config.epochs}\nLr: {config.learning_rate}\nTrain batchsize: {config.train_batchsize}\nMAX_LENGTH: {config.max_length}\n')
        print('------------------------------')
        if not os.path.exists(log_file_dir):
            os.system(f"mkdir -p {log_file_dir}")
        with open(log_file_name, 'a') as file:
            file.write(f'Epochs:{config.epochs}\nLr: {config.learning_rate}\nTrain batchsize: {config.train_batchsize}\nMAX_LENGTH: {config.max_length}\n')
        start = False
    print(string)
    Time = get_timeString()
    with open(log_file_name, 'a') as file:
        file.write('['+Time+'] '+string+ '\n')
    
