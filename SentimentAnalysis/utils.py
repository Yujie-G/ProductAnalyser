import datetime
import os
from config import *
def get_timeString():
    # 获取当前时间
    current_time = datetime.now()
    # 格式化当前时间，使用 "-" 分隔年、月和日
    return current_time.strftime("%Y-%m-%d-%H:%M")

start = True
def myLog(dir, string):
    global start
    log_file_dir = os.path.join(dir,'train_log')
    log_file_name = os.path.join(log_file_dir, 'log.txt')
    if start:
        print('--------Logger start----------')
        print(f'Epochs:{EPOCHS}\nLr: {LEARNING_RATE}\nTrain batchsize: {TRAIN_BATCHSIZE}\nMAX_LENGTH: {MAX_LENGTH}')
        print('------------------------------')
        if not os.path.exists(log_file_dir):
            os.system(f"mkdir -p {log_file_dir}")
        with open(log_file_name, 'a') as file:
            file.write(f'Epochs:{EPOCHS}\nLr: {LEARNING_RATE}\nTrain batchsize: {TRAIN_BATCHSIZE}\nMAX_LENGTH: {MAX_LENGTH}')
        start = False
    print(string)
    Time = get_timeString()
    with open(log_file_name, 'a') as file:
        file.write('['+Time+'] '+string+ '\n')
    
