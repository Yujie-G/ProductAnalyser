import datetime
import os
def get_timeString():
    # 获取当前时间
    current_time = datetime.datetime.now()
    # 格式化当前时间，使用 "-" 分隔年、月和日
    return current_time.strftime("%Y-%m-%d")
def myLog(dir, string):
    print(string)
    Time = get_timeString()
    log_file_dir = os.path.join(dir,'train_log')
    if not os.path.exists(log_file_dir):
        os.mkdir(log_file_dir)
    with open(os.path.join(log_file_dir, 'log.txt'), 'a') as file:
        file.write('['+Time+'] '+string)
    
