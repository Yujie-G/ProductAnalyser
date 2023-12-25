import os
import sys

sys.path.append('/root/ProductAnalyser/SentimentAnalysis')

# test run
# os.system(f"python /root/ProductAnalyser/SentimentAnalysis/train.py 3 1e-3 64 SGD")
# exit(0)

for epoch in [3, 7]:
    for lr in [1e-3, 1e-4, 1e-5]:
        for batchsize in[8, 32]:
            for optimizer in ['AdamW', 'SGD']:
                os.system(f"python /root/ProductAnalyser/SentimentAnalysis/train.py {epoch} {lr} {batchsize} {optimizer}")

# os.system("/usr/bin/shutdown")