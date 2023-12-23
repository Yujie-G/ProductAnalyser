import os
import sys

sys.path.append('/root/ProductAnalyser/SentimentAnalysis')

for epoch in [3, 7, 10]:
    for lr in [1e-3, 1e-4, 3e-5, 1e-5]:
        for batchsize in[4, 8, 12]:
            os.system(f"python /root/ProductAnalyser/SentimentAnalysis/train.py {epoch} {lr} {batchsize}")

os.system("/usr/bin/shutdown")