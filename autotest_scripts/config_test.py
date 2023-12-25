import os
import sys

sys.path.append('/root/ProductAnalyser/SentimentAnalysis')

# test run
# os.system(f"python /root/ProductAnalyser/SentimentAnalysis/train.py 2 1e-3 8")
# exit(0)

for epoch in [3, 7]:
    for lr in [1e-3, 1e-4, 1e-5]:
        for batchsize in[4, 8, 12]:
            os.system(f"python /root/ProductAnalyser/SentimentAnalysis/train.py {epoch} {lr} {batchsize}")

# os.system("/usr/bin/shutdown")