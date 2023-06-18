from dataloader_multi import DataLoader
import time
import os
import sys


dl_train_config = {
    "TinyStoriesV2": 100,
    "TinyStories-Instruct": 100,
    "alpaca_zh_51k": 50,
    "dolly-15k": 50,
    "lightnovels-en": 50,
    "openwebtext": 40,
    "chinese-dolly-15k": 50,
}
dl = DataLoader("prepare-data/data", "config/tokenizer.model", dl_train_config, worker_num=10)

for i in range(10000):
    #sys.stdout.write("calling get_batch\n")
    # record the time to get a batch
    t = time.time()
    tmp = dl.get_batch(256, 1024)
    t1 = time.time()
    sys.stdout.write(f"get_batch takes {t1 - t} seconds, current buf size: {dl.get_buf_size()}\n")
    print(f"iter: {i}")
    #print(tmp[0])
    time.sleep(0.1)