import torch
import torch.nn.functional as F
import lightning
from model_lr import GPT, GPTConfig
import os
import random
from tokenizer import Tokenizer
import os
import json
import trainer

# exmaple record object
# example = {
#     "recentIterNum": 0,
#     "recentLoss": -1,
#     "recentEvalLoss": -1,
#     "iters": {
#         "1000": {
#             "type": "ckpt" | "ckpt_eval" | "loss",
#             "loss": 0.1,
#             "evalLoss": 0.2,
#             "example_eval":[],
#             "runningTime": 1000, # in seconds
#             "iter_count": 1000
#         }
#     },
#     "dataset_config": {}
# }

class Trainer(trainer.Trainer):
    def create(self, config: GPTConfig, lr, device, dtype=torch.bfloat16, mini_batchSize=8, whole_batchSize=128):
        self.config = config
        self.mini_batchSize = mini_batchSize
        self.whole_batchSize = whole_batchSize
        self.device = device
        
        self.model = GPT(config).type(dtype)

        # step the scheduler on each whole batch, since there are too many batches per epoch
        # every 1000*2 steps, lr decays to 1/20, then rise to original lr
        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        #self.model = torch.compile(self.model) # need pytorch 2.0
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=80000, eta_min=lr/20, last_epoch=-1)
        self.loss = torch.nn.CrossEntropyLoss(reduction='none').type("float32")

        self.loss = self.loss.to(device)