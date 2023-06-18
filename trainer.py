import torch
import torch.nn.functional as F
import lightning
from model import GPT, GPTConfig
import os
import random
from tokenizer import Tokenizer
import os
import json

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

class Trainer:
    def __init__(self, tokenizer: Tokenizer, name):
        self.config = {}
        self.mini_batchSize = 8
        self.whole_batchSize = 128
        self.name = name
        self.record_path = f"./record/{name}"
        if not os.path.exists(self.record_path):
            os.makedirs(self.record_path, exist_ok=True)
        self.ckpt_path = os.path.join(self.record_path, "ckpt")
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path, exist_ok=True)
        self.log_path = os.path.join(self.record_path, "log.json")
        self.log = dict()
        self.log["recentIterNum"] = 0
        self.log["recentLoss"] = -1
        self.log["recentEvalLoss"] = -1
        self.log["iters"] = dict()
        if os.path.exists(self.log_path):
            self.log = json.load(open(self.log_path, "r"))
        
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss = None
        self.device = None
        self.tokenizer = tokenizer


    def train_oneWholeBatch(self, x, y) -> float:
        if torch.isnan(x).any() or torch.isnan(y).any():
            print("nan in x or y")
            return 0
        self.model.train()
        self.optimizer.zero_grad()
        minibatch_count = self.whole_batchSize // self.mini_batchSize
        x_len = x.shape[1]
        x = x.long().to(self.device)
        y = y.long().to(self.device)

        ret_loss = 0
        for i in range(0, minibatch_count):
            logits = self.model(x[i:i+self.mini_batchSize, :])
            # prevent to have 0 in logits
            loss = self.loss(logits[:, -1, :].view(-1, self.config.vocab_size), y[i:i+self.mini_batchSize, :].view(-1))
            loss = torch.clamp(loss, min=-20.0, max=20.0).mean()  / minibatch_count
            loss.backward()
            ret_loss += loss.item()
            torch.nn.utils.clip_grad.clip_grad_value_(self.model.parameters(), 1000)
        torch.nn.utils.clip_grad.clip_grad_value_(self.model.parameters(), 1.5)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        return ret_loss
    
    @torch.no_grad()
    def inference(self, x):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            return logits
    
    @torch.no_grad()
    def evaluate(self, x, y):
        self.model.eval()
        i = 0
        total_loss = 0
        batch_count = 0
        while i + self.mini_batchSize < x.shape[0]:
            x_batch = x[i:i+self.mini_batchSize, :].long().to(self.device)
            y_batch = y[i:i+self.mini_batchSize, :].long().to(self.device)
            logits = self.model(x_batch)
            loss = self.loss(logits[:, -1, :].view(-1, self.config.vocab_size), y_batch[:, -1:].view(-1))
            loss = torch.clamp(loss, min=-20.0, max=20.0).mean()
            i += self.mini_batchSize
            total_loss += loss.item()
            batch_count += 1
        return total_loss / batch_count

        
    def save(self, path):
        a = {
            'model_state_dict': self.model.state_dict(),
            'opt': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'loss': self.loss.state_dict(),
            'config': self.config,
            'mini_batchSize': self.mini_batchSize,
            'whole_batchSize': self.whole_batchSize
        }
        torch.save(a, path)
        

    def load(self, path, device):
        assert os.path.exists(path)
        a = torch.load(path)
        self.create(a['config'], 0.0001, device, mini_batchSize=a['mini_batchSize'], whole_batchSize=a['whole_batchSize'])
        self.model.load_state_dict(a['model_state_dict'])
        self.optimizer.load_state_dict(a['opt'])
        self.scheduler.load_state_dict(a['scheduler'])
        self.loss.load_state_dict(a['loss'])
        

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

    
    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens, temperature=1.0, top_k=200):
        assert prompt.ndim == 2 and prompt.size(0) == 1, "prompt shape should be (1, seqlen)"
        self.model.eval()
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            prompt = prompt if prompt.size(1) <= self.config.max_token_len else prompt[:, -self.config.max_token_len:]
            # forward the model to get the logits for the index in the sequence
            logits = self.model(prompt)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            token_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            prompt = torch.cat((prompt, token_next), dim=1)

        return prompt
    
    def checkpoint(self, iter, loss, runningTime, iter_count):
        ckpt_name = f"ckpt-{iter}.model"
        ckpt_path = os.path.join(self.ckpt_path, ckpt_name)
        self.save(ckpt_path)
        # only save the last 5 checkpoints
        allckpt_list = os.listdir(self.ckpt_path)
        ckpt_list = []
        for ckpt in allckpt_list:
            if ckpt.startswith("ckpt-"):
                ckpt_list.append(ckpt)
        ckpt_list.sort(
            key=lambda x: int(x.split("-")[1].split(".")[0]),
        )
        if len(ckpt_list) > 7:
            os.remove(os.path.join(self.ckpt_path, ckpt_list[0]))
        self.log["recentIterNum"] = iter
        self.log["recentLoss"] = loss
        self.log["iters"][str(iter)] = {
            "type": "ckpt",
            "loss": loss,
            "runningTime": runningTime,
            "iter_count": iter_count
        }
        json.dump(self.log, open(self.log_path, "w"), indent=4)


    def evalCheckpoint(self, iter, loss, runningTime, iter_count, x, y):
        ckpt_name = f"evalckpt-{iter}.model"
        ckpt_path = os.path.join(self.ckpt_path, ckpt_name)
        self.save(ckpt_path)

        # only save the last 10 eval-checkpoints
        allckpt_list = os.listdir(self.ckpt_path)
        evalckpt_list = []
        for ckpt in allckpt_list:
            if ckpt.startswith("evalckpt-"):
                evalckpt_list.append(ckpt)
        evalckpt_list.sort(
            key=lambda x: int(x.split("-")[1].split(".")[0]),
        )
        if len(evalckpt_list) > 10:
            os.remove(os.path.join(self.ckpt_path, evalckpt_list[0]))

        r1, r2, r3, r4 = self.eval_examplePrompt()
        evalLoss = self.evaluate(x, y)
        self.log["recentEvalLoss"] = evalLoss
        self.log["recentIterNum"] = iter
        self.log["recentLoss"] = loss
        self.log["iters"][str(iter)] = {
            "type": "ckpt_eval",
            "loss": loss,
            "evalLoss": evalLoss,
            "example_eval": [r1, r2, r3, r4],
            "runningTime": runningTime,
            "iter_count": iter_count
        }
        json.dump(self.log, open(self.log_path, "w"), indent=4)

    def recordLoss(self, iter, loss, runningTime, iter_count):
        self.log["iters"][str(iter)] = {
            "type": "loss",
            "loss": loss,
            "runningTime": runningTime,
            "iter_count": iter_count
        }
        json.dump(self.log, open(self.log_path, "w"), indent=4)

    def get_recentIterNum(self):
        return self.log["recentIterNum"]

    def get_recentIter(self):
        return self.log["iters"][str(self.log["recentIterNum"])]
    
    def set_dataset_config(self, dataset_config):
        self.log["dataset_config"] = dataset_config
    
    def get_recentCkptPath(self):
        ckpt_list = os.listdir(self.ckpt_path)
        ckpt_list.sort(
            key=lambda x: int(x.split("-")[1].split(".")[0]),
        )
        if len(ckpt_list) == 0:
            return None
        return os.path.join(self.ckpt_path, ckpt_list[-1])

    def eval_examplePrompt(self):
        eval_p1 = "Once upon a time there was a pumpkin. It was a very special pumpkin, it could speak. It was sad because it couldn’t move. Every day, it would say"
        eval_pt1 = torch.tensor(self.tokenizer.encode(eval_p1, bos=True, eos=False)).view(1, -1).to(self.device)
        eval_p2 = "Lily and Ben were having an argument. Ben said that cake is much better than ice cream and Lily said that"
        eval_pt2 = torch.tensor(self.tokenizer.encode(eval_p2, bos=True, eos=False)).view(1, -1).to(self.device)
        eval_p3 = "instruction: 给一家餐厅写一封邮件，询问20人烧烤大概需要多少钱。\ninput\nresponse: "
        eval_pt3 = torch.tensor(self.tokenizer.encode(eval_p3, bos=True, eos=False)).view(1, -1).to(self.device)
        eval_p4 = "instruction: Is a flower an animal?。\ninput\nresponse: "
        eval_pt4 = torch.tensor(self.tokenizer.encode(eval_p4, bos=True, eos=False)).view(1, -1).to(self.device)

        eval_rt1 = self.generate(eval_pt1, 32, temperature=0.75, top_k=100)
        eval_rt2 = self.generate(eval_pt2, 32, temperature=0.75, top_k=100)
        eval_rt3 = self.generate(eval_pt3, 32, temperature=0.75, top_k=100)
        eval_rt4 = self.generate(eval_pt4, 32, temperature=0.75, top_k=100)
        eval_r1 = self.tokenizer.decode(eval_rt1[0].cpu().numpy().tolist())
        eval_r2 = self.tokenizer.decode(eval_rt2[0].cpu().numpy().tolist())
        eval_r3 = self.tokenizer.decode(eval_rt3[0].cpu().numpy().tolist())
        eval_r4 = self.tokenizer.decode(eval_rt4[0].cpu().numpy().tolist())

        print(f"eval1: {eval_r1}")
        print(f"eval2: {eval_r2}")
        print(f"eval3: {eval_r3}")
        print(f"eval4: {eval_r4}")

        return eval_r1, eval_r2, eval_r3, eval_r4