from queue import Queue
from threading import Thread, Event, Lock
import json
import orjson
import os
import random
import time
import torch
from tokenizer import Tokenizer
import math
import sys
import uuid


ExampleDataloaderConfig = {
    "dataname1": 100,
    "dataname2": 20 # it means 20/120 of the data will from this dataset
}

def dbg_print_ifLongerThan(msg, startTime):
    if time.time() - startTime > 0.8:
        sys.stdout.write(msg)

class DataLoaderWorker(Thread):
    def __init__(self, datanames, proportion, datanames_to_config, data_rootpath, tokenizer_path, 
                 command_queue, working_queue, stop_event, workder_id, 
                 get_working_buf, swap_buffer, get_working_buf_lock, cache_lock, cache_obj: dict=None):
        super().__init__(daemon=True)
        self.datanames = datanames
        self.proportion = proportion
        self.datanames_to_config = datanames_to_config
        self.data_rootpath = data_rootpath
        self.command_queue = command_queue
        self.working_queue = working_queue
        self.tokenizer = Tokenizer(tokenizer_path)
        self.stop_event = stop_event
        self.worker_id = workder_id
        self.get_working_buf = get_working_buf
        self.swap_buffer = swap_buffer
        self.get_working_buf_lock = get_working_buf_lock
        self.cache_lock = cache_lock
        self.cache_obj = cache_obj

        self.total_datasets_weight = sum(proportion)
    
    def __del__(self):
        #print(f"DataLoaderWorker is stopped")
        sys.stdout.write(f"DataLoaderWorker is stopped\n")

    def run(self):
        while True:
            if self.stop_event.is_set():
                sys.stdout.write(f"DataLoaderWorker exit the loop\n")
                return
            try:
                command = self.command_queue.get(timeout=2)
                sys.stdout.write(f"DataLoaderWorker {self.worker_id}: get command\n")
            except:
                continue
            if command is None:
                continue
            if command['type'] == "assign":
                self.add_task(command['num'])
            elif command['type'] == "load":
                self.add_random_toBuffer(command['file_path'], command['randidx'])
            else:
                raise ValueError(f"Unknown command {command}")

    def add_task(self, num=100000):
        '''
        num: how many rows to add to buf; this number should be large enough to prevent frequent file reading
        '''
        randidx = {}
        for i in range(num):
            dataset_name, in_which_file, r_row = self._get_random_dataIndex()
            file_path = os.path.join(self.data_rootpath, dataset_name, in_which_file)
            if file_path not in randidx:
                randidx[file_path] = []
            randidx[file_path].append(r_row)
        
        # add tasks to command_queue, read each file and add to buf
        task_count = len(randidx.keys())
        task_batch_id = uuid.uuid4()
        for i, file_path in enumerate(randidx.keys()):
            task = {
                "type": "load",
                "file_path": file_path,
                "randidx": randidx[file_path],
                "task_batch_id": task_batch_id
            }
            self.command_queue.put(task)
            if i == task_count - 1:
                self.working_queue.put("sync")
            else:
                self.working_queue.put("no_sync")

    def add_random_toBuffer(self, file_path, randidx):
        rows = []
        if self.cache_obj is not None and file_path in self.cache_obj:
            raw = self.cache_obj[file_path]
        else:
            try:
                with open(file_path, 'r') as f:
                    #raw = json.load(f)
                    raw = orjson.loads(f.read())
            except:
                after_task_command = self.working_queue.get()
                if after_task_command == "sync":
                    self.sync()
                sys.stderr.write(f"Cannot open {file_path}, or fail to parse the file\n")
                return
            if self.cache_obj is not None:
                self.cache_lock.acquire()
                self.cache_obj[file_path] = raw
                self.cache_lock.release()
        for r_row in randidx:
            rows.append(self.tokenizer.encode(raw[str(r_row)])) # it add a list of int to rows
        self.get_working_buf_lock().acquire()
        self.get_working_buf().extend(rows)
        self.get_working_buf_lock().release()

        after_task_command = self.working_queue.get()
        if after_task_command == "sync":
            self.sync()

    def _get_random_dataIndex(self):
        r_dataset = random.random() * self.total_datasets_weight
        dataset_name = None
        for i, p in enumerate(self.proportion):
            if r_dataset < p:
                dataset_name = self.datanames[i]
                break
            r_dataset -= p
        if dataset_name is None:
            raise ValueError("Cannot find dataset_name")

        dataset_config = self.datanames_to_config[dataset_name]
        total_rows = dataset_config['total_rows']
        r_row = random.randint(0, total_rows-1)

        in_which_file = None
        for file_name in dataset_config['file_names']:
            if r_row >= dataset_config['file_names'][file_name]['start_idx'] and r_row < dataset_config['file_names'][file_name]['end_idx']:
                in_which_file = file_name
                break
        
        return dataset_name, in_which_file, r_row
        
    def sync(self):
        self.swap_buffer()

class DataLoader_async:
    def __init__(self, data_rootpath: str, tokenizer_path: str, config: object, worker_num=4, expected_buf_size=50000, use_cache=True):
        self.data_rootpath = data_rootpath
        self._build_datamap(config)
        self.expected_buf_size = max(expected_buf_size, 5000)

        self.command_queue = Queue()
        self.result_queue = Queue()
        self.stop_event = Event()
        self.tokenizer = Tokenizer(tokenizer_path)
        self.buf = []
        self.working_buf = []
        self.workders = []
        self.buf_lock = Lock() # used when swap buffer, or get_batch
        self.working_buf_lock = Lock() # used when workers want to operate on working_buf
        self.cache_lock = Lock() # used when workers want to write on cache
        self.cache_obj = dict() if use_cache else None
        
        for i in range(worker_num):
            worker = DataLoaderWorker(self.datanames, self.proportion, self.datanames_to_config, data_rootpath, 
                                      tokenizer_path, self.command_queue, self.result_queue, self.stop_event, i,
                                      self._get_working_buf, self._swap_buffer, self._get_working_buf_lock, self.cache_lock, self.cache_obj)
            self.workders.append(worker)
            worker.start()
        
        #self._assign_task(min(self.expected_buf_size // 3, 5000)) # prevent waiting for too long for the first loading
        self._assign_task(self.expected_buf_size)
        while len(self.buf) <= 0:
            time.sleep(0.1)
        sys.stdout.write("DataLoader is ready\n")
  
    def __del__(self):
        self.stop_event.set()
        #print("DataLoader is stopped")
        sys.stdout.write("Main DataLoader is stopped\n")

    def get_batch(self, batch_size, seqlen) -> torch.Tensor:
        # dump the result queue to buf
        while len(self.buf) < batch_size:
            sys.stdout.write(f"DataLoader: get_batch: WARNING: do not have enough rows in buffer; \
                             increase expected_buf_size to avoid this problem;\n \
                             current buf size: {len(self.buf)}\n")
            #time.sleep(0.05)
            time.sleep(0.5)
        
        #sys.stdout.write(f"DataLoader: get_batch: before qcquire buf_lock: buf size: {len(self.buf)}\n")
        self.buf_lock.acquire()
        #sys.stdout.write(f"DataLoader: get_batch: after qcquire buf_lock: buf size: {len(self.buf)}\n")
        if len(self.buf) <= min(5000, self.expected_buf_size // 2):
            self._assign_task(int(1.5 * (self.expected_buf_size - len(self.buf))))

        ret = self._sampling(batch_size, seqlen)
        self.buf_lock.release()
        return ret
        
    def _sampling(self, batch_size, seqlen) -> torch.Tensor:
        #ret = torch.Tensor.new_full((batch_size, seqlen), fill_value=self.tokenizer.pad_id())
        ret = torch.zeros((batch_size, seqlen), dtype=torch.int32)
        #ret = ret.fill_(self.tokenizer.pad_id())
        ret = ret.fill_(-1)
        tmp_t = time.time()
        for i in range(batch_size):
            dbg_print_ifLongerThan("line 214\n", tmp_t)
            row = self.buf.pop()
            dbg_print_ifLongerThan("line 216\n", tmp_t)
            ret[i, :] = self._cut_tokens(row, seqlen)[0, :]
            dbg_print_ifLongerThan("line 218\n", tmp_t)
        return ret

    def _cut_tokens(self, tokens, seqlen):
        if len(tokens) <= seqlen: # if the tokens is shorter than seqlen, pad it at the front
            #ret = torch.Tensor.new_full((1, seqlen), fill_value=self.tokenizer.pad_id())
            ret = torch.zeros((1, seqlen), dtype=torch.int32)
            #ret = ret.fill_(self.tokenizer.pad_id())
            ret = ret.fill_(-1)
            tokens_tensor = torch.Tensor(tokens).view(1, -1)
            ret[0, -len(tokens):] = tokens_tensor[0, :]
            return ret
        else:
            count_slice = math.ceil(len(tokens) / seqlen)
            r = random.randint(0, count_slice-1)
            if r == count_slice-1:
                return torch.Tensor(tokens[-seqlen:]).view(1, -1)
            else:
                return torch.Tensor(tokens[r*seqlen:(r+1)*seqlen]).view(1, -1)

    
    def _build_datamap(self, config: object):
        self.datanames_to_config = {}
        self.proportion = []
        self.datanames = []
        for dataname in config.keys():
            if not os.path.exists(os.path.join(self.data_rootpath, f"{dataname}.json")):
                raise FileNotFoundError(f"Cannot find {dataname}.json in {self.data_rootpath}")
            with open(os.path.join(self.data_rootpath, f"{dataname}.json"), 'r') as f:
                self.datanames_to_config[dataname] = json.load(f)
                self.datanames.append(dataname)
                self.proportion.append(config[dataname])
                sys.stdout.write(f"DataLoader: {dataname} is loaded\n")

    def _assign_task(self, num):
        command = {
            "type": "assign",
            "num": num
        }
        self.command_queue.put(command)
    
    def _swap_buffer(self):
        # record the time to swap buffer
        t = time.time()
        self.buf_lock.acquire()
        self.working_buf_lock.acquire()
        self.working_buf.extend(self.buf)
        random.shuffle(self.working_buf)
        self.buf = self.working_buf
        self.working_buf = []
        self.working_buf_lock.release()
        self.buf_lock.release()
        sys.stdout.write(f"DataLoader: ------- swap_buffer takes {time.time() - t} seconds\n")
    
    def _get_working_buf(self):
        return self.working_buf
    
    def _get_working_buf_lock(self):
        return self.working_buf_lock


class DataLoader:
    def __init__(self, data_rootpath: str, tokenizer_path: str, config: object):
        self.data_rootpath = data_rootpath
        self._build_datamap(config)
        self.total_datasets_weight = sum(self.proportion)
        self.datasets = dict()
        sys.stdout.write("DataLoader: start loading data\n")
        self._load()
        self.tokenizer = Tokenizer(tokenizer_path)

        sys.stdout.write("DataLoader is ready\n")
    
    def _load(self):
        for dataname in self.datanames:
            self.datasets[dataname] = dict()
            for file_name in self.datanames_to_config[dataname]['file_names']:
                with open(os.path.join(self.data_rootpath, dataname, file_name), 'r') as f:
                    self.datasets[dataname][file_name] = orjson.loads(f.read())

    def _build_datamap(self, config: object):
        self.datanames_to_config = {}
        self.proportion = []
        self.datanames = []
        for dataname in config.keys():
            if not os.path.exists(os.path.join(self.data_rootpath, f"{dataname}.json")):
                raise FileNotFoundError(f"Cannot find {dataname}.json in {self.data_rootpath}")
            with open(os.path.join(self.data_rootpath, f"{dataname}.json"), 'r') as f:
                self.datanames_to_config[dataname] = json.load(f)
                self.datanames.append(dataname)
                self.proportion.append(config[dataname])
                sys.stdout.write(f"DataLoader: {dataname} found\n")
    
    def get_batch(self, batch_size, seqlen) -> torch.Tensor:
        ret = torch.ones((2,), dtype=torch.int64).new_full((batch_size, seqlen), fill_value=(self.tokenizer.pad_id))
        for i in range(batch_size):
            dataset_name, in_which_file, r_row = self._get_random_dataIndex()
            text = self.datasets[dataset_name][in_which_file][str(r_row)]
            tokens = self.tokenizer.encode(text)
            ret[i, :] = self._cut_tokens(tokens, seqlen)[0, :]
        return ret
    
    def _sampling(self, batch_size, seqlen) -> torch.Tensor:
        ret = torch.ones((2,), dtype=torch.int64).new_full((batch_size, seqlen), fill_value=self.tokenizer.pad_id)
        ret = ret.fill_(-1)
        tmp_t = time.time()
        for i in range(batch_size):
            dbg_print_ifLongerThan("line 214\n", tmp_t)
            row = self.buf.pop()
            dbg_print_ifLongerThan("line 216\n", tmp_t)
            ret[i, :] = self._cut_tokens(row, seqlen)[0, :]
            dbg_print_ifLongerThan("line 218\n", tmp_t)
        return ret

    def _cut_tokens(self, tokens, seqlen):
        if len(tokens) <= seqlen: # if the tokens is shorter than seqlen, pad it at the front
            ret = torch.ones((2,), dtype=torch.int64).new_full((1, seqlen), fill_value=self.tokenizer.pad_id)
            ret = ret.fill_(-1)
            tokens_tensor = torch.Tensor(tokens).view(1, -1)
            ret[0, -len(tokens):] = tokens_tensor[0, :]
            return ret
        else:
            count_slice = math.ceil(len(tokens) / seqlen)
            r = random.randint(0, count_slice-1)
            if r == count_slice-1:
                return torch.Tensor(tokens[-seqlen:]).view(1, -1).long()
            else:
                return torch.Tensor(tokens[r*seqlen:(r+1)*seqlen]).view(1, -1).long()
    

    def _get_random_dataIndex(self):
        r_dataset = random.random() * self.total_datasets_weight
        dataset_name = None
        for i, p in enumerate(self.proportion):
            if r_dataset < p:
                dataset_name = self.datanames[i]
                break
            r_dataset -= p
        if dataset_name is None:
            raise ValueError("Cannot find dataset_name")

        dataset_config = self.datanames_to_config[dataset_name]
        total_rows = dataset_config['total_rows']
        r_row = random.randint(0, total_rows-1)

        in_which_file = None
        for file_name in dataset_config['file_names']:
            if r_row >= dataset_config['file_names'][file_name]['start_idx'] and r_row < dataset_config['file_names'][file_name]['end_idx']:
                in_which_file = file_name
                break
        
        return dataset_name, in_which_file, r_row