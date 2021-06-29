from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from easydict import EasyDict as edict
from abc import abstractmethod

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pytorch_lightning as pl


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()
    
    @property
    def features(self):
        pass
    
    @property
    def split(self):
        pass
    
    def _get_end_answer_index(self, answers_src: edict, context: str):
        answers = edict(answers_src.copy())
        len_answers = len(answers.text)
        answers.answer_end = list()
        
        for i in range(len_answers):
            gold_answer = answers.text[i]
            start_ans_idx = answers.answer_start[i]
            end_ans_idx = start_ans_idx + len(gold_answer)
            
            if context[start_ans_idx:end_ans_idx] == gold_answer:
                answers.answer_end.append(end_ans_idx)
            elif context[start_ans_idx-1:end_ans_idx-1] == gold_answer:
                answers.answer_start[i] = start_ans_idx-1
                answers.answer_end.append(end_ans_idx-1)
            elif context[start_ans_idx-2:end_ans_idx-2] == gold_answer:
                answers.answer_start[i] = start_ans_idx-2
                answers.answer_end.append(end_ans_idx-2)
            else:
                raise Exception("Invalid start:end annotation")
        
        return dict(answers=answers)
            

class SQUADv1(BaseDataset):
    def __init__(self, 
                 train_set: bool=True, 
                 **kwargs):
        super().__init__()
        
        if train_set:
            self._split = "train"
            self._dataset = load_dataset("squad", **kwargs)['train']
        else:
            self._split = "dev/test"
            self._dataset = load_dataset("squad", **kwargs)['validation']
        
        self._features = self._dataset.features
        self._num_rows = self._dataset.num_rows
    
    @property
    def features(self):
        return list(self._features.keys())
    
    @property
    def split(self):
        return self._split
    
    
    def __len__(self):
        return self._num_rows
    
    def __getitem__(self, idx):
        data = edict(self._dataset[idx])
        answers_with_end = self._get_end_answer_index(data.answers, data.context)
        data.update(answers_with_end)
        
        return data 



class SQUADv2(BaseDataset):
    def __init__(self, 
                 train_set: bool=True, 
                 **kwargs):
        super().__init__()
        
        if train_set:
            self._split = "train"
            self._dataset = load_dataset("squad_v2", **kwargs)['train']
        else:
            self._split = "dev/test"
            self._dataset = load_dataset("squad_v2", **kwargs)['validation']
        
        self._features = self._dataset.features
        self._num_rows = self._dataset.num_rows
    
    @property
    def features(self):
        return list(self._features.keys())
    
    @property
    def split(self):
        return self._split
    
    def __len__(self):
        return self._num_rows
    
    def __getitem__(self, idx):
        data = edict(self._dataset[idx])
        answers_with_end = self._get_end_answer_index(data.answers, data.context)
        data.update(answers_with_end)
        
        return data



name2dataset = dict(squad=SQUADv1, squadv2=SQUADv2)
    
class SQUADDataLoader(pl.LightningDataModule):
    def __init__(self, dataset:str, dataset_cfg:dict={}, trainLoader_cfg:dict={}, valLoader_cfg:dict={}, testLoader_cfg:dict={}):
        super().__init__()
        
        # set dataloader attributes and config
        self._dataset_name = dataset 
        self._dataset_cfg = dataset_cfg
        self._trainLoader_cfg = trainLoader_cfg
        self._valLoader_cfg = valLoader_cfg
        self._testLoader_cfg = testLoader_cfg
        
        # init train/val/test set
        self._init_train_set()
        self._init_val_set()     
    

    def _init_train_set(self):
        self.train_set = name2dataset[self._dataset_name](train_set=True, **self._dataset_cfg)
    
    def _init_val_set(self):
        self.val_set = name2dataset[self._dataset_name](train_set=False, **self._dataset_cfg)

    def train_dataloader(self):
        self.train_loader = DataLoader(self.train_set, **self._trainLoader_cfg)
        return self.train_loader
    
    def val_dataloader(self):
        self.val_loader = DataLoader(self.val_set, **self._valLoader_cfg)
        return self.val_loader
    
    def test_dataloader(self, batch_size):
        # self.test_loader = DataLoader(self.val_set, **self._testLoader_cfg)
        return None   