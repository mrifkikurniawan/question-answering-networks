from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from easydict import EasyDict as edict

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pytorch_lightning as pl

from qan.datasets.entities import Question, Answer, Context



class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()
    
    @property
    def features(self):
        pass

    @property
    def tokenizer(self):
        pass
    
    @property
    def split(self):
        pass
    
    def _postprocess(self):
        pass



class SQUADv1(Dataset):
    def __init__(self, name: str, tokenizer: callable, train_set: bool=True, **kwargs):
        super().__init__()
        
        if train_set:
            self._split = "train"
            self._dataset = load_dataset("squad", **kwargs)['train']
        else:
            self._split = "dev/test"
            self._dataset = load_dataset("squad", **kwargs)['validation']
        
        self._name = name
        self._features = self._dataset.features
        self._num_rows = self._dataset.num_rows
        self._tokenizer = tokenizer
    
    @property
    def features(self):
        return list(self._features.keys())

    @property
    def tokenizer(self):
        return self._tokenizer
    
    @property
    def split(self):
        return self._split
    
    def _postprocess(self, text):
        pass
    
    def __len__(self):
        return self._num_rows
    
    def __getitem__(self, idx):
        data = edict(self._dataset[idx])
        
        # create instance
        question = Question(data.question)
        answer = Answer(data.answers)
        context = Context(data.context)
        
        # tokenize
        question.tokenize(self._tokenizer, self._postprocess)
        answer.tokenize(self._tokenizer, self._postprocess)
        context.tokenize(self._tokenizer, self._postprocess)
        
        new_data = edict(answers=answer, question=question, context=context)
        
        return data.update(new_data) 



class SQUADv2(Dataset):
    def __init__(self, name: str, tokenizer: callable, train_set: bool=True, **kwargs):
        super().__init__()
        
        if train_set:
            self._split = "train"
            self._dataset = load_dataset("squad_v2", **kwargs)['train']
        else:
            self._split = "dev/test"
            self._dataset = load_dataset("squad_v2", **kwargs)['validation']
        
        self._name = name
        self._features = self._dataset.features
        self._num_rows = self._dataset.num_rows
        self._tokenizer = tokenizer
    
    @property
    def features(self):
        return list(self._features.keys())

    @property
    def tokenizer(self):
        return self._tokenizer
    
    @property
    def split(self):
        return self._split
    
    def _postprocess(self, text):
        pass
    
    def __len__(self):
        return self._num_rows
    
    def __getitem__(self, idx):
        data = edict(self._dataset[idx])
        
        # create instance
        question = Question(data.question)
        answer = Answer(data.answers)
        context = Context(data.context)
        
        # tokenize
        question.tokenize(self._tokenizer, self._postprocess)
        answer.tokenize(self._tokenizer, self._postprocess)
        context.tokenize(self._tokenizer, self._postprocess)
        
        new_data = edict(answers=answer, question=question, context=context)
        
        return data.update(new_data) 



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
    
    def prepare_data(self):
        pass
    
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
        self.test_loader = DataLoader(self.val_set, **self._testLoader_cfg)
        return self.test_loader