from dataclasses import dataclass
from tqdm import tqdm
import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
import pandas as pd


class IndexingTrainDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer
    ):
        self.train_data=pd.read_csv(path_to_data, \
                     names=['query', 'content'],\
                     header=None, sep='\t',encoding='utf8')

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.total_len = len(self.train_data)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        data = self.train_data.iloc[item]
        query=data['query']
        content=data['content']
        content_ids = self.tokenizer(content,
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        return content_ids,query


class GenerateDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
    ):
        self.data = []
        with open(path_to_data, 'r') as f:
            for data in f:
                title, content = data.split('\t')
                self.data.append((title, f'{content}'))

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.total_len = len(self.data)


    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        title, content = self.data[item]
        input_ids = self.tokenizer(content,
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        return input_ids, item


@dataclass
class IndexingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        queries = [x[1] for x in features]
        labels = self.tokenizer(
            queries, padding="longest", return_tensors="pt"
        ).input_ids
        inputs=super().__call__(input_ids)
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        return inputs


@dataclass
class QueryEvalCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        ids = [x[1] for x in features]
        inputs = super().__call__(input_ids)
        inputs['labels'] = ids
        
        return inputs
