from torch.utils.data import Dataset
from typing import Dict
from dataclasses import dataclass

"""
下面这个数据集的作用似乎只是将query和passage编码成token
"""

@dataclass
class RetrDataset(Dataset):
    def __init__(self, queries: Dict[str, str], passages: Dict[str, str], tokenizer):
        self.queries = queries
        self.passages = passages
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.queries)
        
    def __getitem__(self, idx):
        query = self.queries[idx]
        pos_passage = self.passages[idx] # 正例
        
        # 编码
        query_encoded = self.tokenizer(
            query,
            max_length=32,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        passage_encoded = self.tokenizer(
            pos_passage, 
            max_length=128,
            padding='max_length', 
            truncation=True,
            return_tensors='pt'
        )
        
        return query_encoded, passage_encoded