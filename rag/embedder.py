from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import List, Union
from transformers import AutoTokenizer, AutoModel
from contrastive_learning.modeling import RetrievalModel
import os


class BaseEmbedder(ABC):
    def __init__(self, model: Union[str, AutoModel], tokenizer: Union[str, AutoTokenizer] = None, device: str = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(model, str):
            self.model = AutoModel.from_pretrained(model, trust_remote_code=True).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model if tokenizer is None else tokenizer)
        else:
            self.device = model.device
            if tokenizer is None:
                raise ValueError("When passing a model instance, tokenizer must be provided")
            self.tokenizer = tokenizer if isinstance(tokenizer, AutoTokenizer) else AutoTokenizer.from_pretrained(tokenizer)
        
        self.model.eval()

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        pass

    def _normalize(self, embeddings: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

class MeanPoolingEmbedder(BaseEmbedder):
    def encode(self, texts: List[str]) -> np.ndarray:
        with torch.no_grad():
            encoded_input = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            ).to(self.device)
            
            model_output = self.model(**encoded_input)
            attention_mask = encoded_input['attention_mask']
            embeddings = self._mean_pooling(model_output, attention_mask)
        return self._normalize(embeddings)
    
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9) 
