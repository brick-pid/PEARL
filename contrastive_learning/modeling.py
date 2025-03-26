import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, BertModel, RobertaModel
from dataclasses import dataclass

def mean_pooling_pt(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sentence_reps = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sentence_reps

@dataclass
class RetrievalOutput:
    loss: torch.Tensor
    scores: torch.Tensor
    q_reps: torch.Tensor
    p_reps: torch.Tensor

class RetrievalModel(PreTrainedModel):
    def __init__(self, base_model, temperature: float = 1.0):
        super().__init__(base_model.config)
        self.model = base_model
        self.temperature = temperature

    def forward(self, inputs):
        outputs = self.model(**inputs)
        sentence_reps = mean_pooling_pt(outputs, inputs["attention_mask"])
        # normalize
        sentence_reps = F.normalize(sentence_reps, p=2, dim=1)
        return sentence_reps

    def encode(self, inputs):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = self.model(**inputs)
            sentence_reps = mean_pooling_pt(outputs, inputs["attention_mask"])
            # normalize
            sentence_reps = F.normalize(sentence_reps, p=2, dim=1)
        return sentence_reps
