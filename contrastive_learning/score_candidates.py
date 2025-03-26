"""
Score candidates from the candidate pool.
"""
from typing import List, Dict, Any
from dataclasses import dataclass
import json
import hydra
from omegaconf import DictConfig
from engine import OnlineEngine
from vllm import SamplingParams
import os
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer
import time
from openai import APIConnectionError
BASE_DIR = os.environ['BASE_DIR']

@dataclass
class ScoredCandidate:
    text: str
    perplexity: float
    geometric_avg_prob: float

@dataclass
class ContrastiveSample:
    query: str
    pos_samples: List[str]
    neg_samples: List[str]
    pos_scores: List[float]
    neg_scores: List[float]

def select_samples(
    scored_candidates: List[ScoredCandidate],
    k: int
) -> tuple[List[str], List[str], List[float], List[float]]:
    """Select top-k and bottom-k samples"""
    sorted_candidates = sorted(
        scored_candidates,
        key=lambda x: x.perplexity
    )
    
    pos_samples = [c.text for c in sorted_candidates[:k]]
    neg_samples = [c.text for c in sorted_candidates[-k:]]
    pos_scores = [c.geometric_avg_prob for c in sorted_candidates[:k]]
    neg_scores = [c.geometric_avg_prob for c in sorted_candidates[-k:]]
    
    return pos_samples, neg_samples, pos_scores, neg_scores

def score_with_retry(engine, prefix, suffix, candidates, prefix_len, suffix_len, max_retries=4):
    """Score candidates with retry logic for connection errors"""
    retry_delays = [1, 3, 5, 10]  # Delays in minutes
    
    for attempt in range(max_retries):
        try:
            return engine.score_candidates(
                prefix,
                suffix,
                candidates,
                prefix_len=prefix_len,
                suffix_len=suffix_len
            )
        except APIConnectionError as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e
            
            delay_minutes = retry_delays[attempt]
            print(f"Connection error. Retrying in {delay_minutes} minutes...")
            time.sleep(delay_minutes * 60)
    
    raise RuntimeError("Failed after all retry attempts")

@hydra.main(config_path=f"{BASE_DIR}/config", config_name="contrastive", version_base="1.3")
def main(cfg: DictConfig):
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=1, logprobs=0, prompt_logprobs=0)
    engine = OnlineEngine(cfg.model_name, cfg.base_url, cfg.api_key)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    
    # Initialize lists to collect all examples
    examples: List[Dict[str, Any]] = []
    
    with open(cfg.candidate_pool_path, 'r') as f_in:
        for line in tqdm(f_in, desc="Scoring candidates"):
            pool = json.loads(line)
            
            # Get token lengths for prefix and suffix
            prefix_tokens = tokenizer(pool['prefix'], return_tensors='np', add_special_tokens=False)
            suffix_tokens = tokenizer(pool['suffix'], return_tensors='np', add_special_tokens=False)
            prefix_len = prefix_tokens.input_ids.shape[1]
            suffix_len = suffix_tokens.input_ids.shape[1]
            
            scored_code_candidates = score_with_retry(
                engine,
                pool['prefix'],
                pool['suffix'],
                pool['code_candidates'],
                prefix_len=prefix_len,
                suffix_len=suffix_len
            )
            scored_knowledge_candidates = score_with_retry(
                engine,
                pool['prefix'],
                pool['suffix'],
                pool['knowledge_candidates'],
                prefix_len=prefix_len,
                suffix_len=suffix_len
            )
            
            examples.append({
                "content": pool['content'],
                "prefix": pool['prefix'],
                "suffix": pool['suffix'],
                "scored_code_candidates": scored_code_candidates,
                "scored_knowledge_candidates": scored_knowledge_candidates
            })
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_list(examples)
    
    # Save the dataset
    dataset.save_to_disk(cfg.output_path)

if __name__ == "__main__":
    main()
