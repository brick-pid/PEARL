"""
inference engine, support openai and vllm.
"""
import os
import torch
from vllm import LLM, SamplingParams
from openai import OpenAI
import numpy as np
import time
import asyncio

class VLLM:
    def __init__(self, name, revision, tokenizer_name=None, tokenizer_revision=None, sampling_params=None, num_gpus=1):
        dtype = os.getenv("DTYPE")
        self.model = LLM(
            model=name,
            tokenizer=tokenizer_name,
            dtype=dtype,
            revision=revision,
            max_model_len=4096,
            tokenizer_revision=tokenizer_revision,
            trust_remote_code=True,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=0.85,
        )
        self.sampling_params = sampling_params

    def generate(self, prompts, use_tqdm=False, stop=None):
        """
        return a list of completion strings
        """
        if stop is not None:
            self.sampling_params.stop = stop
        codes = self.model.generate(prompts, self.sampling_params, use_tqdm=use_tqdm)
        return [o.outputs[0].text for o in codes]

    def score_candidates(self, x, y, candidates):
        """
        Use the LM to score the candidates by calculating the conditional probability of the target response given the candidate and the prompt.
        Higher target response probability indicates better/more likely candidates.

        score(c) = P_LM(y|c, x)
        
        Args:
            x (str): The input prompt
            y (str): The target response
            candidates (List[str]): List of conditional candidate to score
            
        Returns:
            List[dict]: each dict contains the candidate and its perplexity score
        """
        # Create full prompts by combining query with each candidate
        prompts = [c + x + y for c in candidates]
        
        # Generate logprobs for each prompt
        outputs = self.model.generate(
            prompts,
            SamplingParams(
                temperature=0.0,
                top_p=1,
                max_tokens=1,
                logprobs=0,
                prompt_logprobs=0,
            )
        )
        scores = []
        for output, candidate in zip(outputs, candidates):
            # Get logprobs for the target sequence y
            logprobs = output.outputs[0].logprobs[-len(y):]
            
            # Calculate metrics
            # Arithmetic mean of logprobs
            arithmetic_avg_logprob = sum(logprobs) / len(logprobs)
            
            # Geometric mean of probabilities = exp(mean(logprobs))
            geometric_avg_prob = torch.exp(torch.tensor(arithmetic_avg_logprob)).item()
            # Geometric perplexity is reciprocal of geometric average probability
            geometric_perplexity = 1.0 / geometric_avg_prob
            perplexity = torch.exp(-torch.tensor(arithmetic_avg_logprob)).item()
            
            scores.append({
                "text": candidate,
                "avg_logprob": arithmetic_avg_logprob,
                "geometric_avg_prob": geometric_avg_prob,
                "perplexity": perplexity
            })
            
        # Sort by perplexity (lower is better)
        scores.sort(key=lambda x: x["perplexity"])
        return scores


class OpenAIChatEngine:
    """
    Openai chat engine
    """
    def __init__(self, lm_name, base_url: str = None, api_key: str = None):
        self.engine = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.lm_name = lm_name

    def generate(self, prompts, stop=None):
        # change the prompts to chat format
        prompts = [[{"role": "user", "content": p}] for p in prompts]
        
        # 使用异步方式并行发送所有请求
        async def _async_generate():
            async def _get_completion(p):
                response = await asyncio.to_thread(
                    self.engine.chat.completions.create,
                    model=self.lm_name,
                    messages=p,
                    max_tokens=2048,
                    temperature=0.2,
                    top_p=0.95,
                    stop=stop,
                )
                return response.choices[0].message.content
            
            # 并行发送所有请求
            tasks = [_get_completion(p) for p in prompts]
            return await asyncio.gather(*tasks)
        
        # 运行异步函数并返回结果
        return asyncio.run(_async_generate())

class OpenAIEngine:
    """
    OpenAI compatible engine
    """
    def __init__(self, lm_name, base_url: str = None, api_key: str = None):
        self.engine = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.lm_name = lm_name
    
    def generate(self, prompts, stop=None):
        # retry_delays = [20, 30, 50, 100, 200]  # Delays in seconds
        
        # for attempt, delay in enumerate(retry_delays):
        #     try:
        #         response = self.engine.completions.create(
        #             model=self.lm_name,
        #             prompt=prompts,
        #             max_tokens=2048,
        #             stop=stop,
        #             temperature=0.2,
        #             top_p=0.95,
        #         )
        #         break
        #     except Exception as e:
        #         if attempt == len(retry_delays) - 1:  # Last attempt
        #             raise RuntimeError("Failed to get response after all retry attempts") from e
        #         print(f"Error occurred. Retrying in {delay} seconds... (Attempt {attempt + 1}/{len(retry_delays)})")
        #         time.sleep(delay)
        # # breakpoint()
        response = self.engine.completions.create(
                model=self.lm_name,
                prompt=prompts,
                max_tokens=2048,
                stop=stop,
                temperature=0.2,
                top_p=0.95,
            )
        return [o.text for o in response.choices]
    
    def score_candidates(self, prefix, suffix, candidates, lm_tokenizer):
        """
        Score candidates using geometric mean of token probabilities.
        
        Args:
            prefix: str, the prefix context
            suffix: str, the target completion
            candidates: List[str], list of candidates to score
            
        Returns:
            List[dict]: Sorted list of dicts containing candidates and their scores
        """
        # Combine prefix and suffix with each candidate
        prompts = [c + prefix + suffix for c in candidates]
        retry_delays = [20, 30, 50, 100, 200]  # Delays in seconds
        
        for attempt, delay in enumerate(retry_delays):
            try:
                response = self.engine.completions.create(
                    model=self.lm_name,
                    prompt=prompts, 
                    temperature=0,
                    max_tokens=1,
                    logprobs=1,
                    echo=True
                )
                break
            except Exception as e:
                if attempt == len(retry_delays) - 1:  # Last attempt
                    raise RuntimeError("Failed to get response after all retry attempts") from e
                print(f"Error occurred. Retrying in {delay} seconds... (Attempt {attempt + 1}/{len(retry_delays)})")
                time.sleep(delay)

        # get suffix token len
        suffix_token_len = len(lm_tokenizer.encode(suffix, add_special_tokens=False))
        scores = []
        for candidate, choice in zip(candidates, response.choices):
            # Get logprobs for suffix portion
            logprobs = choice.logprobs.token_logprobs[-suffix_token_len:]

            # Calculate metrics
            ari_avg_prob = np.mean(np.exp(logprobs))
            geo_avg_prob = np.exp(np.mean(logprobs))
            perplexity = np.exp(-np.mean(logprobs))
            
            scores.append({
                "text": candidate,
                "ari_avg_prob": ari_avg_prob,
                "geo_avg_prob": geo_avg_prob,
                "perplexity": perplexity
            })
        
        # Sort by geometric average probability (higher is better)
        scores.sort(key=lambda x: x["geo_avg_prob"], reverse=True)
        return scores


