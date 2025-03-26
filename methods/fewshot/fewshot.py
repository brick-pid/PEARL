"""
few shot

input:
    - problem: str, 问题描述
    - inference server: 推理服务器
output:
    - BaseOutput:
        - code: str, 仅包括补全代码，不使用 --full-function 参数
        - full_completion: str, 完整prompt
"""
import random
from utils.process_utils import get_long_language_name
from datasets import load_dataset
from typing import List
from engine import VLLM


def load_code_snippets(lang: str):
    """
    Load code snippets from a multipl-t dataset
    """
    long_lang = get_long_language_name(lang)
    dataset = load_dataset("nuprl/MultiPL-T", split=long_lang)
    return dataset['content']

class FewShot:
    def __init__(self, lang: str, engine: VLLM) -> None:
        self.lang = lang
        self.k = 3
        self.code_snippets = load_code_snippets(lang)
        self.engine = engine
    
    def random_sample(self):
        return random.sample(self.code_snippets, self.k)

    def completions(self, prompts: List[str], stop: List[str]):
        few_shot_prompts = []
        for p in prompts:
            fewshots = self.random_sample()
            fs_prompt = '\n\n'.join(fewshots) + '\n\n' + p
            few_shot_prompts.append(fs_prompt)

        # breakpoint()
        codes = self.engine.generate(few_shot_prompts, stop=stop)
        return codes, codes
            
