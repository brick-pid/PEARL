"""
raw method, just generate the code directly with the model
"""
from typing import List
from engine import VLLM

class RAW():
    def __init__(self, lang: str, engine: VLLM) -> None:
        self.lang = lang
        self.engine = engine

    def completions(self, prompts: List[str], stop: List[str]):
        completions = self.engine.generate(prompts, stop=stop)
        return completions, []
