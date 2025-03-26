"""
The HyCode method is first prompt the LLM to generate a hypothetical solution to the problem, 
then use the hypothetical solution to retrieve the most relevant code snippets,
then augment the prompt with the hypothetical solution and the code snippets
"""

"""
The HyCode method first prompts the LLM to generate a hypothetical solution to the problem, 
then uses the hypothetical solution to retrieve the most relevant code snippets,
then augments the prompt with the hypothetical solution and the code snippets.

input:
    - problem: str, problem description
    - inference server: inference server
output:
    - BaseOutput:
        - code: str, completion code only, without --full-function parameter
        - full_completion: str, complete prompt
"""
from rag.retriever import CodeRetriever
from omegaconf import OmegaConf
from typing import List
from engine import VLLM
from pathlib import Path

def load_prompt_template(template_name: str) -> str:
    """Load prompt template from file."""
    template_path = Path(__file__).parent / "prompts" / template_name
    with open(template_path, "r") as f:
        return f.read()

class HyCode:
    def __init__(self, lang: str, engine: VLLM):
        self.lang = lang
        self.engine = engine
        self.cfg = OmegaConf.load("config/hycode.yaml")
        
        # Initialize code retriever
        retriever_name_replace = self.cfg.retriever_name.replace('/', '_')
        code_index_cache_path = self.cfg.index_cache_dir + '/' + 'code_' + lang + '_' + retriever_name_replace + '.index'
        self.code_retriever = CodeRetriever(self.cfg.retriever_name, code_index_cache_path, lang)

    def completions(self, prompts: List[str], stop: List[str]):
        # 1. Generate hypothetical solutions
        hycodes = self.engine.generate(prompts, stop=stop)
        hycodes = [p + c for p, c in zip(prompts, hycodes)]

        # 2. Use hypothetical solutions to retrieve relevant code snippets
        relevant_codes = self.code_retriever.retrieve(hycodes, top_k=self.cfg.top_k)
        ctxs = []
        for rels in relevant_codes:
            ctx = '\n\n'.join(rels)
            ctxs.append(ctx)

        # 3. Build final prompts with hypothetical solutions and retrieved code
        final_prompts = []
        for prompt, ctx in zip(prompts, ctxs):
            final_prompt = "ctx" + "\n\n" + prompt
            final_prompts.append(final_prompt)

        # 4. Generate final solutions
        completions = self.engine.generate(final_prompts, stop=stop)
        
        # construct full completions: hycode + relevant codes + final solution
        full_completions = []
        for hycode, ctx, completion in zip(hycodes, ctxs, completions):
            full_completion = hycode + '\n\n' + "-" * 30 + '\n\n' + ctx + '\n\n' + completion
            full_completions.append(full_completion)
        return completions, full_completions