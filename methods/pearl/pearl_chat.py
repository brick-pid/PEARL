"""
use trained retriever to retrieve code and knowledge
and then use cot based on retrieved content to generate code
"""
from engine import OpenAIChatEngine
from typing import List
from rag.retriever import RetrieverFactory
from rag.embedder import MeanPoolingEmbedder
from omegaconf import OmegaConf
from utils.process_utils import get_long_language_name, extract_mdcode
import os
import importlib

pearl_template = """### Instructions
1. In the **Chain-of-Thought** section, please list out detailed thoughts and steps to solve the problem;
2. In the **Output** section, only output the complete code implementation within markdown code block;
3. Use ###END to indicate the end of an response.

### Context
{context}

### BEGIN
**Problem**:\n{example_problem}
**Chain-of-Thought**:\n{example_cot}
**Output**:\n
```{long_lang}
{example_result}
```
### END

### BEGIN
**Problem**:\n{problem}
**Chain-of-Thought**:\n
"""

class PEARL_CHAT:
    def __init__(self, lang: str, engine: OpenAIChatEngine):
        self.lang = lang
        self.long_lang = get_long_language_name(lang)
        self.engine = engine
        self.cfg = OmegaConf.load("config/pearl.yaml")
        
        retriever_name_replace = self.cfg.retriever_name.replace('/', '_')
        code_index_cache_path = self.cfg.index_cache_dir + '/' + 'code_' + lang + '_' + retriever_name_replace + '.index'
        knowledge_index_cache_path = self.cfg.index_cache_dir + '/' + 'knowledge_' + lang + '_' + retriever_name_replace + '.index'
        knowledgebase_path = self.cfg.knowledgebase_path.format(full_lang=self.long_lang)

        if 'bm25' in self.cfg.retriever_name:
            retriever_type = "bm25"
            embedder = None
        else:
            retriever_type = "dense"
            # init embedder
            embedder = MeanPoolingEmbedder(model=self.cfg.retriever_name, tokenizer=self.cfg.tokenizer_name)

        self.code_retriever = RetrieverFactory.create(retriever_type=retriever_type, data_source_type="code", index_path=code_index_cache_path, data_source=lang, embedder=embedder)
        self.knowledge_retriever = RetrieverFactory.create(retriever_type=retriever_type, data_source_type="knowledge", index_path=knowledge_index_cache_path, data_source=knowledgebase_path, embedder=embedder)
        
        self.example_problem, self.example_cot, self.example_result = self._load_example_prompts()

    def _load_example_prompts(self):
        """Load language-specific example prompts for one-shot CoT"""
        try:
            # Try to load language-specific example
            example_module_name = f"prompts.{self.lang}_example"
            example_module = importlib.import_module(example_module_name)
            example_problem = getattr(example_module, "problem", "")
            example_cot = getattr(example_module, "cot", "")
            example_result = getattr(example_module, "result", "")
        except (ImportError, AttributeError):
            raise ValueError(f"No example prompts found for {self.lang}")
        return example_problem, example_cot, example_result

    def completions(self, prompts: List[str], stop: List[str]):
        """
        Generate completions for a list of prompts
        
        Args:
            prompts: List of prompts to generate completions for
            stop: List of stop sequences
            
        Returns:
            Tuple of (completions, full_completions)
        """
        # Retrieve code and knowledge - only retrieve once
        codes = self.code_retriever.retrieve_results(prompts, top_k=self.cfg.top_k)
        # breakpoint()
        knowledges = self.knowledge_retriever.retrieve_results(prompts, top_k=self.cfg.top_k)
        
        # Build prompts with chain of thought
        pearl_prompts = []
        for c, k, p in zip(codes, knowledges, prompts):
            # prompt
            ctx = '\n\n'.join(k) + '\n\n' + '\n\n'.join(c)
            prompt = pearl_template.format(context=ctx, example_problem=self.example_problem, example_cot=self.example_cot, 
                                           long_lang=self.long_lang, example_result=self.example_result, problem=p)
            pearl_prompts.append(prompt)
        
        # breakpoint()
        stop = ['### END']
        # Generate completions with CoT
        # breakpoint()
        raw_completions = self.engine.generate(pearl_prompts, stop=stop)
        codes = extract_mdcode(raw_completions, self.long_lang)
        return codes, raw_completions
