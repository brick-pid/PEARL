from engine import VLLM, OpenAIEngine
from typing import List
from rag.retriever import CodeRetriever, KnowledgeRetriever
from rag.embedder import MeanPoolingEmbedder
from omegaconf import OmegaConf
from utils.process_utils import get_long_language_name
import importlib

pearl_template = """### Context
{context}

### BEGIN
**Problem**:\n{example_problem}
**Output**:\n{example_result}
### END

### BEGIN
**Problem**:\n{problem}
**Output**:\n
"""

class PEARL_wo_cot:
    def __init__(self, lang: str, engine: VLLM|OpenAIEngine):
        self.lang = lang
        self.long_lang = get_long_language_name(lang)
        self.engine = engine
        self.cfg = OmegaConf.load("config/pearl.yaml")
        
        retriever_name_replace = self.cfg.retriever_name.replace('/', '_')
        code_index_cache_path = self.cfg.index_cache_dir + '/' + 'code_' + lang + '_' + retriever_name_replace + '.index'
        knowledge_index_cache_path = self.cfg.index_cache_dir + '/' + 'knowledge_' + lang + '_' + retriever_name_replace + '.index'
        knowledgebase_path = self.cfg.knowledgebase_path.format(full_lang=self.long_lang)

        # init embedder
        embedder = MeanPoolingEmbedder(model=self.cfg.retriever_name, tokenizer=self.cfg.tokenizer_name)
        self.code_retriever = CodeRetriever(code_index_cache_path, lang, embedder)
        self.knowledge_retriever = KnowledgeRetriever(knowledge_index_cache_path, knowledgebase_path, embedder)
        
        self.example_problem, self.example_result = self._load_example_prompts()

    def _load_example_prompts(self):
        """Load language-specific example prompts for one-shot"""
        try:
            # Try to load language-specific example
            example_module_name = f"prompts.{self.lang}_example"
            example_module = importlib.import_module(example_module_name)
            example_problem = getattr(example_module, "problem", "")
            example_result = getattr(example_module, "result", "")
        except (ImportError, AttributeError):
            raise ValueError(f"No example prompts found for {self.lang}")
        return example_problem, example_result

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
        knowledges = self.knowledge_retriever.retrieve_results(prompts, top_k=self.cfg.top_k)
        
        # Build prompts with chain of thought
        pearl_prompts = []
        for c, k, p in zip(codes, knowledges, prompts):
            # prompt
            ctx = '\n\n'.join(k) + '\n\n' + '\n\n'.join(c)
            prompt = pearl_template.format(context=ctx, example_problem=self.example_problem, 
                                           example_result=self.example_result, problem=p)
            pearl_prompts.append(prompt)

        stop = ['### END']
        raw_completions = self.engine.generate(pearl_prompts, stop=stop)
        codes = [r.strip() for r in raw_completions]
        return codes, raw_completions

        
