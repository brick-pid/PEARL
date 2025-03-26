"""
retrieval-based few shot

input:
    - problem: str, problem description
    - inference server: inference server
output:
    - BaseOutput:
        - code: str, completion code only, without --full-function parameter
        - full_completion: str, complete prompt
"""
from rag.retriever import RetrieverFactory
from omegaconf import OmegaConf
from typing import List
from engine import VLLM

class RetFewShot:
    def __init__(self, lang: str, engine: VLLM):
        self.lang = lang
        self.engine = engine
        self.cfg = OmegaConf.load("config/ret_fewshot.yaml")
        
        retriever_name_replace = self.cfg.retriever_name.replace('/', '_')
        code_index_cache_path = self.cfg.index_cache_dir + '/' + 'code_' + lang + '_' + retriever_name_replace + '.index'
        knowledge_index_cache_path = self.cfg.index_cache_dir + '/' + 'knowledge_' + lang + '_' + retriever_name_replace + '.index'

        if 'bm25' in self.cfg.retriever_name:
            retriever_type = "bm25"
        else:
            retriever_type = "dense"
        self.code_retriever = RetrieverFactory.create(retriever_type=retriever_type, data_source_type="code", index_path=code_index_cache_path, data_source=lang, embedder=None)
        self.knowledge_retriever = RetrieverFactory.create(retriever_type=retriever_type, data_source_type="knowledge", index_path=knowledge_index_cache_path, data_source=self.cfg.knowledgebase_path, embedder=None)


    def completions(self, prompts: List[str], stop: List[str]):
        # retrieve code and knowledge
        codes = self.code_retriever.retrieve(prompts, top_k=self.cfg.top_k)
        knowledges = []
        for p in prompts:
            k = self.knowledge_retriever.retrieve([p], top_k=self.cfg.top_k)
            k = [item['knowledge_entity'] + ' ' + item['intent'] + ' ' + item['content'] + ' ' + item['code_demo'] for item in k]
            knowledges.append(k[:self.cfg.top_k])
        # build prompt
        ret_prompts = []
        for c, k, p in zip(codes, knowledges, prompts):
            fewshots = '\n\n'.join(k) + '\n\n' + '\n\n'.join(c)
            ret_prompt = fewshots + '\n\n' + p
            ret_prompts.append(ret_prompt)

        # breakpoint()
        # generate code
        completions = self.engine.generate(ret_prompts, stop=stop)
        
        return completions, completions
