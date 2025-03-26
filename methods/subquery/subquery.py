from typing import List
from engine import VLLM
from rag.retriever import CodeRetriever, KnowledgeRetriever
from utils.process_utils import extract_content, load_shot, get_long_language_name, extract_mdcode, split_cots
from prompts.gen_prompts import t_problem, t_cot, t_relevant, t_code, t_knowledge
from omegaconf import OmegaConf
import re

class SubQuery:
    def __init__(self, lang: str, engine: VLLM):
        self.lang = lang
        self.full_lang = get_long_language_name(lang)
        self.engine = engine
        self.cfg = OmegaConf.load("config/subquery.yaml")

        # Initialize code retriever
        retriever_name_replace = self.cfg.retriever_name.replace('/', '_')
        code_index_cache_path = f"{self.cfg.index_cache_dir}/code_{lang}_{retriever_name_replace}.index"
        knowledge_index_cache_path = f"{self.cfg.index_cache_dir}/knowledge_{lang}_{retriever_name_replace}.index"
        self.code_retriever = CodeRetriever(
            self.cfg.retriever_name,
            code_index_cache_path,
            lang
        )

        # Initialize knowledge retriever
        self.knowledge_retriever = KnowledgeRetriever(
            self.cfg.retriever_name,
            knowledge_index_cache_path,
            self.cfg.knowledgebase_path,
            lang
        )

    def completions(self, prompts: List[str], stop: List[str]):
        # Load example shots
        e_p, e_cot, e_rel, _, e_res = load_shot(self.lang)
        
        # Generate initial plan and hypothetical code
        prompts_1 = [
            t_problem(e_p) + t_cot(e_cot) + t_code(e_res, self.full_lang) + 
            t_problem(prompt) + "### Solve this problem step by step:"
            for prompt in prompts
        ]
        
        outputs_1 = self.engine.generate(prompts_1, stop=['</code>'])
        plans = extract_content(outputs_1, "thinking")
        hycodes = extract_mdcode(outputs_1)

        # Retrieve similar code and generate final solution
        relevant_codes = self.code_retriever.retrieve(hycodes, top_k=20)
        code_contexts = ['\n\n'.join(codes[:5]) for codes in relevant_codes]

        # Retrieve relevant knowledge
        plan_steps = split_cots(plans)
        knowledge_ctx = []
        for steps in plan_steps:
            k = self.knowledge_retriever.retrieve(steps, top_k=5)
            k = [item['knowledge_entity'] + ' ' + item['intent'] + ' ' + item['content'] + ' ' + item['code_demo'] for item in k]
            knowledge_ctx.append('\n\n'.join(k[:10]))
        # breakpoint()
        # Build final prompts with retrieved code
        final_prompts = []
        for prompt, plan, code_ctx, knowledge_ctx in zip(prompts, plans, code_contexts, knowledge_ctx):
            prompt = (
                f"{t_knowledge(knowledge_ctx)}"
                f"{t_relevant(e_rel)}{t_problem(e_p)}{t_cot(e_cot)}{t_code(e_res, self.full_lang)}"
                f"{t_relevant(code_ctx)}{t_problem(prompt)}{t_cot(plan)}"
                "\n### Final Implementation:\n"
            )
            final_prompts.append(prompt)

        completions = self.engine.generate(final_prompts, stop=['</code>'])
        final_codes = extract_mdcode(completions)
        
        # Construct full completions with plan and retrieved code
        full = [rel_k + rel_c + plan + code for rel_k, rel_c, plan, code in zip(knowledge_ctx, code_contexts, plans, final_codes)]

        return final_codes, full