from typing import List
from rag.retriever import CodeRetriever
from prompts.gen_prompts import t_problem, t_cot, t_relevant, t_code
from utils.process_utils import extract_content, load_shot, get_long_language_name, extract_mdcode
from engine import VLLM
from omegaconf import OmegaConf

class HyCode2:
    def __init__(self, lang: str, engine: VLLM):
        self.lang = lang
        self.full_lang = get_long_language_name(lang)
        self.engine = engine
        self.cfg = OmegaConf.load("config/subquery.yaml")

        # Initialize code retriever only
        retriever_name_replace = self.cfg.retriever_name.replace('/', '_')
        code_index_cache_path = f"{self.cfg.index_cache_dir}/code_{lang}_{retriever_name_replace}.index"
        self.code_retriever = CodeRetriever(
            self.cfg.retriever_name,
            code_index_cache_path,
            lang
        )

    def completions(self, prompts: List[str], stop: List[str]):
        # Load example shots
        e_p, e_cot, e_rel, _, e_res = load_shot(self.lang)
        
        # Stage 1: Initial generation
        prompts_1 = [
            t_problem(e_p) + t_cot(e_cot) + t_code(e_res, self.full_lang) + t_problem(prompt) + "### Solve this problem step by step:"
            for prompt in prompts
        ]
        # breakpoint()
        outputs_1 = self.engine.generate(prompts_1, stop=['</code>'])
        thinkings = extract_content(outputs_1, "thinking")
        hycodes = extract_mdcode(outputs_1)

        # Stage 2: Code retrieval and final generation
        relevant_codes = self.code_retriever.retrieve(hycodes, top_k=5)
        relevant_codes = ['\n\n'.join(codes) for codes in relevant_codes]

        # Build final prompts with retrieved code
        prompts_2 = []
        for problem, cot, rel in zip(prompts, thinkings, relevant_codes):
            prompt = (
                f"{t_relevant(e_rel)}{t_problem(e_p)}{t_cot(e_cot)}{t_code(e_res, self.full_lang)}"
                f"{t_relevant(rel)}{t_problem(problem)}{t_cot(cot)}" + "\n### Code Implementation:\n"
            )
            prompts_2.append(prompt)

        completions_2 = self.engine.generate(prompts_2, stop=['</code>'])
        codes = extract_mdcode(completions_2)
        full = [t + rel for t, rel in zip(thinkings, relevant_codes)]

        return codes, full