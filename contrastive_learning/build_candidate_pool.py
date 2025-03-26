"""
Build a candidate pool with the subquery approach.
"""
import json
from engine import VLLM
from vllm import SamplingParams
from rag.retriever import KnowledgeRetriever, CodeRetriever
import hydra
from omegaconf import DictConfig
from utils.process_utils import extract_content, load_shot, get_long_language_name, extract_mdcode, split_cots
from prompts.gen_prompts import t_problem, t_cot, t_relevant, t_code, t_knowledge
from datasets import load_dataset
import os
from tqdm import tqdm
BASE_DIR = os.environ['BASE_DIR']


@hydra.main(config_path=f"{BASE_DIR}/config", config_name="contrastive", version_base="1.3")
def main(cfg: DictConfig):
    short_lang = cfg.short_lang
    full_lang = cfg.full_lang
    
    dataset = load_dataset('json', data_files=cfg.train_dataset, split='train')

    sampling_params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=2048, min_tokens=8)
    model = VLLM(cfg.model_name, None, None, None, sampling_params, cfg.num_gpus)

    retriever_name_replace = cfg.retriever_model_name.replace('/', '_')
    code_index_cache_path = f"{cfg.index_cache_dir}/code_{short_lang}_{retriever_name_replace}.index"
    knowledge_index_cache_path = f"{cfg.index_cache_dir}/knowledge_{short_lang}_{retriever_name_replace}.index"
    
    code_retriever = CodeRetriever(
        cfg.retriever_model_name,
        code_index_cache_path,
        short_lang
    )
    knowledge_retriever = KnowledgeRetriever(
        cfg.retriever_model_name,
        knowledge_index_cache_path,
        cfg.knowledge_path,
        short_lang
    )

    # Load example shots
    e_p, e_cot, _, _, e_res = load_shot(short_lang)
    
    with open(cfg.candidate_pool_path, 'w') as f:
        for example in tqdm(dataset, desc="Generating candidates"):
            # Generate initial plan and hypothetical code
            prompt_1 = (
                t_problem(e_p) + t_cot(e_cot) + t_code(e_res, full_lang) + 
                t_problem(example['prefix']) + "### Solve this problem step by step:"
            )
            
            output_1 = model.generate(prompt_1, stop=['</code>'])
            plan = extract_content(output_1, "thinking")
            hycode = extract_mdcode(output_1)

            # Retrieve similar code and knowledge
            code_candidates = code_retriever.retrieve(hycode, top_k=30)[0]
            
            plan_steps = split_cots(plan)[0]
            knowledge_candidates = knowledge_retriever.retrieve(plan_steps, top_k=10)[:30]
            knowledge_candidates = [item['knowledge_entity'] + ' ' + item['intent'] + ' ' + item['content'] + ' ' + item['code_demo'] for item in knowledge_candidates]
            
            # save raw results for latter processing
            f.write(json.dumps({
                "content": example['content'],
                "prefix": example['prefix'],
                "suffix": example['suffix'],
                "code_candidates": code_candidates,
                "knowledge_candidates": knowledge_candidates
            }) + '\n')

if __name__ == "__main__":
    main()