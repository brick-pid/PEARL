"""
This script produces completions for roughly any AutoModelForCausalLM.
Use Instruct Mode, will generate full function code, including the function signature.
"""
import os
from multipl_e.multipl_e.completions import make_main
from methods.raw.raw import RAW
from methods.fewshot.fewshot import FewShot
# from methods.ret_fewshot.ret_fewshot import RetFewShot
# from methods.hycode.hycode import HyCode
# from methods.hycode.hycode2 import HyCode2
# from methods.cot.cot import CoT
# from methods.subquery.subquery import SubQuery
from methods.pearl.pearl import PEARL
from methods.pearl.pearl_chat import PEARL_CHAT
# from methods.pearl.pearl_wo_cot import PEARL_wo_cot
import hydra
from omegaconf import DictConfig
from engine import VLLM, OpenAIEngine, OpenAIChatEngine
from vllm import SamplingParams
def do_name_override(model_name):
    """
    Applies the --name-override flag, or uses the model name, correcting / and - which the rest of
    the toolchain does not like.
    """
    return model_name.replace("/", "_").replace("-", "_")

@hydra.main(config_path="config", config_name="inference", version_base="1.3")
def main(cfg: DictConfig):

    cfg.output_dir_prefix = f"{cfg.output_dir_prefix}/{cfg.name.split('/')[-1]}"
    sampling_params = SamplingParams(temperature=cfg.temperature, top_p=cfg.top_p, max_tokens=cfg.max_tokens, min_tokens=8)
    num_gpus = int(os.getenv("NUM_GPUS"))
    if cfg.use_engine == "vllm":
        model = VLLM(cfg.name, None, None, None, sampling_params, num_gpus)
    elif cfg.use_engine == "openai":
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        model = OpenAIEngine(cfg.name, base_url, api_key)
    elif cfg.use_engine == "openai_chat":
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        model = OpenAIChatEngine(cfg.name, base_url, api_key)
    else:
        raise ValueError(f"Unknown engine: {cfg.use_engine}")

    if cfg.method == "raw":
        method = RAW(cfg.lang, model)
    # elif cfg.method == "cot":
    #     method = CoT(cfg.lang, model)
    # elif cfg.method == "fewshot":
    #     method = FewShot(cfg.lang, model)
    # elif cfg.method == "ret_fewshot":
    #     method = RetFewShot(cfg.lang, model)
    # elif cfg.method == "hycode":
    #     method = HyCode(cfg.lang, model)
    # elif cfg.method == "hycode2":
    #     method = HyCode2(cfg.lang, model)
    # elif cfg.method == "subquery":
    #     method = SubQuery(cfg.lang, model)
    elif cfg.method == "pearl":
        method = PEARL(cfg.lang, model)
    elif cfg.method == "pearl_chat":
        method = PEARL_CHAT(cfg.lang, model)
    # elif cfg.method == "pearl_wo_cot":
    #     method = PEARL_wo_cot(cfg.lang, model)
    else:
        raise ValueError(f"Unknown method: {cfg.method}")
    
    cfg.name = do_name_override(cfg.name)
    print(cfg)
    make_main(cfg, cfg.name, method.completions)

if __name__ == "__main__":
    main()
