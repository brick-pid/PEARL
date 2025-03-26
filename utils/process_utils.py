import re
from typing import List
import importlib

def split_cots(cots: List[str]) -> List[List[str]]:
    """
    Split the cot into thinking steps
    each step should begin with a number and a dot
    """
    cots_steps = []
    for cot in cots:
        # Split into lines and filter out empty lines
        lines = [line.strip() for line in cot.split('\n') if line.strip()]
        
        # Initialize variables for current step
        current_step = []
        steps = []
        
        for line in lines:
            # Check if line starts with number and dot (e.g. "1.", "2.", etc)
            if re.match(r'^\d+\.', line):
                # If we have collected lines for a previous step, add them
                if current_step:
                    steps.append('\n'.join(current_step))
                    current_step = []
                current_step.append(line)
            else:
                # Add line to current step if we have started one
                if current_step:
                    current_step.append(line)
        
        # Add the last step if there is one
        if current_step:
            steps.append('\n'.join(current_step))
            
        # Add list of steps for this cot
        cots_steps.append(steps)
    
    return cots_steps

def extract_content(completions: List[str], tag: str) -> List[str]:
    """
    提取 <tag>...</tag> 之间的内容，匹配所有可能的结果并取最后一个
    如果 <tag> 没有对应的 </tag>，则会自动添加 </tag> 到 completion 的末尾
    Args:
        completions (List[str]): A list of strings containing HTML-like content.
        tag (str): The tag name to search for within the strings. e.g. "code" for <code>...</code>
    Returns:
        List[str]: A list of extracted content. If the tag is not found in a string,
                   the corresponding entry will indicate that no content was found.
    """
    if isinstance(completions, str):
        completions = [completions]

    # strip the completions
    completions = [completion.strip() for completion in completions]

    contents = []
    for completion in completions:
        pattern = re.compile(f'<{tag}>(.*?)</{tag}>', re.DOTALL)
        matches = pattern.findall(completion)
        
        if matches:
            content = matches[-1].strip()
        else:
            content = f"NOFOUND"

        contents.append(content)
    return contents


def extract_mdcode(completions: List[str], lang: str) -> List[str]:
    """
    extract the code block in the markdown content
    匹配 ```xxx_lang ``` 形式的代码块，并返回最后一个代码块的内容
    Now with improved pattern matching for various markdown code block formats
    """
    codes = []
    for completion in completions:
        if not isinstance(completion, str):
            codes.append("Invalid completion type, expected string")
            continue
            
        # More robust regex pattern that handles different variations of markdown code blocks
        # This matches:
        # 1. Standard ```lang\n code \n```
        # 2. Code blocks with language specified as ```lang
        # 3. Code blocks with language specified as ```python or other languages if lang is a suffix
        patterns = [
            # Exact language match
            re.compile(r'```' + re.escape(lang) + r'\s*\n(.*?)\n\s*```', re.DOTALL),
            # Language might be a substring of the specified language (e.g., 'python' for 'py')
            re.compile(r'```.*?' + re.escape(lang) + r'.*?\s*\n(.*?)\n\s*```', re.DOTALL),
            # Generic code block as fallback
            re.compile(r'```.*?\n(.*?)\n\s*```', re.DOTALL)
        ]
        
        code_found = False
        for pattern in patterns:
            code_matches = pattern.findall(completion)
            if code_matches:
                # If found code blocks, take the last one
                codes.append(code_matches[-1].strip())
                code_found = True
                break
        
        if not code_found:
            # If no code blocks found with any pattern, indicate that
            codes.append("No markdown code block found")
    
    return codes

def dict2str(d: dict) -> str:
    """
    Convert a dictionary to a string
    """
    return '\n'.join([f"{k}: {v}" for k, v in d.items()])

def load_shot(lang: str):
    """
    return: `problem`, `cot`, `relevant`, `knowledge`, `result`
    """
    if lang in ["rkt", "ml", "jl", "lua", "r"]:
        module = importlib.import_module(f"prompts.{lang}_example")
        return module.problem, module.cot, module.relevant, module.knowledge, module.result
    else:
        raise ValueError(f"Invalid language: {lang}")

def get_long_language_name(lang: str):
    """
    convert short language to long language
    """
    if lang in ["racket", "ocaml", "julia", "lua", "r"]:
        return lang
    if lang == "rkt":
        return "racket"
    elif lang == "ml":
        return "ocaml"
    elif lang == "jl":
        return "julia"
    else:
        return lang

def do_name_override(name):
    return name.replace("/", "_").replace("-", "_")