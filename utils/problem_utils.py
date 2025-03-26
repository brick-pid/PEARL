"""
get test stuites
"""

import os
import json

base_dir = "multipl_e/prompts/humaneval-{}-reworded.jsonl"
langs = ["jl", "lua", "r", "ml", "rkt"]

def get_problems(lang):
    assert lang in langs, "lang not supported, must be one of {}".format(langs)
    file_path = base_dir.format(lang)
    problems = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            problems.append(data)
    problems.sort(key=lambda x: int(x["name"].split('_')[1]))
    return problems
            

# if __name__ == "__main__":
#     lang = "rkt"
#     breakpoint()
#     problems = get_problems(lang)
#     print(problems)
