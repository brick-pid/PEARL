langs=("rkt" "lua" "r" "jl" "ml")
# method="subquery"
# method="pearl_wo_cot"
method="pearl"
name="unsloth/Meta-Llama-3.1-8B-Instruct"
# name="unsloth/Llama-3.2-1B-Instruct"
for lang in "${langs[@]}"; do
    python inference.py lang="$lang" name="$name" method="$method"
done

# if the completion is complete function, the the --full-function command; otherwise, remove it
python multipl_e/evaluation/src/main.py --dir "multiple_results/${method}" --output-dir "multiple_results/${method}" --full-function --recursive
# python multipl_e/evaluation/src/main.py --dir "multiple_results/${method}" --output-dir "multiple_results/${method}"  --recursive
python multipl_e/pass_k.py "multiple_results/${method}/*"