import time
from typing import List
import wandb

class GenerationLogger:
    def __init__(self, project_name: str, run_name: str):
        self.run = wandb.init(
            project=project_name,
            name=run_name
        )
        
        # 只保留生成结果表格
        self.generation_table = wandb.Table(columns=[
            "problem_name",      # HumanEval 问题名称
            "problem",          # 原始问题描述
            "generation_id",    # 生成ID
            "prompt",           # 完整的提示词
            "full_completion",  # 完整的生成结果
            "code",   # 提取出的代码
        ])

    def log_generations(self, 
                       problem_name: str, 
                       original_problem: str,
                       prompts: List[str],
                       full_completions: List[str],
                       codes: List[str]):
        """记录单个问题的所有生成结果"""
        for i in range(len(codes)):
            self.generation_table.add_data(
                problem_name,
                original_problem,
                i + 1,
                prompts[i],
                full_completions[i],
                codes[i]
            )
        
        # 每40个样本记录一次
        if len(self.generation_table.data) % 40 == 0:
            self.log_tables()

    def log_tables(self):
        """记录当前表格状态"""
        self.run.log({
            "generation_details": self.generation_table,
        })

    def finish(self):
        """完成记录，保存最终结果"""
        artifact = wandb.Artifact(
            name=f"generation_results_{int(time.time())}", 
            type="generation_results"
        )
        
        artifact.add(self.generation_table, "generation_details")
        
        self.run.log_artifact(artifact)
        self.log_tables()
        self.run.finish()
