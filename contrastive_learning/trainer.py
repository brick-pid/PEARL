import torch
import torch.nn as nn
from transformers import Trainer
import random
from dataclasses import dataclass
from utils.process_utils import extract_content, load_shot, extract_mdcode, split_cots, get_long_language_name
from prompts.gen_prompts import t_problem, t_cot, t_code


class StringCollator:
    """
    Just collate the strings into a list, without any additional processing.
    Input: batch of examples where each example contains string data
    Output: dictionary with lists of strings
    """
    def __call__(self, examples):
        collated = {}
        keys = examples[0].keys()
        for key in keys:
            collated[key] = [example[key] for example in examples]            
        return collated


class ContrastiveTrainer(Trainer):
    def __init__(self, engine, retr_tokenizer, lm_tokenizer, code_retriever=None, know_retriever=None, topk=10, bottomk=10, lang=None, temperature=0.07, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = engine
        self.retr_tokenizer = retr_tokenizer
        self.lm_tokenizer = lm_tokenizer
        self.code_retriever = code_retriever
        self.know_retriever = know_retriever
        self.topk = topk
        self.bottomk = bottomk
        self.lang = lang
        self.full_lang = get_long_language_name(lang)
        self.temperature = temperature

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step on a batch of inputs.
        inputs: {'content': List[str], 'prefix': List[str], 'suffix': List[str]}
        """
        model.train()
        prefix, suffix = inputs["prefix"], inputs["suffix"]
        batch_size = len(prefix)
        
        # Get candidates for each query
        code_psgs, know_psgs = [], []
        
        for i in range(batch_size):
            p = prefix[i]
            s = suffix[i]
            code_candidates, know_candidates = self.candidate_builder(model, p, s)
            
            # Process code candidates
            code_cand_ordered = self.engine.score_candidates(p, s, code_candidates, self.lm_tokenizer)
            code_cand_texts = [o['text'] for o in code_cand_ordered]
            
            # Assert we have enough candidates
            assert len(code_cand_texts) >= self.topk + self.bottomk, "Not enough code candidates"
            
            # psgs: [pos, neg, neg, ...], for each row, first is positive, the rest are negatives
            code_psgs.append(random.choice(code_cand_texts[:self.topk]))
            code_psgs.extend(code_cand_texts[-self.bottomk:])
            
            # Process knowledge candidates
            if know_candidates:
                know_cand_ordered = self.engine.score_candidates(p, s, know_candidates, self.lm_tokenizer)
                know_cand_texts = [o['text'] for o in know_cand_ordered]
                
                # Assert we have enough candidates
                assert len(know_cand_texts) >= self.topk + self.bottomk, "Not enough knowledge candidates"

                know_psgs.append(random.choice(know_cand_texts[:self.topk]))
                know_psgs.extend(know_cand_texts[-self.bottomk:])
            else:
                # If no knowledge candidates, use code positive as a fallback
                know_psgs.append(code_psgs[0])
                know_psgs.extend(code_psgs[1:self.bottomk])
        
        # Tokenize queries and passages
        q_tokens = self.retr_tokenizer(prefix, padding=True, truncation=True, 
                                      max_length=512, return_tensors="pt")
        
        code_tokens = self.retr_tokenizer(code_psgs, padding=True, truncation=True, max_length=512, return_tensors="pt")
        
        know_tokens = self.retr_tokenizer(know_psgs, padding=True, truncation=True, max_length=512, return_tensors="pt")
        
        # Move to device
        device = model.module.device if hasattr(model, 'module') else model.device
        q_tokens = {k: v.to(device) for k, v in q_tokens.items()}
        code_tokens = {k: v.to(device) for k, v in code_tokens.items()}
        know_tokens = {k: v.to(device) for k, v in know_tokens.items()}
        
        with self.compute_loss_context_manager():
            # Compute embeddings
            q_reps = model(q_tokens)
            code_reps = model(code_tokens)
            know_reps = model(know_tokens)
            
            # Compute combined loss
            loss = self.compute_dual_retrieval_loss(q_reps, code_reps, know_reps)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)
        return loss.detach()

    def compute_dual_retrieval_loss(self, q_reps, code_reps, know_reps):
        """
        Compute separate retrieval losses for code and knowledge retrieval
        
        Args:
            q_reps: query representations [batch_size, dim]
            code_reps: code representations [batch_size * (1 + negs_per_query), dim]
            know_reps: knowledge representations [batch_size * (1 + negs_per_query), dim]
            batch_size: number of queries
        """
        code_loss = self.compute_contrastive_loss(q_reps, code_reps, self.temperature)
        know_loss = self.compute_contrastive_loss(q_reps, know_reps, self.temperature)
        return (code_loss + know_loss) / 2

    def compute_contrastive_loss(self, q_reps, psg_reps, temperature):
        """
        Compute contrastive loss for a batch
        
        Args:
            q_reps: query representations [batch_size, dim]
            psg_reps: positive passage representations [batch_size * group_size, dim]
        """
        batch_size, group_size = q_reps.size(0), psg_reps.size(0) // q_reps.size(0)
        # [batch_size, batch_size * (1 + negs_per_query)], the 0, 1*group_size, 2*group_size, ... are the positive scores
        scores = (q_reps @ psg_reps.T) / temperature
        labels = torch.arange(batch_size, device=scores.device)
        labels = labels * group_size # [batch_size]
        loss = nn.CrossEntropyLoss()(scores, labels)
        return loss

    def candidate_builder(self, model, query, suffix):
        """
        Build candidates for one query by retrieving similar code and knowledge.
        
        Args:
            model (torch.nn.Module): The model to use for encoding
            query (str): The query text (prefix of the problem)
            suffix (str): The suffix of the problem
        Returns:
            list: Combined list of code and knowledge candidates
        """
        assert isinstance(query, str)
        assert isinstance(suffix, str)
        
        # 推理模式
        model.eval()
        device = model.module.device if hasattr(model, 'module') else model.device

        # Generate initial plan and hypothetical code using the model
        e_p, e_cot, _, _, e_res = load_shot(self.lang)
        prompt_1 = (
            "### Instruction: use <thinking> ... </thinking> to enclose the thinking process, use <code> ... </code> to enclose the code.\n" +
            t_problem(e_p) + t_cot(e_cot) + t_code(e_res, self.full_lang) + 
            t_problem(query + suffix) + "\n### Solve this problem step by step:"
        )
        output_1 = self.engine.generate(prompt_1, stop=['</code>'])
        
        plan = extract_content(output_1, "thinking")
        if plan[0] != "NOFOUND":
            plan_steps = split_cots(plan)[0]

            # convert to np array
            plan_steps_tokens = self.retr_tokenizer(plan_steps, padding=True, truncation=True, max_length=512, return_tensors="pt")
            plan_steps_tokens = {k: v.to(device) for k, v in plan_steps_tokens.items()}
            with torch.no_grad():
                plan_steps_repr = model(plan_steps_tokens).cpu().numpy()

            knowledge_candidates = self.know_retriever.retrieve_results(plan_steps_repr, top_k=12)
            knowledge_candidates = [item for sublist in knowledge_candidates for item in sublist]
        else:
            knowledge_candidates = []

        # Retrieve similar code using hypothetical code
        # convert to np array
        query_tokens = self.retr_tokenizer([query + suffix], padding=True, truncation=True, max_length=512, return_tensors="pt")
        query_tokens = {k: v.to(device) for k, v in query_tokens.items()}
        with torch.no_grad():
            query_repr = model(query_tokens).cpu().numpy()
        code_candidates = self.code_retriever.retrieve_results(query_repr, top_k=30)[0]
        
        # check if the prefix is in the code_candidates, if so, remove it
        code_candidates = [c for c in code_candidates if not c.startswith(query)]
        
        # 恢复到训练模式
        model.train()
        return code_candidates, knowledge_candidates

