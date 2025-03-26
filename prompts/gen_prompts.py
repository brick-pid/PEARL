def t_problem(problem):
    return f"""
### Task Description: 
{problem}
"""

def t_cot(cot):
    return f"""
### Solve this problem step by step: 
<thinking>
{cot}
</thinking>
"""

def t_knowledge(knowledge):
    return f"""
### Relevant Knowledge:
<knowledge>
{knowledge}
</knowledge>
"""

def t_relevant(relevant):
    return f"""
### Relevant Code Snippets:
<relevant>
{relevant}
</relevant>
"""

def t_code(code, lang):
    return f"""
### Code Implementation: 
<code>
```{lang}
{code}
```
</code>
"""