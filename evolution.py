import llm
import re
import numpy as np
import config

def evo_prompt(prompt1, prompt2): 
    return f"""
Please follow the instructions based on the Genetic Algorithm step-by-step to generate better prompts in Japanese.
Step 1. Exchange some words or sentences in the following prompts and generate two new prompts:
# prompt1: {prompt1}
# prompt2: {prompt2}
Step 2. Modify each prompt generated in Step 1 (add or remove some sentences, or change the expression of some sentences) and generate two final prompts, each of which is bracketed with <prompt> and </prompt>.
"""
 
def create_children(prompt1, prompt2):
    text = llm.generate(evo_prompt(prompt1, prompt2), 1)
    pattern = r'<prompt>(.*?)</prompt>'
    children = re.findall(pattern, text)
    if children:
        if len(children) >= 2:
            return [children[-1], children[-2]]      
        else:
            return [prompt1, prompt2]
    else:
        return [prompt1, prompt2]

def evolve(genotypes, scores):
    new_genotypes = []
    for _ in range(config.genotypes_num//2):
        c1, c2 = select_child(scores)
        children = create_children(genotypes[c1], genotypes[c2])
        new_genotypes.extend(children)
    
    return new_genotypes
    
def select_child(scores):
    weights = [s/sum(scores) for s in scores]
    fertile_genotypes = range(len(scores))
    c1, c2 = np.random.choice(fertile_genotypes, size=2, replace=True, p=weights)
    
    return c1, c2