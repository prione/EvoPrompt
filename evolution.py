import llm
import re
import numpy as np
import config

def evolve(genotypes):
    new_genotypes = []
    for _ in range(config.genotypes_num//2):
        paprent1, paprent2 = select_parents(genotypes)
        children = create_children(paprent1, paprent2)
        new_genotypes.extend(children)
    return new_genotypes

def select_parents(genotypes):
    alpha = 0.3
    fertile_genotypes = np.arange(0, len(genotypes))
    weights = np.exp(-alpha * fertile_genotypes)
    weights /= weights.sum() 
    p1, p2 = np.random.choice(fertile_genotypes, size=2, replace=True, p=weights)
    return genotypes[p1], genotypes[p2]

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

def evo_prompt(prompt1, prompt2): 
    return f"""
Please follow the instructions based on the Genetic Algorithm step-by-step to generate better prompts in Japanese.
Step 1. Exchange some words or sentences in the following prompts and generate two new prompts:
#prompt1: {prompt1}
#prompt2: {prompt2}
Step 2. Modify each prompt generated in Step 1 (add or remove some sentences, or change the expression of some sentences) and generate two final prompts, each of which is bracketed with <prompt> and </prompt>.
"""
