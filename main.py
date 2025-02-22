import llm
import re
from evaluation import evaluate
from evolution import evolve
from individual import individual
import systemprompt
import config
import dill

def run(genotypes:list, expected:str, generation:int):
    result = [] 
    
    for gen in range(generation):
        phenotypes = []
        individuals = []
        for g in genotypes:
            phenotypes.append(llm.generate(g,0))
        
        genotypes, phenotypes, scores = evaluate(genotypes, phenotypes, expected)
        print(f"{gen}th best score: {scores[0]}")
        print(f"{gen}th best individual: {genotypes[0]}", end="\n\n")

        for prompt, answer, score in zip(genotypes, phenotypes, scores):
            individuals.append(individual(prompt, answer, score))
        result.append(individuals)
        save(result)
        
        genotypes = evolve(genotypes, scores)
    
    return result
    
def save(result):
    with open(f"{config.save_dir}/result.pkl", "wb") as f:
        dill.dump(result, f)
 
if __name__ == "__main__":
    result = run(config.initial_genotypes, config.expected_answer, config.generation)
    
