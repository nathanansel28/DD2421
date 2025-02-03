from typing import Union, Literal, List, Tuple
from scipy.stats import uniform, norm
import dtree as d
import numpy as np
import matplotlib.pyplot as plt
import math 
import random


def calculate_entropy(
    data: List[int]
): 
    mydict = {}
    for item in data: 
        if item in mydict: 
            mydict[item] += 1
        else:
            mydict[item] = 1

    if len(mydict) == 0: 
        return 0.0 
    
    entropy = 0
    n = len(data)
    for item in mydict: 
        p_i = mydict[item]/n
        entropy += -1 * p_i * math.log(p_i, 2)

    return entropy


def generate_samples(
    type: Literal['uniform', 'non_uniform'],
    n: int, 
    mean: float = 0, 
    std: float = 10
) -> List[int]: 
    assert type in ['uniform', 'non_uniform']

    if type == 'uniform': 
        return list(range(1,n+1))
    elif type == 'non_uniform': 
        random_samples = np.random.normal(mean, std, size=n)
        return [round(num) for num in random_samples]


def partition(data, fraction, use_seed=False, seed_value=12): 
    if use_seed:
        random.seed(seed_value)
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


def prune_tree(initial_tree, validation_data):
    best_trees = [initial_tree]
    best_score = d.check(initial_tree, validation_data)
    
    while True:
        new_best_trees = []
        for tree in best_trees:
            pruned_trees = list(d.allPruned(tree))
            if not pruned_trees:
                continue  # No more pruning possible
            
            current_best_score = best_score
            candidates = []
            for pruned_tree in pruned_trees:
                score = d.check(pruned_tree, validation_data)
                if score > current_best_score:
                    candidates = [pruned_tree]
                    current_best_score = score
                elif score == current_best_score:
                    candidates.append(pruned_tree)
            
            if candidates:
                new_best_trees = candidates
                best_score = current_best_score
        
        if not new_best_trees:
            break  # Stop pruning when no improvement is found
        
        best_trees = new_best_trees
    
    return best_trees[0]

