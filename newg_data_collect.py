import numpy as np
import cvxpy as cp
import networkx as nx
import copy
from collections import deque
import math
import random
import itertools
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from random import randrange
from sklearn.model_selection import train_test_split

from amod import AMOD
from bb_class import BB_node
from bb_class import BB
import datacollector as dc


def number_od_pair(amod_prob, pair):
  o = pair[0]
  d = pair[1]
  od_identifier = o * amod_prob.road.number_of_nodes() + d
  return od_identifier

def powerset(iterable):
  s = list(iterable)
  return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s)+1))


def generate_req_sets(amod_prob, num_sets, num_per_set):
    print("Generating requests...")
    req_subsets = []
    
    for i in range(num_sets):
        n = random.randint(15, num_per_set)
        
        random_sample = random.sample(amod_prob.all_od_pairs, n)
        
        req_subsets.append(random_sample)
    
    return req_subsets


def input_matrices(amod_prob, req_subsets):
    input_data = []
    
    for subset in req_subsets:
        matrix = np.zeros(len(amod_prob.all_od_pairs))
    
        for request in amod_prob.all_od_pairs:
            if request in subset:
                matrix[amod_prob.all_od_pairs.index(request)] = 1

        input_data.append(torch.tensor(torch.from_numpy(matrix), dtype=torch.float))
  
    return input_data


def output_matrices(amod_prob, req_subsets):
    
    output_data = []
    
    for subset in req_subsets:
        print("Current request set:", subset)
        output_matrix = np.zeros(len(amod_prob.all_od_pairs) + 1)
        permutations_table = {}
        
        index_ref = copy.deepcopy(subset)
        
        for req in index_ref:
            node_counts = []
            repeat_shuffles = 2
            
            while repeat_shuffles != 0:
                subset.remove(req)
                random.shuffle(subset)

                subset.insert(0, req)
                amod_prob.requests = subset

                bb_class = BB(amod_prob)
                bb_class.build_tree()
                
                if bb_class.all_ub[0] == math.inf:
                    if bb_class.all_ub.count(bb_class.all_ub[0]) == len(bb_class.all_ub):
                        print("This request first = infeasible with constraints")
                
                elif bb_class.all_lb[0] == math.inf:
                    if bb_class.all_lb.count(bb_class.all_lb[0]) == len(bb_class.all_lb):
                        print("This request first = infeasible no constraints")
                
                else:
                    node_counts.append(bb_class.node_count)
                
                repeat_shuffles -= 1
            
            if node_counts:
                permutations_table[req] = max(node_counts)
        

        if permutations_table:
            print("permutations table:", permutations_table)
            
            best_req_options = [first_req for (first_req, node_count) in permutations_table.items()
                               if node_count == min(permutations_table.values())]

            for req in best_req_options:
                output_matrix[amod_prob.all_od_pairs.index(req)] = 1 / len(best_req_options)

            output_data.append(torch.tensor(torch.from_numpy(output_matrix), dtype=torch.float))
        
        else:
            output_matrix[-1] = 1
            
            output_data.append(output_matrix)
  

    return output_data



def make_train_test_data(amod_prob, num_sets, num_per_set):
    req_subsets = generate_req_sets(amod_prob, num_sets, num_per_set)
    
    input_data = input_matrices(amod_prob, req_subsets)
    output_data = output_matrices(amod_prob, req_subsets)
    
    in_train, in_test, out_train, out_test = train_test_split(input_data, output_data)
    
    return in_train, in_test, out_train, out_test
