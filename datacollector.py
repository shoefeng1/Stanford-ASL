import numpy as np
import cvxpy as cp
import networkx as nx
import copy
from collections import deque
import math
import random
import itertools
import torch

from amod import AMOD
from bb_class import BB_node
from bb_class import BB


def number_od_pair(amod_prob, pair):
  o = pair[0]
  d = pair[1]
  od_identifier = o * amod_prob.road.number_of_nodes() + d
  return od_identifier

def powerset(iterable):
  s = list(iterable)
  return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s)+1))


def input_matrices(request_set):
  req_subsets = list(powerset(request_set))
  input_data = []

  for subset in req_subsets:
    matrix = np.zeros(len(request_set))
    
    for request in request_set:
      if request in subset:
        matrix[request_set.index(request)] = 1

    input_data.append(torch.tensor(torch.from_numpy(matrix), dtype=torch.float))
  
  return input_data


def make_permutations_table(amod_prob):
  all_permutations = list(itertools.permutations(amod_prob.requests))
  permutations_table = {}

  for permutation in all_permutations:
    
    amod_prob.requests = permutation

    bb_class = BB(amod_prob)
    bb_class.build_tree()

    permutation_id_list = [number_od_pair(amod_prob, request) for request in permutation]
    permutation_by_id = tuple(permutation_id_list)

    permutations_table[permutation_by_id] = bb_class.node_count
  
  return permutations_table


def output_matrix(request_set, permutations_table):
  matrix = np.zeros(len(request_set))
  best_req_options = []

  minval = min(permutations_table.values())

  for (permutation, nodes_made) in permutations_table.items():
    if permutation[0] not in best_req_options:
      if nodes_made == minval:
        best_req_options.append(permutation[0])
  
  for req_option in best_req_options:
    matrix[request_set.index(req_option)] = 1 / len(best_req_options)
  
  return matrix


def all_output_matrices(amod_prob):
  all_requests = copy.deepcopy(amod_prob.requests)
  all_requests_by_id = [number_od_pair(amod_prob, request) for request in all_requests]
  output_data = []
  all_subsets = list(powerset(amod_prob.requests))

  for subset in all_subsets:
    amod_prob.requests = subset
    permutation_table = make_permutations_table(amod_prob)

    output = output_matrix(all_requests_by_id, permutation_table)

    output_data.append(torch.tensor(torch.from_numpy(output), dtype=torch.float))
  
  return output_data


def random_data_indices(num_requests):
  num_subsets = 2 ** num_requests - 1
  req_indices = [k for k in range(num_subsets)]
  testing_set = np.random.choice(req_indices, round(num_subsets / 5), replace=False).tolist()
  return testing_set


def create_input_data(amod_prob, testing_indices):
  request_set = [number_od_pair(amod_prob, pair) for pair in amod_prob.requests]
  list_inputs = input_matrices(request_set)
  testing_set = []
  
  for index in testing_indices:
    tester = list_inputs.pop(index)
    testing_set.append(tester)
  
  training_set = list_inputs
  return training_set, testing_set


def create_output_data(amod_prob, testing_indices):
  list_outputs = all_output_matrices(amod_prob)
  testing_set = []

  for index in testing_indices:
    tester = list_outputs.pop(index)
    testing_set.append(tester)

  training_set = list_outputs
  return training_set, testing_set
