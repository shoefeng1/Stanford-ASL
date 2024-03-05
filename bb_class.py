import numpy as np
import cvxpy as cp
import networkx as nx
import copy
from collections import deque
import math
import random

from amod import AMOD


class BB_node:

  def __init__(self, index, req_path, parent_key):
    self.index = index
    self.__req_path = req_path
    self.parent = parent_key
    if req_path is not None:
      self.__request = req_path[0], req_path[-1]
    self.obj_value = None
    self.children = []

  def add_req_path(self, req_path):
    self.__req_path = req_path
    self.__request = req_path[0], req_path[-1]

  def get_req_path(self):
    return self.__req_path

  def get_request(self):
    return self.__request


class BB:

  def __init__(self, amod_prob):
    self.amod_prob = amod_prob
    self.all_bb_nodes = {}
    self.open_nodes = deque()
    self.lowest_ub = math.inf
    self.node_count = 0
    self.all_ub = []
    self.all_lb = []


  def find_ancestors(self, node):
    k = node.parent
    ancestors = [k]
    considering = deque()
    considering.append(k)

    while considering:
      current_node = considering.popleft()
      if current_node in self.all_bb_nodes:
        k_parent = self.all_bb_nodes[current_node].parent
        ancestors.append(k_parent)
        k = k_parent
        considering.append(k)
    return ancestors


  def find_ancestors_recursively(self, node):
    if node.parent is None:
      return [node.index]
    else:
      ancestors_of_parent = self.find_ancestors_recursively(self.all_bb_nodes[node.parent])
      ancestors_of_parent.append(node.index)
      return ancestors_of_parent
  
  def adjust_capacities(self, caps_dict, paths):
    for path in paths:
      for k in range(len(path) - 1):
        if self.amod_prob.road.has_edge(path[k], path[k+1]):
          if (path[k], path[k+1]) in caps_dict.keys():
            caps_dict[(path[k], path[k+1])] -= 1
          elif (path[k+1], path[k]) in caps_dict.keys():
            caps_dict[(path[k+1], path[k])] -= 1
          else:
            raise Error('Key does not exist')
        else:
          raise Error('Edge does not exist')
      
    return caps_dict


  def adjust_caps_ancestors(self, node, ancestors = None):
    if node is None and ancestors is None:
      capacities_dict = copy.deepcopy(self.amod_prob.capacity)
      return capacities_dict
    elif node is not None:
      ancestors_of_node = self.find_ancestors_recursively(node)
    else:
      ancestors_of_node = ancestors
    
    capacities_dict = copy.deepcopy(self.amod_prob.capacity)
    req_paths = [self.all_bb_nodes[ancestor].req_path for ancestor in ancestors_of_node]
    capacities_dict = self.adjust_capacities(capacities_dict, req_paths)
    
    return capacities_dict
  

  def ancestors_path_value(self, node):
    node_value = self.amod_prob.evaluate_objective([node.get_req_path()])
    if node.parent is None:
      node.obj_value = node_value
      return node.obj_value
    else:
      if self.all_bb_nodes[node.parent].obj_value is not None:
        node.obj_value = node_value + self.all_bb_nodes[node.parent].obj_value
      else:
        node.obj_value = node_value + self.ancestors_path_value(self.all_bb_nodes[node.parent])
      return node.obj_value


  def open_node(self, request_id, parent):
    request = self.amod_prob.requests[request_id]
    
    if parent is None:
      capacity = copy.deepcopy(self.amod_prob.capacity)
      requests = copy.deepcopy(self.amod_prob.requests)
    else:
      parent_node = self.all_bb_nodes[parent]
      capacity = parent_node.remaining_capacity
      requests = [self.amod_prob.requests[r] for r in parent_node.reqs_to_route]

    F, obj = self.amod_prob.LP_relaxation(requests, capacity)

    if obj == math.inf:
      print("obj was infinity and exited")
      path = None
      
      self.all_bb_nodes[self.node_count] = BB_node(self.node_count, path, parent)
      child_node = self.all_bb_nodes[self.node_count]
      self.node_count += 1
      
      child_node.upper_bound = obj
      self.all_ub.append(child_node.upper_bound)
      
      child_node.lower_bound = obj
      self.all_lb.append(child_node.lower_bound)
      
      child_node.children = None
      return

    else:

      all_round_paths = self.amod_prob.all_round_paths(F, request, requests)

      for path in all_round_paths:
        self.all_bb_nodes[self.node_count] = BB_node(self.node_count, path, parent)
        child_node = self.all_bb_nodes[self.node_count]
        self.node_count += 1

        child_node.add_req_path(path)

        if parent is None:
          child_node.reqs_to_route = [k for k in range(len(self.amod_prob.requests))]
          child_node.reqs_to_route.remove(request_id)

          capacity = copy.deepcopy(self.amod_prob.capacity)
          child_node.remaining_capacity = self.adjust_capacities(capacity, [path])
        
        else:
          child_node.reqs_to_route = copy.deepcopy(self.all_bb_nodes[parent].reqs_to_route)
          child_node.reqs_to_route.remove(request_id)

          self.all_bb_nodes[parent].children.append(self.node_count - 1)

          child_node.remaining_capacity = copy.deepcopy(self.all_bb_nodes[parent].remaining_capacity)
          child_node.remaining_capacity = self.adjust_capacities(child_node.remaining_capacity, [path])

        lower_bound = self.compute_lower_bound(child_node)
        self.all_lb.append(lower_bound)
        
        upper_bound = self.compute_upper_bound(child_node)
        self.all_ub.append(upper_bound)

        if lower_bound == math.inf:
          child_node.children = None
        elif math.isclose(lower_bound, upper_bound):
          self.lb_is_ub(child_node.index)
        elif lower_bound > self.lowest_ub + 0.001:
          child_node.children = None
        else:
          self.open_nodes.append(child_node)


  def compute_upper_bound(self, node):
    if hasattr(node, 'reqs_to_route'):
      remaining_reqs = [self.amod_prob.requests[r] for r in node.reqs_to_route]
    else:
      remaining_reqs = copy.deepcopy(self.amod_prob.requests)
      ancestors_of_parent = self.find_ancestors_recursively(node)
      for ancestor in ancestors_of_parent:
        remaining_reqs.remove(self.all_bb_nodes[ancestor].request)
    if len(remaining_reqs) == 0:
      node.upper_bound = self.ancestors_path_value(node)
      return self.ancestors_path_value(node)

    if hasattr(node, 'remaining_capacity'):
      caps_dict = copy.deepcopy(node.remaining_capacity)
    else:
      caps_dict = self.adjust_caps_ancestors(node)
      node.remaining_capacity = copy.deepcopy(caps_dict)
      
    rounded_paths = self.amod_prob.upper_bound_routes(remaining_reqs, caps_dict)

    if rounded_paths == math.inf:
      node.upper_bound = math.inf
      return math.inf
    else:
      ub_obj_value = self.amod_prob.evaluate_objective(rounded_paths)
    
    upper_bound = ub_obj_value + self.ancestors_path_value(node)

    node.upper_bound = upper_bound
    self.lowest_ub = min(self.lowest_ub, upper_bound)
    return upper_bound


  def compute_lower_bound(self, node):
    if hasattr(node, 'remaining_capacity'):
      caps_dict = node.remaining_capacity
    else:
      caps_dict = self.adjust_caps_ancestors(node)
    
    if hasattr(node, 'reqs_to_route'):
      remaining_reqs = [self.amod_prob.requests[r] for r in node.reqs_to_route]
    else:
      remaining_reqs = copy.deepcopy(self.amod_prob.requests)
      ancestors_of_parent = self.find_ancestors_recursively(node)
      for ancestor in ancestors_of_parent:
        remaining_reqs.remove(self.all_bb_nodes[ancestor].request)

    if len(remaining_reqs) == 0:
      lb_obj_value = 0
    else:
      _, lb_obj_value = self.amod_prob.LP_relaxation(overrideReqs = remaining_reqs, overrideCaps = caps_dict)

    lower_bound = lb_obj_value + self.ancestors_path_value(node)

    node.lower_bound = lower_bound
    return lower_bound


  def get_next_request(self, node):
    if node is None:
      return 0
    elif len(node.reqs_to_route) == 0:
      return None
    else:
      return node.reqs_to_route[0]
  
  
  def get_random_request(self, node):
    if node is None:
      random_req_id = random.choice(range(len(self.amod_prob.requests)))
    elif len(node.reqs_to_route) == 0:
      return None
    else:
      random_req_id = random.choice(node.reqs_to_route)
    
    return random_req_id


  def build_tree(self):
    if self.node_count > 0:
      print("Warning: building tree on top of existing tree")

    req_id = self.get_next_request(None)

    self.open_node(req_id, None)

    while self.open_nodes:
      node = self.open_nodes.popleft()
      next_req_id = self.get_next_request(node)

      if next_req_id is None:
        break
      
      self.open_node(next_req_id, node.index)


  def build_tree_random(self):
    random_req_id = self.get_random_request(None)

    self.open_node(random_req_id, None)

    while self.open_nodes:
      node = self.open_nodes.popleft()
      next_random_id = self.get_random_request(node)

      if next_random_id is None:
        break
      
      self.open_node(next_random_id, node.index)
  
  
  def visualize_tree(self):
    tree = nx.Graph()
    labels_dict = {}
    
    tree.add_nodes_from([k for k in range(self.node_count)])
    for (id, node) in self.all_bb_nodes.items():
      labels_dict[id] = ', '.join(str(e) for e in node.get_req_path())
      if node.children is not None:
          if len(node.children) != 0:
            for child in node.children:
              tree.add_edge(id, child)

    pos = nx.spring_layout(tree, k=1, iterations=1000)
    nx.draw(tree, labels=labels_dict, with_labels = True)


  def pick_best_routings(self):
    best_routings = []
    
    for (id, node) in self.all_bb_nodes.items():
      if len(node.children) == 0:
        if math.isclose(node.upper_bound, node.lower_bound):
          routing = self.find_ancestors_recursively(node)
          best_routings.append(routing)
    
    return best_routings


  def lb_is_ub(self, current_id):
    current_node = self.all_bb_nodes[current_id]

    if len(current_node.reqs_to_route) == 0:
      return
    
    requests = [self.amod_prob.requests[r] for r in current_node.reqs_to_route]
    solution_paths = self.amod_prob.upper_bound_routes(requests, current_node.remaining_capacity)
    
    for path in solution_paths:
      self.all_bb_nodes[self.node_count] = BB_node(self.node_count, path, current_id)
      current_node.children.append(self.node_count)
      self.all_bb_nodes[self.node_count].upper_bound = current_node.upper_bound
      self.all_bb_nodes[self.node_count].lower_bound = current_node.lower_bound

      current_node = self.all_bb_nodes[self.node_count]
      current_id = self.node_count
      self.node_count += 1
      

  def create_all_od_pairs(self):
    road_size = self.amod_prob.road.number_of_nodes()
    all_od_pairs = [(o, d) for o in range(road_size) for d in range(road_size) if o != d]
    return all_od_pairs
