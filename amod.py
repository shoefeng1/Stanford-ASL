import networkx as nx
import numpy as np
import cvxpy as cp
import copy
import matplotlib.pyplot as plt
from collections import deque
import math

class AMOD:
  def __init__(self, road, requests, capacity):
    self.road = road
    self.requests = copy.deepcopy(requests)
    self.capacity = copy.deepcopy(capacity)
    self.all_od_pairs = [(o, d) for o in range(self.road.number_of_nodes()) 
                for d in range(o + 1, self.road.number_of_nodes())]


  def LP_relaxation(self, overrideReqs = None, overrideCaps = None):
    if overrideReqs is None:
      reqs = self.requests
    else:
      reqs = overrideReqs
    
    if overrideCaps is None:
      caps = self.capacity
    else:
      caps = overrideCaps
    
    n = self.road.number_of_nodes()
    
    num_requests = len(reqs)
    F = {}
    for i in range(num_requests):
      F[i] = cp.Variable((n, n), nonneg = True)
    obj = cp.Minimize(cp.sum([cp.sum(vals) for vals in F.values()]))


    cons = []
    for m in range(num_requests):
      for j in self.road.nodes:
        is_j_origin = (reqs[m][0] == j)
        is_j_destination = (reqs[m][1] == j)
        j_identifier = j

        cons.append(cp.sum([F[m][i, j_identifier] for i in list(self.road.neighbors(j))]) + is_j_origin \
                    == cp.sum([F[m][j_identifier, k] for k in list(self.road.neighbors(j))]) + is_j_destination)
        
    for edge in caps.keys():
      cons.append(cp.sum([(F[m][edge[0], edge[1]] + F[m][edge[1], edge[0]]) for m in range(num_requests)]) <= caps[edge])


    prob = cp.Problem(obj, cons)
    prob.solve()
    if prob.status == "infeasible":
      return F, math.inf

    return F, prob.value
  
  def LP_branch(self, path):
    branch_requests = copy.deepcopy(self.requests)
    branch_requests.remove((path[0], path[len(path) - 1]))

    branch_capacity = copy.deepcopy(self.capacity)

    for k in range(len(path) - 1):
      if self.road.has_edge(path[k], path[k+1]):
        if (path[k], path[k+1]) in branch_capacity.keys():
          branch_capacity[path[k], path[k+1]] -= 1
        elif (path[k+1], path[k]) in branch_capacity.keys():
          branch_capacity[path[k+1], path[k]] -= 1
        else:
          print("ERROR: Key does not exist in self.capacity",path[k],path[k+1])

    return self.LP_relaxation(branch_requests, branch_capacity)


  def round_LP(self, F, round_requests):
    
    rounded_paths = []
    n = self.road.number_of_nodes()
    num_requests = len(round_requests)

    for m in range(num_requests):
      origin = round_requests[m][0]
      destination = round_requests[m][1]

      rounded_path = [origin]
      possible_next = {}
      eps = 1e-8
      at_j = False

      while not at_j:
        possible_next = {}

        for j in range(n):          
            if F[m][origin, j].value >= eps:
              possible_next[j] = F[m][origin, j].value
          
        k = max(possible_next, key=possible_next.get)
        rounded_path.append(k)
        
        origin = k
        at_j = k == destination

      rounded_paths.append(rounded_path)
    
    return rounded_paths
    print(rounded_paths)


  def check_capacities(self, testSolns, overrideCaps = None):
    if overrideCaps is None:
      capacity = self.capacity
    else:
      capacity = overrideCaps
    
    occupancy = {}
    violations = {}

    for edge in self.road.edges():
      occupancy[edge] = 0
    
    for m in range(len(testSolns)):
      test_path = testSolns[m]

      for k in range(len(test_path) - 1):
        if self.road.has_edge(test_path[k], test_path[k+1]):
          if (test_path[k], test_path[k+1]) in occupancy.keys():
            occupancy[(test_path[k], test_path[k+1])] += 1
          elif (test_path[k+1], test_path[k]) in occupancy.keys():
            occupancy[(test_path[k+1], test_path[k])] += 1
        else:
          raise Error('Path does not exist')

    for edge in self.road.edges():
      if occupancy[edge] > capacity[edge]:
        violations[edge] = occupancy[edge] - capacity[edge]

    return violations


  def ub_branch_routes(self, given_branch):
    temp_capacity = copy.deepcopy(self.capacity)
    temp_request = copy.deepcopy(self.requests)
    temp_request.remove((given_branch[0], given_branch[-1]))
    F,_ = self.LP_branch(given_branch)
    rounded_paths = self.round_LP(F, temp_request)
    rounded_paths.append(given_branch)
    violations = self.check_capacities(rounded_paths)
    rounded_paths.remove(given_branch)

    reroute_requests = []
    for rounded_path in rounded_paths:
      for k in range(len(rounded_path) - 1):
        if ((rounded_path[k], rounded_path[k+1]) in violations.keys()) or ((rounded_path[k+1], rounded_path[k]) in violations.keys()):
          reroute_requests.append((rounded_path[0], rounded_path[-1]))
          
    unique_rr_requests = set([])
    for request in reroute_requests:
      unique_rr_requests.add(request)
    reroute_requests = list(unique_rr_requests)

    for bad_request in reroute_requests:
      for path in rounded_paths:
        if bad_request == (path[0], path[-1]):
          rounded_paths.remove(path)

    rounded_paths.append(given_branch)
    for path in rounded_paths:
      for k in range(len(path) - 1):
        if self.road.has_edge(path[k], path[k+1]):
          if (path[k], path[k+1]) in temp_capacity.keys():
            temp_capacity[path[k], path[k+1]] -= 1
          elif (path[k+1], path[k]) in temp_capacity.keys():
            temp_capacity[path[k+1], path[k]] -= 1
          else:
            print("ERROR: Key does not exist in self.capacity",path[k],path[k+1])

    for request in reroute_requests:
      F,obj = self.LP_relaxation([request], temp_capacity)
      rerouted_path_list = self.round_LP(F, [request])
      rerouted_path = rerouted_path_list[0]
      for k in range(len(rerouted_path) - 1):
        if self.road.has_edge(rerouted_path[k], rerouted_path[k+1]):
          if (rerouted_path[k], rerouted_path[k+1]) in temp_capacity.keys():
            temp_capacity[rerouted_path[k], rerouted_path[k+1]] -= 1
          elif (rerouted_path[k+1], rerouted_path[k]) in temp_capacity.keys():
            temp_capacity[rerouted_path[k+1], rerouted_path[k]] -= 1
          else:
            print("ERROR: Key does not exist in self.capacity",rerouted_path[k],rerouted_path[k+1])
        else:
          print("ERROR: road does not have edge", rerouted_path[k], rerouted_path[k+1])
      rounded_paths.append(rerouted_path)
    
    return rounded_paths


  def upper_bound_routes(self, overrideReqs = None, overrideCaps = None):
    if overrideCaps is None:
      temp_capacity = copy.deepcopy(self.capacity)
    else:
      temp_capacity = overrideCaps

    if overrideReqs is None:
      temp_requests = copy.deepcopy(self.requests)
    else:
      temp_requests = overrideReqs
    
    F, obj = self.LP_relaxation(temp_requests, temp_capacity)
    
    if obj == math.inf:
        print("in upper bound routes, obj was infinity and returned inf")
        return math.inf
      
    rounded_paths = self.round_LP(F, temp_requests)
    violations = self.check_capacities(rounded_paths, temp_capacity)

    reroute_requests = []
    for rounded_path in rounded_paths:
      for k in range(len(rounded_path) - 1):
        if ((rounded_path[k], rounded_path[k+1]) in violations.keys()) or ((rounded_path[k+1], rounded_path[k]) in violations.keys()):
          reroute_requests.append((rounded_path[0], rounded_path[-1]))
          
    unique_rr_requests = set([])
    for request in reroute_requests:
      unique_rr_requests.add(request)
    reroute_requests = list(unique_rr_requests)

    for request in reroute_requests:
      for path in rounded_paths:
        if request == (path[0], path[-1]):
          rounded_paths.remove(path)

    for path in rounded_paths:
      for k in range(len(path) - 1):
        if self.road.has_edge(path[k], path[k+1]):
          if (path[k], path[k+1]) in temp_capacity.keys():
            temp_capacity[path[k], path[k+1]] -= 1
          elif (path[k+1], path[k]) in temp_capacity.keys():
            temp_capacity[path[k+1], path[k]] -= 1
          else:
            print("ERROR: Key does not exist in self.capacity",path[k],path[k+1])
        else:
          print("ERROR: road does not have edge", path[k],path[k+1])

    for request in reroute_requests:
      F, obj = self.LP_relaxation([request], temp_capacity)
      if obj == math.inf:
        return math.inf
      rerouted_path_list = self.round_LP(F, [request])
      rerouted_path = rerouted_path_list[0]
      for k in range(len(rerouted_path) - 1):
        if self.road.has_edge(rerouted_path[k], rerouted_path[k+1]):
          if (rerouted_path[k], rerouted_path[k+1]) in temp_capacity.keys():
            temp_capacity[rerouted_path[k], rerouted_path[k+1]] -= 1
          elif (rerouted_path[k+1], rerouted_path[k]) in temp_capacity.keys():
            temp_capacity[rerouted_path[k+1], rerouted_path[k]] -= 1
          else:
            print("ERROR: Key does not exist in self.capacity",rerouted_path[k],rerouted_path[k+1])
        else:
          print("ERROR: road does not have edge", rerouted_path[k], rerouted_path[k+1])
      rounded_paths.append(rerouted_path)
    
    return rounded_paths


  def all_round_paths(self, F, request, modified_reqs = None):
    G = nx.DiGraph()

    if modified_reqs is None:
      requests = self.requests
    else:
      requests = modified_reqs

    k = requests.index(request)
    origin = request[0]
    destination = request[1]
    G.add_node(origin)
    n = self.road.number_of_nodes()

    eps = 1e-8
    considering = deque()
    considering.append(origin)
    while considering:
      node = considering.popleft()
      next_nodes = [m for m in range(n) if F[k][node, m].value >= eps]
      G.add_nodes_from(next_nodes)
      G.add_edges_from([(node, m) for m in next_nodes])
      considering.extend([m for m in next_nodes if m != destination])

    possible_paths = nx.all_simple_paths(G, origin, destination)
    return list(possible_paths)
  

  def evaluate_objective(self, paths):
    objective_value = sum(len(path) - 1 for path in paths)
    
    return objective_value


  def make_ub_lb(self, path):
    rounded_paths = self.upper_bound_routes(path)
    upper_bound = self.evaluate_objective(rounded_paths)

    _,lower_bound = self.LP_branch(path)

    return upper_bound, lower_bound
