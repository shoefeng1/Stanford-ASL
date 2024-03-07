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
from sklearn.model_selection import train_test_split
import datetime
import time

from amod import AMOD
from bb_class import BB_node
from bb_class import BB
import data_collector as dc
import make_train_test as mtt

class Network(nn.Module):
  def __init__(self, L1, L2, lr, epochs):
    super().__init__()
    self.hidden1 = nn.Linear(153, L1)
    self.hidden2 = nn.Linear(L1, L2)
    self.output = nn.Linear(L2, 154)
    self.weight_bias_dict1 = self.hidden1.state_dict()
    self.weight_bias_dict2 = self.hidden2.state_dict()
    self.sigmoid = nn.Sigmoid()
    self.criterion = nn.BCELoss()
    self.lr = lr
    self.epochs = epochs

  def forward(self, x):
    x = self.hidden1(x)
    x = self.sigmoid(x)
    x = self.hidden2(x)
    x = self.sigmoid(x)
    x = self.output(x)
    x = self.sigmoid(x)
    return x
  

  def train(self, inputs, outputs, test_inputs, test_outputs):
    optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
    epochs = self.epochs
    x_axis = []
    y_axis = []
    test_y_axis = []
    
    for e in range(epochs):
      
      x_axis.append(e)
      running_loss = 0
      test_running_loss = 0
      
      for x in range(len(inputs)):
        inputx = inputs[x]
        actual_output = outputs[x]
        
        optimizer.zero_grad()
        
        model_output = self.forward(inputx)
        loss = self.criterion(model_output, actual_output)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

      y_axis.append(running_loss / len(inputs))
      
      for i in range(len(test_inputs)):
        test_input = test_inputs[i]
        test_output = test_outputs[i]

        test_model_output = self.forward(test_input)

        test_loss = self.criterion(test_model_output, test_output)

        test_running_loss += test_loss.item()
      
      test_y_axis.append(test_running_loss / len(test_inputs))
      
    
    plt.plot(x_axis, y_axis, label="train loss")
    plt.plot(x_axis, test_y_axis, label="test loss")
    plt.legend()
    plt.show()
  
  def loss(self, inputx, output):
    model_output = self.forward(inputx)
    loss = self.criterion(model_output, output)
    return loss


# Replace this with your own custom road network. Make sure your road network is part of a new amod_prob instance.
new_roads = nx.Graph()
for k in range(18):
  new_roads.add_node(k)
# "bridge" goes from (1, 2)
new_roads.add_edges_from([(0, 1), (0, 3), (0, 14), (0, 15), (1, 2), (2, 7), \
                            (3, 4), (3, 11), (3, 13), (4, 5), (4, 8), (4, 11), (5, 6), (6, 7), \
                            (7, 10), (8, 9), (8, 10), (8, 12), (9, 10), (11, 12), \
                            (12, 13), (13, 14), (13, 17), (14, 15), (14, 16), (16, 17)])

new_requests = []

capacities = {}
for edge in new_roads.edges():
  capacities[edge] = 13
capacities[(1, 2)] = 1
capacities[(4, 8)] = 1
capacities[(13, 14)] = 5
capacities[(5, 6)] = 6
capacities[(0, 3)] = 5
capacities[(4, 5)] = 11
capacities[(7, 10)] = 5
capacities[(0, 15)] = 9
capacities[(14, 16)] = 7
capacities[(3, 13)] = 11
capacities[(4, 11)] = 10
capacities[(8, 9)] = 8
capacities[(13, 17)] = 12
capacities[(3, 4)] = 10
capacities[(9, 10)] = 11
capacities[(2, 7)] = 10


nroads_amod = AMOD(new_roads, new_requests, capacities)


tr_inp_df = pd.read_csv('/train-test-data/arr_tr_in.csv').reset_index()
tr_inp = [torch.Tensor(row) for index, row in tr_inp_df.drop(tr_inp_df.columns[[0, 1]], axis=1).iterrows()]

tr_outp_df = pd.read_csv('/train-test-data/arr_tr_out.csv').reset_index()
tr_outp = [torch.Tensor(row) for index, row in tr_outp_df.drop(tr_outp_df.columns[[0, 1]], axis=1).iterrows()]

te_inp_df = pd.read_csv('/train-test-data/arr_te_in.csv').reset_index()
te_inp = [torch.Tensor(row) for index, row in te_inp_df.drop(te_inp_df.columns[[0, 1]], axis=1).iterrows()]

te_outp_df = pd.read_csv('/train-test-data/arr_te_out.csv').reset_index()
te_outp = [torch.Tensor(row) for index, row in te_outp_df.drop(te_outp_df.columns[[0, 1]], axis=1).iterrows()]


nroads_NN = Network(1024, 512, 0.1, 500)

combo_trl_before = sum([nroads_NN.loss(tr_inp[k], tr_outp[k]) 
                 for k in range(len(tr_inp))]) / len(tr_inp)
combo_tel_before = sum([nroads_NN.loss(te_inp[k], te_outp[k]) 
                 for k in range(len(te_inp))]) / len(te_inp)

nroads_NN.train(tr_inp, tr_outp, te_inp, te_outp)

combo_trl_after = sum([nroads_NN.loss(tr_inp[k], tr_outp[k]) 
                 for k in range(len(tr_inp))]) / len(tr_inp)
combo_tel_after = sum([nroads_NN.loss(te_inp[k], te_outp[k]) 
                 for k in range(len(te_inp))]) / len(te_inp)



random_reqs = mtt.generate_req_sets(nroads_amod, 10, 30)
#     make reqs into input tensors
input_tensors = mtt.input_matrices(nroads_amod, random_reqs)

nn_node_counts = []
avg_all_node_counts = []

nn_best_times = []
avg_all_times = []

for k in range(len(random_reqs)):
    req_specific_node_counts = []
    req_specific_times = []
    
    random_sample = random_reqs[k]
    print(random_sample)
    in_tensor = input_tensors[k]
    
    req_reference = copy.deepcopy(random_sample)
    
# get nn predicted best req
# what if network outputs 154 (infeasible)?
    
    out_tensor = nroads_NN.forward(in_tensor)
    if torch.argmax(out_tensor) == 153:
        print("NN predicts problem is infeasible")
        continue
    else:
        best_req = nroads_amod.all_od_pairs[torch.argmax(out_tensor)]
    
    if best_req not in random_sample:
        print("NN predicted req not in original list of requests")
        best_req = random.choice(random_sample)
    
    move_item_in_list(random_sample, random_sample.index(best_req), 0)
    print("Random sample after ordering:", random_sample)
    req_reference.remove(best_req)
    
    nroads_amod.requests = random_sample
    throwaway = BB(nroads_amod)
    throwaway.build_tree()
    
    cool_tree = BB(nroads_amod)
    
    start_nn = time.time()
    cool_tree.build_tree()
    end_nn = time.time()
    
    req_specific_times.append(end_nn - start_nn)
    req_specific_node_counts.append(cool_tree.node_count)
    nn_best_times.append(end_nn - start_nn)
    nn_node_counts.append(cool_tree.node_count)
    
    for other_req in req_reference:
        move_item_in_list(random_sample, random_sample.index(other_req), 0)
        nroads_amod.requests = random_sample
        less_cool_tree = BB(nroads_amod)
        
        start = time.time()
        less_cool_tree.build_tree()
        end = time.time()
        
        req_specific_times.append(end - start)
        req_specific_node_counts.append(less_cool_tree.node_count)
    
    req_avg_time = sum(req_specific_times) / len(req_specific_times)
    req_avg_node_count = sum(req_specific_node_counts) / len(req_specific_node_counts)
    
    avg_all_times.append(req_avg_time)
    avg_all_node_counts.append(req_avg_node_count)
    
    print("Req specific node counts:", req_specific_node_counts)
    print("Req specific times:", req_specific_times)
