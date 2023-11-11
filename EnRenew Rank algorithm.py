import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy
import random
import math
import pandas as pd





# coding: utf-8

# In[1]:

def EnRenewRank(G, topk, order):
    # N - 1
    all_degree = nx.number_of_nodes(G) - 1
    # avg degree
    k_ = nx.number_of_edges(G) * 2 / nx.number_of_nodes(G)
    # E<k>
    k_entropy = - k_ * ((k_ / all_degree) * math.log((k_ / all_degree)))

    # node's information pi
    node_information = {}
    for node in nx.nodes(G):
        information = (G.degree(node) / all_degree)
        node_information[node] = - information * math.log(information)

    # node's entropy Ei
    node_entropy = {}
    for node in nx.nodes(G):
        node_entropy[node] = 0
        for nbr in nx.neighbors(G, node):
            node_entropy[node] += node_information[nbr]

    rank = []
    for i in range(topk):
        # choose the max entropy node
        max_entropy_node, entropy = max(node_entropy.items(), key=lambda x: x[1])
        rank.append((max_entropy_node, entropy))

        cur_nbrs = nx.neighbors(G, max_entropy_node)
        for o in range(order):
            for nbr in cur_nbrs:
                if nbr in node_entropy:
                        node_entropy[nbr] -= (node_information[max_entropy_node] / k_entropy) / (2**o)
            next_nbrs = []
            for node in cur_nbrs:
                nbrs = nx.neighbors(G, node)
                next_nbrs.extend(nbrs)
            cur_nbrs = next_nbrs

        #set the information quantity of selected nodes to 0
        node_information[max_entropy_node] = 0
        # set entropy to 0
        node_entropy.pop(max_entropy_node)
    return rank




# read data
df_L = pd.read_csv("WLUSC.csv")
df_L.head(10)

# Create the directed and weighted graph using Source and Target for connections.

G_L = nx.from_pandas_edgelist(df_L, 'Source', 'Target', edge_attr='p', create_using=nx.DiGraph())

# Info of network

print('Lung Cancer Network:')
print(nx.info(G_L))
G_L.is_directed()


# Check connected or disconnected network --> directed--> Strongly/Weakly connectivity.

print('Is Lung Cancer Network strongly connected?', nx.is_strongly_connected(G_L))
print('Is Lung Cancer Network weakly connected?', nx.is_weakly_connected(G_L))
print('\n')

#Returns number of strongly connected components in graph.
print('The number strongly connected components in Lung Cancer Network:', nx.number_strongly_connected_components(G_L))

#Returns the number of weakly connected components in graph.
print('The number weakly connected components in Lung Cancer Network:', nx.number_weakly_connected_components(G_L))


# Generate connected components and select the largest:

largest_component_L = max(nx.weakly_connected_components(G_L), key=len)



Gconnected_L = G_L.subgraph(largest_component_L)

# UNFrozed the graph

Gconnected_LL = nx.Graph(Gconnected_L)
    # remove the self loop node in the graph 

Gconnected_LL.remove_edges_from(nx.selfloop_edges(Gconnected_LL))

print('Lung Cancer Network')
print(nx.info(Gconnected_LL))





l1=EnRenewRank(Gconnected_LL, 242,1)


