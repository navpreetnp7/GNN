import numpy as np
import sys
import pickle as pkl
import networkx as nx


def load_data():

    graph = pkl.load(open("data/Taxi2017_total_net.pkl", "rb"))
    adj = np.array(nx.adjacency_matrix(graph[0]).todense(), dtype=float)
    adj = adj[graph[1],:][:,graph[1]]

    return adj
