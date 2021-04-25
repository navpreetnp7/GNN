import numpy as np
import sys
import pickle as pkl
import networkx as nx


def load_data():

    graph = pkl.load(open("data/Taxi2017_total_net.pkl", "rb"))
    adj = np.array(nx.adjacency_matrix(graph[0]).todense(), dtype=float)
    adj = adj[graph[1],:][:,graph[1]]

    return adj

def toy_data():

    graph = nx.DiGraph()
    graph.add_nodes_from([1, 2, 3, 4, 5])
    graph.add_edge(1, 2, weight=10)
    graph.add_edge(1, 5, weight=57)
    graph.add_edge(2, 1, weight=8)
    graph.add_edge(2, 4, weight=34)
    graph.add_edge(2, 5, weight=75)
    graph.add_edge(4, 1, weight=24)
    graph.add_edge(5, 4, weight=14)
    graph.add_edge(5, 1, weight=73)
    graph.add_edge(5, 2, weight=48)

    adj = np.array(nx.adjacency_matrix(graph).todense(), dtype=float)
    return adj