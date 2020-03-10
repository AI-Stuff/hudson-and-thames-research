import numpy as np
import pandas as pd


"""
Minimum spanning tree function.
Parameters
    ----------
    matrix : Pandas DataFrame
        Adjacency matrix of graph structure to return the MST of.
"""


def mst(matrix):
    graph = matrix.replace(matrix, 0)  # intialize 0 edge adjacency matrix
    for pair in edges(matrix):
        i, j = pair
        if graph[j][i]:  # prevent symmetric redundancy
            graph[i][j] = graph[j][i]
        elif j not in dfs(graph, i, []):  # use depth first to check connection
            graph[i][j] = matrix[i][j]

    return graph


"""
Minimum spanning tree function.
Parameters
    ----------
    m_adj: Pandas DataFrame
        Adjacency matrix of graph to reference nodes from.
    
    vtx: str, int
        Label of starting node to perform depth first search from.
        
    path: list
        List used in modifying recursion of depth first search.
        
        
"""


def dfs(m_adj, vtx, path):
    src = vtx  # initialize source node
    path += [src]  # append source node

    neighbors = m_adj[src].where(m_adj[src] > 0)\
        .dropna()\
        .index.tolist()  # list of source node neighbors

    for neighbor in neighbors:
        if neighbor not in path:
            # recursively search for new paths
            path = dfs(m_adj, neighbor, path)

    return path


"""
Function returning list of edge pairs.
Parameters
    ----------
    m_adj : Pandas DataFrame
        Adjacency matrix of graph structure to derive list of pairs from.
"""

def edges(m_adj):
    raw_edges = m_adj.unstack()  # series of raw edge pairs
    edges_ord = raw_edges.sort_values(
        kind="mergesort")  # edge pairs by distance
    pairs = edges_ord[edges_ord != 0].index.tolist()  # remove non-edges
    return pairs
