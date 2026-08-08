# -*- coding: utf-8 -*-

import numpy as np
def floyd_warshall(matrix):
   """
   Calculates the shortest path between each pair of nodes using the Floyd-Warshall algorithm.
   :param matrix: Adjacency matrix, a 2D list or numpy array
   :return: Matrix of shortest paths between all pairs of nodes
   """
   num_nodes = len(matrix)
   # Initialize distance matrix, set unreachable distances to infinity
   dist = np.where(np.array(matrix) == 0, float('inf'), np.array(matrix))
   np.fill_diagonal(dist, 0)  # Distance to self is 0
   # Apply Floyd-Warshall algorithm
   for k in range(num_nodes):
       for i in range(num_nodes):
           for j in range(num_nodes):
               dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
   return dist
def find_best_nodes(matrix, dis=None):
   """
   Finds nodes that are optimal based on the highest degree (connections)
   and lowest total distance to other nodes.
   :param matrix: Adjacency matrix, a 2D list or numpy array
   :return: List of nodes with the best combined score
   """
   disReal =  dis if dis is not None else np.ones_like(matrix)
   disReal = matrix*dis
   total_disReal = disReal.sum(axis=1)
   
   dist = floyd_warshall(matrix)
   num_nodes = len(matrix)
   # Calculate degree (number of direct connections) and centrality score for each node
   degrees = np.sum(matrix, axis=1)
   total_distances = [sum([dist[i][j] for j in range(num_nodes) if dist[i][j] != float('inf')]) for i in range(num_nodes)]
   # Combine scores: prioritize degree, with centrality as a secondary metric
   scores = [(i, degrees[i], -total_distances[i], -total_disReal[i]) for i in range(num_nodes)]
   # Sort by degree (descending), then by centrality (ascending) for ties
   scores.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
   # Extract nodes with the highest score
   best_nodes = scores[0][0] #[node[0] for node in scores if node[1] == scores[0][1]]
   return best_nodes

def is_reachable(matrix, start_node, end_node):
   """
   Checks if there is a path from start_node to end_node using DFS.
   :param matrix: Adjacency matrix
   :param start_node: Starting node index
   :param end_node: Ending node index
   :return: True if path exists, False otherwise
   """
   visited = set()
   def dfs(node):
       if node == end_node:
           return True
       visited.add(node)
       for neighbor, connected in enumerate(matrix[node]):
           if connected and neighbor not in visited:
               if dfs(neighbor):
                   return True
       for neighbor, connected in enumerate(matrix[:,node]):
           if connected and neighbor not in visited:
               if dfs(neighbor):
                   return True               
       return False
   return dfs(start_node)
def check_solvability(matrix, refNode):
   """
   Checks if each node is solvable (reachable) from the reference node.
   :param matrix: Adjacency matrix, a 2D list or numpy array
   :return: Dictionary indicating whether each node is solvable
   """
   solvability = {}
   for node in range(len(matrix)):
       if node == refNode:
           solvability[node] = True  # The reference node is always solvable
       else:
           solvability[node] = is_reachable(matrix, node, refNode)
   return solvability
if __name__=='__main__':
    # Example
    matrix = [
       [0, 1, 0, 0],
       [1, 0, 1, 1],
       [0, 1, 0, 1],
       [0, 1, 1, 0]
    ]
    
    # Example
    matrix = [
       [0, 1, 0, 0],
       [1, 0, 1, 1],
       [1, 1, 0, 1],
       [0, 0, 0, 0]
    ]
    matrix=np.array(matrix).astype(int)
    np.random.seed(44)
    dis=np.random.rand(*matrix.shape)
    
    np.fill_diagonal(dis, 0)
    best_nodes = find_best_nodes(matrix, dis=dis)
    print("Best nodes:", best_nodes)
    
    solvability = check_solvability(matrix, best_nodes)
    print("Node solvability:", solvability)
