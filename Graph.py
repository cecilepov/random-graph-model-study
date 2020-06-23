#!/usr/bin/env python
#coding: utf-8

"""
This module provides a simple class implementation for undirected graphs.

A graph is a data structure composed of vertices and edges.
It also provides features computation for network analysis.

Typical usage example:
    vertices = [0,1,2,3]
    edges    = [[0,1],[2,3],[0,3]]
    my_graph = Graph(vertices, edges)
    my_graph.get_connected_components()

"""

# Import the modules needed to run the code.
import numpy as np
import itertools
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
import collections
from   collections import Counter
import pandas as pd
import math
import pylab
import random
from collections import deque
from statistics import mean
import time

from itertools import chain
from functools import reduce
from copy import deepcopy


# Authorship information
__author__  = "Pov Cécile"
__credits__ = ["Tabourier Lionel","Tarissan Fabien"]
__version__ = "1.0"
__date__    = "20/04/2020"
__email__   = "cecile.pov@gmail.com"



class Graph:
    """This class is a implementation for undirected graph.
    A graph is a data structure composed of vertices and edges.

    Attributes:
        vertices (list of int): list of vertices id.
        edges (list of list): list of edges.

    """
    def __init__(self, nb_sets):
        nb_sets = all



    @classmethod
    def watts_strogatz(self, n, k, p):
        """
        Constructs an instance of the class using Watts-Strogatz model.
        Args:
            n (int)   : number of vertices.
            k (int)   : each vertex is connected to its k nearest neighbors.
            p (float) : probability to rewire any edge. Must be between 0 and 1.

        Returns:
            Graph : A instance of the class.
        """
        assert p>=0 and p<=1, "The probability p must be between 0 and 1."

        G = self.ring_lattice(n,k)
        nb_edges = len(G.edges)
        halfk = k//2

        for lap in range(halfk):
            for i in range (lap, nb_edges, halfk):
                current_vertex = G.edges[i][0]
                if np.random.binomial(1, p):
                    choice_set = set(G.vertices) - set(G.get_neighbors(current_vertex)) - {current_vertex}
                    if choice_set:
                        new_extremity = random.choice(list(choice_set))
                        G.edges[i] = [current_vertex,new_extremity]
        return G



    def plot_nx(self, circular=False, node_size = 400, figsize=(5,4)):
        """Plots the graph using NetworkX function."""
        G=nx.Graph()
        G.add_nodes_from(self.vertices)
        G.add_edges_from(self.edges)
        if circular==True:
            pos=nx.circular_layout(G)
            pylab.figure(3,figsize=figsize)
            nx.draw(G, pos, with_labels=True, node_color='y', edge_color='#909090', node_size=node_size)
            pylab.show()
            #nx.draw_circular(G, node_color='y', edge_color='#909090', node_size=500, with_labels=True)
        else:
            nx.draw(G, node_color='y', edge_color='#909090',with_labels=True)
        plt.show()


    def get_degree(self,vertex,set1):
        """
        Returns the degree of the given vertex.
        """
        return len(set1.get(vertex))


    def get_all_degrees(self,set1):
        """
        Returns a dictionary with vertices id as keys, and degrees as values.

        """
        return {key: len(value) for key, value in set1.items()}



    def get_degree_mean(self,set):
        return np.mean(list(self.get_all_degrees().values()))


    def get_degree_max(self):
            return max(list(self.get_all_degrees().values()))

    def get_degree_min(self):
                return min(list(self.get_all_degrees().values()))


    def get_degree_distribution(self, set1, plot = 1):
        """
        Returns a dictionary with degrees as keys, and number of vertices that have this degree as values.
        """
        degrees_all = self.get_all_degrees(set1)
        distribution = Counter(degrees_all.values()) # return a dict (degree:count)

        if plot == 1:
            df_distribution = pd.DataFrame.from_dict(distribution, orient='index').reset_index()
            df_distribution.rename(columns={'index':'degree',0:'count'}, inplace=True)
            df_distribution.sort_values(by="degree",ascending=1, inplace=True) # sort the degree by ascending order
            ax = df_distribution.plot.bar(x="degree", y="count", title="Degree distribution", rot=0)

        return distribution


    def get_neighbors(self,vertex,set1):
        """
        Returns a list containing all the neighbors of the given vertex.
        Args:
            vertex (int): the vertex id.
        Returns:
            int: degree of the vertex.
        """
        # adjacent_edges = [list(edge) for edge in self.edges if vertex in edge]
        # flatten_edges = [item for sublist in adjacent_edges for item in sublist]
        # neighbors = [i for i in flatten_edges if i != vertex]
        # return neighbors
        return set1.get(vertex)


    def get_clustering_coeff(self,set1):
        """
        Returns the clustering coefficient CC(G) of a graph G.
        For all v vertices, where degree(v) > 1:
            CC(v) = (2*Nv) / (Kv*(Kv-1))
            With : Nv the number of edges between neighbors of v
                   Kv the degree of v
                   Kv*(Kv-1) the max number of possible interconnections between neighbors of v
        CC(G) is the mean of all Cc(v)
        """

        CCV_all = [] #list to store all local CC

        for vertex, kv in self.get_all_degrees(set1).items():
            if kv > 1:
                neighbors = self.get_neighbors(vertex,set1)
                nv = sum( len(self.get_neighbors(neighbor,set1) & neighbors) for neighbor in neighbors)/2 #nb edges between neighbors
                CC_local = (2*nv)/(kv*(kv-1))
                CCV_all.append(CC_local)

        return np.mean(CCV_all)




    def bfs(self, source,set1):
        queue = deque([source])
        level = {source: 0}
        parent = {source: None}

        while queue:
            current_vertex = queue.popleft()
            for neighbor in self.get_neighbors(current_vertex,set1):
                if neighbor not in level:
                    queue.append(neighbor)
                    level[neighbor] = level[current_vertex] + 1
                    parent[neighbor] = current_vertex
        return level, parent


    def get_diameter(self,set1):
        """
        Returns the diameter of the graph.
        The diameter is the longest shortest path between any pair of vertices of the graph.
        """
        max_global = float('-inf')
        for vertex in set1:
            distances,_ = self.bfs(vertex,set1)
            max_local = max(distances.values())
            if max_global < max_local:
                max_global = max_local
        return max_global


    def get_path_length(self,set1):
        """
        Returns the mean path length of the graph.
        The path length is the number of edges in the shortest path between 2 vertices,
        averaged over all pairs of vertices.
        """
        distances_sum = 0
        nb_elts = 0
        for elt in set1:
            distances,_ = self.bfs(elt,set1)
            distances_sum += sum(distances.values())
            nb_elts += len(distances) -1 # distance to current element = 0

        return distances_sum/nb_elts


    def get_degrees_sum(self,set1):
        """
        Returns the sum of all degrees, for a simple graph (without loops).
        """
        return len(list(chain.from_iterable(set1.values())))



    def depth_first_search(self, vertex, cc_vertex, visited,tupleset, set1):#toggle):
        """
        Computes depth-first search (DFS) algorithm starting from a vertex.
        """
        visited.append(vertex)
        cc_vertex.append(vertex)


        #print(cc_vertex)
        double = next(toggle)
        t = double[0]
        print("-----\n")
        print("visited",visited)
        print(vertex, "in", double[1])

        # if vertex is "B":
        #     z = "JE SUIS A"
        # else:
        #      z = "NON"
        z =

        neighbors = self.get_neighbors(vertex,t)
        print("neighbors", neighbors)
        for neighbor in neighbors:
            if neighbor not in visited:
                print("goes in neighbor",neighbor)
                cc_vertex = self.depth_first_search(neighbor, cc_vertex, visited,toggle)
                next(toggle)
                print("toggle value : ",z)
                # print("ME", vertex)
            else:
                print(neighbor, "ignored")

        print()
        return cc_vertex


    def get_connected_components(self,set1,set2):
        """
        Returns a list of list containing the connected components.
        [ [CC1_vertex1, CC1_vertex2, CC2_vertex3], [CC2_vertex1, CC2_vertex2]]
        """
        def toggler():
            while True:
                yield set1,"top"
                yield set2,"bot"

        toggle = toggler()

        visited = []
        cc_all = []

        for vertex in set1:
            if vertex not in visited:
                # print(vertex)
                cc_vertex = []
                cc_all.append(self.depth_first_search(vertex, cc_vertex, visited, (set1,set2))
        return cc_all



# MARCHE POUR GRAPHE SIMPLE !!

    # def depth_first_search(self, vertex, cc_vertex, visited,set1):
    #     """
    #     Computes depth-first search (DFS) algorithm starting from a vertex.
    #     """
    #     visited.append(vertex)
    #     cc_vertex.append(vertex)
    #
    #     #print(cc_vertex)
    #     for neighbor in self.get_neighbors(vertex,set1):
    #         if neighbor not in visited:
    #             cc_vertex = self.depth_first_search(neighbor, cc_vertex, visited,set1)
    #     return cc_vertex
    #
    #
    # def get_connected_components(self,set1):
    #     """
    #     Returns a list of list containing the connected components.
    #     [ [CC1_vertex1, CC1_vertex2, CC2_vertex3], [CC2_vertex1, CC2_vertex2]]
    #     """
    #     def toggler():
    #         while True:
    #             yield 0
    #             yield 1
    #
    #     visited = []
    #     cc_all = []
    #
    #     for vertex in set1:
    #         if vertex not in visited:
    #             cc_vertex = []
    #             cc_all.append(self.depth_first_search(vertex, cc_vertex, visited,set1))
    #     return cc_all
#-------------------------------------------------

    # @staticmethod
    # def get_symmetric_edges(edges):
    #     """
    #     Returns the symmetric edges of the graph.
    #     """
    #     return [edge[::-1] for edge in edges]
    #
    #
    # def get_assortativity(self):
    #     """
    #     Returns the assortativity coefficient of the graph.
    #     This coefficient is the Pearson correlation coefficient of degree between pairs of linked nodes.
    #     """
    #     degrees_all = self.get_all_degrees()
    #     edges_symmetry = Graph.get_symmetric_edges(self.edges)
    #     edges_double = self.edges + edges_symmetry
    #
    #     edges_degree = list(map(lambda v:(degrees_all.get(v[0]),degrees_all.get(v[1])),edges_double))
    #
    #     edges_degree = np.array(edges_degree)
    #     x = edges_degree[:,0]
    #     y = edges_degree[:,1]
    #
    #     x_mean_deviation = x.mean()
    #     y_mean_deviation = y.mean()
    #
    #     x_deviation = x-x_mean_deviation
    #     y_deviation = y-y_mean_deviation
    #     # print(x_deviation)
    #     # print(y_deviation)
    #     deviation_product_sum = (x_deviation*y_deviation).sum()
    #     deviation_squared_product_sum = np.sqrt((x_deviation**2).sum()*(y_deviation**2).sum())
    #
    #     #print(deviation_product_sum)
    #     #print(deviation_squared_product_sum)
    #     assortativity = deviation_product_sum/deviation_squared_product_sum
    #
    #     return assortativity
    #
    #
    # def analyze(self):
    #     start_time = time.time()
    #     self.get_degree_distribution()
    #     features = ["nb_vertices",
    #                 "nb_edges",
    #                 "clustering_coeff",
    #                 "connected_components",
    #                 "diameter",
    #                 "path_length",
    #                 "assortativity",
    #                 "avg_degree",
    #                 "degrees_sum",
    #                 "degree_mean",
    #                 "degree_min",
    #                 "degree_max"]
    #
    #     df = pd.DataFrame(columns = {"value":""})
    #
    #     df.loc["nb_vertices"] = len(self.vertices)
    #     df.loc["nb_edges"] = len(self.edges)
    #     df.loc["clustering_coeff"] = self.get_clustering_coeff()
    #     df.loc["connected_components"] = len(self.get_connected_components())
    #     df.loc["diameter"] = self.get_diameter()
    #     df.loc["Assortativity"] = self.get_assortativity()
    #     df.loc["path_length"] = self.get_path_length()
    #     df.loc["degree_avg"] = self.get_degree_mean()
    #     df.loc["degree_sum"] = self.get_degrees_sum()
    #     df.loc["degree_min"] = self.get_degree_min()
    #     df.loc["degree_max"] = self.get_degree_max()
    #     print("--- %s secconds ---" % (time.time() - start_time))
    #
    #     return df


    def test(self):
        print("TEST CHANGES")


    #---------------------------------------------------------------------------
    # OLD FUNCTIONS
    #---------------------------------------------------------------------------

    # @classmethod
    # def erdos_renyi(self,n,p):
    #     """
    #     Constructs an instance of the class using Erdos-Renyi model.
    #     Args:
    #         n (int)  : number of vertices.
    #         p (float): probability to connect any pairs of vertices. Must be between 0 and 1.
    #
    #     Returns:
    #         Graph : A instance of the class.
    #     """
    #     assert p>=0 and p<=1, "The probability p must be between 0 and 1."
    #     id_vertices = list(range(0, n))
    #     possible_edges = list(itertools.combinations(id_vertices,2))
    #
    #     nb_trials = int(n*(n-1)/2)  # number of trials, probability of each trial
    #     flips = np.random.binomial(1, p, nb_trials)
    #
    #     dictionary = dict(zip(possible_edges, flips))
    #     edges = [k for (k,v) in dictionary.items() if v == 1]
    #
    #     return self(id_vertices, edges)
    #
    #
    # @classmethod
    # def barabasi_albert(self, G_start, t):
    #     """
    #     Constructs an instance of the class using Barabasi-Albert model.
    #     Args:
    #         G_start (Graph) : the initial connected graph.
    #         t (int): the number of vertices to add.
    #
    #     Returns:
    #         Graph : A instance of the class.
    #     """
    #     G = self(G_start.vertices[:], G_start.edges[:])
    #     nb_vertices = len(G.vertices)
    #
    #     for i in range (t):
    #         new_vertex_id = len(G.vertices)
    #         new_edges = []
    #         degrees_all = G.get_degrees_sum()
    #
    #         for vertex in G.vertices:
    #             degree = len(G.get_neighbors(vertex))
    #             probability = degree/degrees_all
    #             flip = np.random.binomial(1, probability, 1)
    #             if flip == 1:
    #                 new_edges.append((new_vertex_id, vertex))
    #             #print(new_vertex_id, vertex, probability, flip)
    #
    #         G.vertices.append(new_vertex_id)
    #         G.edges.extend(new_edges)
    #
    #     return G
    #
    #
    # @staticmethod
    # def ring_lattice_edges(vertices, k):
    #     """
    #         Return all the edges for a defined regular ring lattice.
    #         Args:
    #             vertices (list<int>) : list of vertices.
    #             k (int) : each vertex is connected to its k nearest neighbors.
    #
    #         Returns:
    #             generator : A generator containing all the edges of the regular ring lattice.
    #     """
    #     halfk = k//2
    #     n = len(vertices)
    #     for i, u in enumerate(vertices):
    #         for j in range(i+1, i+halfk+1):
    #             v = vertices[j % n] #edges for "next" neighbors
    #             yield [u, v]
    #
    #
    # @classmethod
    # def ring_lattice(self, n, k):
    #     """
    #     Constructs a regular ring lattice.
    #     Args:
    #         n (int) : number of vertices.
    #         k (int) : each vertex is connected to its k nearest neighbors.
    #
    #     Returns:
    #         Graph : A regular ring lattice.
    #     """
    #     assert k%2 == 0, "k (number of nearest neighbors) must be an even number."
    #     assert k<=n, "k (number of nearest neighbors) must be equal or less than n (number of vertices)."
    #
    #     vertices = list(range(n))
    #     edges = list(self.ring_lattice_edges(vertices, k))
    #     print(type(self.ring_lattice_edges(vertices, k)))
    #     assert len(edges) == n*k/2, "Incorrect number of edges."
    #     return self(vertices,edges)
