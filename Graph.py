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
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges


    @classmethod
    def erdos_renyi(self,n,p):
        """
        Constructs an instance of the class using Erdos-Renyi model.
        Args:
            n (int)  : number of vertices.
            p (float): probability to connect any pairs of vertices. Must be between 0 and 1.

        Returns:
            Graph : A instance of the class.
        """
        assert p>=0 and p<=1, "The probability p must be between 0 and 1."
        id_vertices = list(range(0, n))
        possible_edges = list(itertools.combinations(id_vertices,2))

        nb_trials = int(n*(n-1)/2)  # number of trials, probability of each trial
        flips = np.random.binomial(1, p, nb_trials)

        dictionary = dict(zip(possible_edges, flips))
        edges = [k for (k,v) in dictionary.items() if v == 1]

        return Graph(id_vertices, edges)


    @classmethod
    def barabasi_albert(self, G_start, t):
        """
        Constructs an instance of the class using Barabasi-Albert model.
        Args:
            G_start (Graph) : the initial connected graph.
            t (int): the number of vertices to add.

        Returns:
            Graph : A instance of the class.
        """
        G = self(G_start.vertices[:], G_start.edges[:])
        nb_vertices = len(G.vertices)

        for i in range (t):
            new_vertex_id = len(G.vertices)
            new_edges = []
            degrees_all = G.get_degrees_sum()

            for vertex in G.vertices:
                degree = len(G.get_neighbors(vertex))
                probability = degree/degrees_all
                flip = np.random.binomial(1, probability, 1)
                if flip == 1:
                    new_edges.append((new_vertex_id, vertex))
                #print(new_vertex_id, vertex, probability, flip)

            G.vertices.append(new_vertex_id)
            G.edges.extend(new_edges)

        return G


    def plot_nx(self):
        """Plots the graph using NetworkX function."""
        G=nx.Graph()
        G.add_nodes_from(self.vertices)
        G.add_edges_from(self.edges)
        nx.draw(G, with_labels=True)
        plt.show()


    def get_degree(self,vertex):
        """
        Returns the degree of the given vertex.
        """
        flatten_edges = [item for sublist in self.edges for item in sublist]
        degree = flatten_edges.count(vertex)
        return degree


    def get_all_degrees(self):
        """
        Returns a dictionary with vertices id as keys, and degrees as values.

        """
        flatten_edges = [item for sublist in self.edges for item in sublist]

        degrees_all = {} # dict (vertex_id:degree). Counting occurences in flatten_edges would have skipped 0-degree vertices.
        for vertex in self.vertices:
            degree = flatten_edges.count(vertex)
            degrees_all[vertex] = degree

        return degrees_all


    def get_degree_distribution(self, plot = 1):
        """
        Returns a dictionary with degrees as keys, and number of vertices that have this degree as values.
        """
        degrees_all = self.get_all_degrees()
        distribution = Counter(degrees_all.values()) # return a dict (degree:count)

        if plot == 1:
            df_distribution = pd.DataFrame.from_dict(distribution, orient='index').reset_index()
            df_distribution.rename(columns={'index':'degree',0:'count'}, inplace=True)
            df_distribution.sort_values(by="degree",ascending=1, inplace=True) # sort the degree by ascending order
            ax = df_distribution.plot.bar(x="degree", y="count", rot=0)

        return distribution


    def get_neighbors(self,vertex):
        """
        Returns a list containg all the neighbors of the given vertex.
        Args:
            vertex (int): the vertex id.
        Returns:
            int: degree of the vertex.
        """
        adjacent_edges = [list(edge) for edge in self.edges if vertex in edge]
        flatten_edges = [item for sublist in adjacent_edges for item in sublist]
        neighbors = [i for i in flatten_edges if i != vertex]
        return neighbors


    def get_clustering_coeff(self):
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

        for vertex in self.vertices:
            counts = self.get_all_degrees()
            kv = counts.get(vertex)

            if kv > 1: # we only consider vertices with degree > 1
                neighbors = self.get_neighbors(vertex)
                edges_between_neigh = [elt for elt in self.edges if elt[0] in neighbors and elt[1] in neighbors] # both extremities are neighbors
                nv = len(edges_between_neigh)
                CCV = (2*nv)/(kv*(kv-1))
                CCV_all.append(CCV)

        print(CCV_all)
        CCG = np.mean(CCV_all)
        return CCG


    def dijkstra_simple(self, source_index):
        """
        Computes a simplfied version of Dijkstra shortest path solving algorithm for
        the given source vertex. In this version, we consider all edge weights = 1.

        Args:
            vertex (int): the vertex index in the vertices list.
        Returns:
            DataFrame: a pandas dataframe containing all the shortest path starting from
            the given vertex.

        """
        visited = []
        unvisited = self.vertices[:] # copy by values, not by reference

        df = pd.DataFrame({'vertex': self.vertices, 'dist': math.inf, 'previous':""}) # set all distances to +inf
        df.set_index('vertex',inplace=True)

        df.at[unvisited[source_index],'dist'] = 0 # initialize distance for source

        while unvisited:

            d = df.loc[~df.index.isin(visited),["dist"]]

            min_dist_vertex = d[d["dist"] == d["dist"].min()].index[0]
            current_vertex = min_dist_vertex

            neighbors = self.get_neighbors(current_vertex)

            for neighbor in neighbors:
                candidate_path = df.iloc[current_vertex]["dist"]+1
                if candidate_path < df.iloc[neighbor]["dist"]:
                    df.at[neighbor,'dist'] = candidate_path
                    df.at[neighbor,'previous'] = current_vertex

            old_current = current_vertex
            unvisited.remove(old_current)
            visited.append(old_current)

        return df


    def get_all_distances(self):
        """
        Finds all the shortest paths between any pairs of vertices.
        """
        n = len(self.vertices)
        dist_all = np.empty((n,n))
        dist_all[:] = np.nan

        for source in self.vertices: #calculate distances for all vertices
            df_source = self.dijkstra_simple(source)
            dist_all[source] = df_source["dist"]

        return dist_all


    def get_diameter(self):
        """
        Returns the diameter of the graph.
        The diameter is the longest shortest path between any pair of vertices of the graph.
        """
        dist_all = self.get_all_distances(self)
        return dist_all.max()


    def get_path_length(self):
        """
        Returns the mean path length of the graph.
        The path length is the number of edges in the shortest path between 2 vertices,
        averaged over all pairs of vertices.

        """
        n = len(vertices)
        dist_all = self.get_all_distances()

        dist_sum = dist_all[np.triu_indices(n, k=1)] #extract upper triangle of the matrix, without diagonal
        avg_path_length = np.mean(dist_sum)

        return avg_path_length, dist_all


    def get_degrees_sum(self):
        """
        Returns the sum of all degrees, for a simple graph (without loops).
        """
        return len(self.edges)*2


    def depth_first_search(self, cc_vertex, vertex, visited):
        """
        Computes depth-first search (DFS) algorithm starting from a vertex.
        """
        visited.append(vertex)
        cc_vertex.append(vertex)

        print(cc_vertex)
        for neighbor in self.get_neighbors(vertex):
            if neighbor not in visited:
                cc_vertex = self.depth_first_search(cc_vertex, neighbor, visited)
        return cc_vertex


    def get_connected_components(self):
        """
        Returns a list of list containing the connected components.
        [ [CC1_vertex1, CC1_vertex2, CC2_vertex3], [CC2_vertex1, CC2_vertex2]]
        """

        visited = []
        cc_all = []

        for vertex in self.vertices:
            if vertex not in visited:
                cc_vertex = []
                cc_all.append(self.depth_first_search(cc_vertex, vertex, visited))
        return cc_all

    @staticmethod
    def get_symmetric_edges(edges):
        """
        Returns the symmetric edges of the graph.
        """
        return [edge[::-1] for edge in edges]


    def get_assortativity(self):
        """
        Returns the assortativity coefficient of the graph.
        This coefficient is the Pearson correlation coefficient of degree between pairs of linked nodes.
        """
        degrees_all = self.get_all_degrees()
        edges_symmetry = Graph.get_symmetric_edges(self.edges)
        edges_double = self.edges + edges_symmetry

        edges_degree = list(map(lambda v:(degrees_all.get(v[0]),degrees_all.get(v[1])),edges_double))

        edges_degree = np.array(edges_degree)
        x = edges_degree[:,0]
        y = edges_degree[:,1]

        x_mean_deviation = x.mean()
        y_mean_deviation = y.mean()

        x_deviation = x-x_mean_deviation
        y_deviation = y-y_mean_deviation

        deviation_product_sum = (x_deviation*y_deviation).sum()
        deviation_squared_product_sum = np.sqrt((x_deviation**2).sum()*(y_deviation**2).sum())

        assortativity = deviation_product_sum/deviation_squared_product_sum

        return assortativity


    def test(self):
        print("TEST CHANGES")
