from Graph import Graph
from collections import defaultdict

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
from collections import namedtuple
# class SimpleGraph(Graph):

# Authorship information
__author__  = "Pov Cécile"
__credits__ = ["Tabourier Lionel","Tarissan Fabien"]
__version__ = "2.0"
__date__    = "12/10/2020"
__email__   = "cecile.pov@gmail.com"


class UnipartiteGraph(Graph):

    """
    This class is a implementation for undirected and unweighted unipartite / 1-mode graphs.
    A unipartite graph is a k-partite graph such that k = 1.
    A graph is a data structure composed of vertices and edges.

    Attributes:
        all (dict): an adjacency list represented as a dictionnary.
                    Each element is a pair vertex_ID:set of neighbors of vertex_ID.

    Example:
    The following graph G:
    A --- B --- C
    |
    D

    Is represented as:
    G.all = {"A":{"B","D"},
             "B":{"A","C"},
             "C":{"B"},
             "D":{"A"}}
    """

    def __init__(self, all):

        Graph.__init__(self, 1)
        self.all = all
        # if self.default_set == None:
        #     self.default_set = self.all


    #===========================================================================
    # CLASS METHODS
    #===========================================================================
    @classmethod
    def from_lists(cls,vertices, edges):
        """
        Constructs an instance of the class from a list of vertices and edges.
        Args:
            vertices (list) : list of vertices ID. Ex: [1,2,3,4] or [A,B,C,D]
            edges (list<list>) : list of edges. An edge is described as a pair (A,B), with A and B as vertices ID.
                              Ex: [(1,2),(2,4)]
        Returns:
            Graph : A instance of the class.
        """
        all = {vertice:set() for vertice in vertices}
        all = defaultdict(set,all)

        for extremity1,extremity2 in edges:
            all[extremity1].add(extremity2)
            all[extremity2].add(extremity1)

        return cls(all)


    @classmethod
    def from_file(cls,filename,separator = ",", start = 0):
        """
        Constructs an instance of the class using a formatted file.
        The file format must be a list of edges that may be preceded by a header/metadata (non formatted).
        Except for the header, each line is formatted as "A B", describing an edge between
        vertices A and B.

        Args:
            filename  (str)
            separator (str)
            start (int) : first formatted line

        Returns:
            Graph : A instance of the class.
        """
        with open(filename, 'r') as filehandle: # OK
            lines = filehandle.readlines()
            d = defaultdict(set)

            for i in range (start, len(lines)):
                elts = list(map(int, lines[i].strip().split(separator)))
                extremity1 = elts[0]
                extremity2 = elts[1]

                if extremity1 != extremity2:
                    d[extremity1].add(extremity2)
                    d[extremity2].add(extremity1)
                else:
                    d[extremity1] = set()

            return cls(d)


    @classmethod
    def configuration_model(cls,degree_dist):
        """
        Constructs an instance of the class using the Configuration Model.
        Args:
            degree_dist (dict<int,int>): degree distribution of the output graph.
                                         Each element of the dictionary is a pair degree:count.
        Returns:
            Graph : A instance of the class.
        """
        stubs = [k for k,v in degree_dist.items() for i in range(v)]
        shuffle(stubs)

        edges = [stubs[i:i+2] for i in range(0,len(stubs),2)]
        new_all = defaultdict(set,{k:set() for k in degree_dist.keys()})

        for elt in edges:
            new_all[elt[0]].add(elt[1])
            new_all[elt[1]].add(elt[0])

        return cls(new_all)


    @classmethod
    def erdos_renyi(cls,n,p):
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

        return cls.from_lists(id_vertices,edges)


    @classmethod
    def barabasi_albert(cls, G_start, t):
        """
        Constructs an instance of the class using Barabasi-Albert model.
        Args:
            G_start (Graph) : the initial connected graph.
            t (int): the number of vertices to add.
        Returns:
            Graph : A instance of the class.
        """
        G = deepcopy(G_start)

        for i in range (t):
            new_vertex_id = len(G.all)
            degrees_all = G.get_degrees_sum(G.all)
            new_edges = set()

            for vertex in G.all:
                degree = G.get_degree(vertex,G.all)
                probability = degree/degrees_all
                flip = np.random.binomial(1, probability, 1)
                if flip == 1:
                    new_edges.add(vertex)

            G.add_vertex(new_vertex_id, new_edges)

        return G


    @staticmethod
    def ring_lattice_edges(vertices, k):
        """
            Return all the edges for a defined regular ring lattice.
            Args:
                vertices (list<int>) : list of vertices.
                k (int) : each vertex is connected to its k nearest neighbors.
            Returns:
                generator : A generator containing all the edges of the regular ring lattice.
        """
        halfk = k//2
        n = len(vertices)
        for i, u in enumerate(vertices):
            for j in range(i+1, i+halfk+1):
                v = vertices[j % n] #edges for "next" neighbors
                yield [u, v]



    @classmethod
    def ring_lattice(self, n, k):
        """
        Constructs a regular ring lattice.
        Args:
            n (int) : number of vertices.
            k (int) : each vertex is connected to its k nearest neighbors.
        Returns:
            Graph : A regular ring lattice.
        """
        assert k%2 == 0, "k (number of nearest neighbors) must be an even number."
        assert k<=n, "k (number of nearest neighbors) must be equal or less than n (number of vertices)."

        vertices = list(range(n))
        edges = list(self.ring_lattice_edges(vertices, k))
        print(type(self.ring_lattice_edges(vertices, k)))
        assert len(edges) == n*k/2, "Incorrect number of edges."
        return self.from_lists(vertices,edges) # To be improved



    # @classmethod
    # def watts_strogatz(cls, n, k, p):
    #     """
    #     Constructs an instance of the class using Watts-Strogatz model.
    #     Args:
    #         n (int)   : number of vertices.
    #         k (int)   : each vertex is connected to its k nearest neighbors.
    #         p (float) : probability to rewire any edge. Must be between 0 and 1.
    #
    #     Returns:
    #         Graph : A instance of the class.
    #     """
    #     assert p>=0 and p<=1, "The probability p must be between 0 and 1."
    #
    #     G = cls.ring_lattice(n,k)
    #     nb_edges = int(G.get_nb_edges())
    #     halfk = k // 2
    #     # print(halfk)
    #     halfk = int(halfk)
    #     for lap in range(halfk):
    #         print(lap, nb_edges, halfk)
    #         for i in range (lap, nb_edges, halfk):
    #             current_vertex = G.edges[i][0] G.all.get()
    #             if np.random.binomial(1, p):
    #                 choice_set = set(G.vertices) - set(G.get_neighbors(current_vertex)) - {current_vertex}
    #                 if choice_set:
    #                     new_extremity = random.choice(list(choice_set))
    #                     G.edges[i] = [current_vertex,new_extremity]
    #     return G


#===============================================================================
# TOOLS
#===============================================================================

    def remove_vertex(self, vertex):
        # , set_other = self.switcher(vertex_set)
        # if vertex_set == self.top:
        vertex_neighbors = self.get_neighbors(vertex,self.all)
        vertex_set = vertex_set.pop(vertex, None)

        for neighbor in vertex_neighbors:
            self.all[neighbor].remove(vertex)


    def add_vertex(self, vertex, vertex_neighbors):
        self.all[vertex] = vertex_neighbors

        for neighbor in vertex_neighbors:
            self.all[neighbor].add(vertex)


    def add_edge(self, vertex1, vertex2):
        self.all[vertex1].add(vertex2)
        self.all[vertex2].add(vertex1)


    def depth_first_search(self, vertex, cc_vertex, visited): # Improvement : to rename
        """
        Computes depth-first search (DFS) algorithm starting from a vertex.
        """
        visited.append(vertex)
        cc_vertex.append(vertex)

        #print(cc_vertex)
        for neighbor in self.get_neighbors(vertex,self.all):
            if neighbor not in visited:
                cc_vertex = self.depth_first_search(neighbor, cc_vertex, visited)
        return cc_vertex


    def bfs(self, source,set1):
        """
        Traverse the graph using Breadth First Search (BFS) algorithm.
        Args:
            source : the starting vertex ID
            set1   : the starting vertex set #Improvement: parameter can be removed
        Returns:
            level (dict<_,int>) : distances of each vertex to the source vertex.
                                  Each element of the dictionary is a pair vertex_ID:distance
            parent (dict<_,int>) : parent of each vertex (considering the source vertex).
        """
        queue = deque([source]) # Doubly Ended Queue for BFS
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


    def get_bfs_all(self,set1):
        BfsInfo = namedtuple('BfsInfo',["level","parent"])

        all_bfs_set1 = defaultdict(BfsInfo)
        for elt in set1:
            level,parent = self.bfs(elt,set1)
            all_bfs_set1[elt] = BfsInfo(level,parent)

        return all_bfs_set1


#===============================================================================
# ANALYSIS
#===============================================================================


    # def get_clustering_coeff_local(self, vertex, set1):
    #     CCV_all = [] #list to store all local CC
    #
    #     for vertex, kv in self.get_all_degrees(set1).items():
    #         if kv > 1:
    #             neighbors = self.get_neighbors(vertex,set1)
    #             nv = sum( len(self.get_neighbors(neighbor,set1) & neighbors) for neighbor in neighbors)/2 #nb edges between neighbors
    #             CC_local = (2*nv)/(kv*(kv-1))
    #             CCV_all.append(CC_local)
    #
    #     return np.mean(CCV_all)
    #
    #     kv = self.get_all_degrees(set1).get(vertex)
    #     if kv > 1:
    #     CC_local = (2*nv)/(kv*(kv-1))

    def get_clustering_coeff2(self,set1):
        """
        Old implemention of get_clustering_coefficient (Deprecated)
        """

        CCV_all = [] #list to store all local CC

        for vertex, kv in self.get_all_degrees(set1).items():
            if kv > 1:
                neighbors = self.get_neighbors(vertex,set1)
                nv = sum( len(self.get_neighbors(neighbor,set1) & neighbors) for neighbor in neighbors)/2 #nb edges between neighbors
                CC_local = (2*nv)/(kv*(kv-1))
                CCV_all.append(CC_local)

        return np.mean(CCV_all)


    def get_clustering_coeff(self,set1):
        """
        Returns the global clustering coefficient CC(G) of a graph G.
        For all v vertices, where degree(v) > 1:
            CC(v) = (2*Nv) / (Kv*(Kv-1))
            With : Nv the number of edges between neighbors of v
                   Kv the degree of v
                   Kv*(Kv-1) the max number of possible interconnections between neighbors of v
        CC(G) is the mean of all Cc(v)
        """
        return np.mean(self.get_clustering_coeff_all(set1))

    def get_clustering_coeff_local(self, vertex, set1):

        kv = self.get_degree(vertex, set1)
        if kv > 1:
            neighbors = self.get_neighbors(vertex,set1)
            nv = sum( len(self.get_neighbors(neighbor,set1) & neighbors) for neighbor in neighbors)/2 #nb edges between neighbors
            CC_local = (2*nv)/(kv*(kv-1))
            return CC_local


    def get_clustering_coeff_all(self, set1):
        """
        Computes the local clustering coefficient for every vertex of the graph.
        Returns:
            CCV_all (list<float>): list of all local clustering coefficient of the graph.

        """
        # Improvement : return a dictionnary vertex_ID:local_CC
        CCV_all = [] #list to store all local CC
        for vertex in set1.keys():
            CC_vertex = self.get_clustering_coeff_local(vertex, set1)
            # print(CC_vertex)
            if  CC_vertex is not None:
                CCV_all.append(CC_vertex)
        return CCV_all


        # for vertex, kv in self.get_all_degrees(set1).items():
        #     if kv > 1:
        #         neighbors = self.get_neighbors(vertex,set1)
        #         nv = sum( len(self.get_neighbors(neighbor,set1) & neighbors) for neighbor in neighbors)/2 #nb edges between neighbors
        #         CC_local = (2*nv)/(kv*(kv-1))
        #         CCV_all.append(CC_local)

        # return np.mean(CCV_all)


    def get_diameter(self,set1,bfs_all = None):
        """
        Returns the diameter of the graph.
        The diameter is the longest shortest path between any pair of vertices of the graph.
        """
        max_global = float('-inf')
        for vertex in set1:
            if bfs_all:
                distances = bfs_all[vertex].level
            else:
                distances,_ = self.bfs(vertex,set1)

            max_local = max(distances.values())
            if max_global < max_local:
                max_global = max_local
        return max_global


    def get_path_length(self,set1, bfs_all=None):
        """
        Returns the mean path length of the graph.
        The path length is the number of edges in the shortest path between 2 vertices,
        averaged over all pairs of vertices.
        """
        distances_sum = 0
        nb_elts = 0
        for elt in set1:
            if bfs_all:
                distances = bfs_all[elt].level
            else:
                distances,_ = self.bfs(elt,set1)
            # distances,_ = self.bfs(elt,set1)
            distances_sum += sum(distances.values())
            nb_elts += len(distances) -1 # distance to current element = 0

        return distances_sum/nb_elts



    @staticmethod
    def get_symmetric_edges(edges):
        """
        Returns the symmetric edges of the graph.
        """
        return [edge[::-1] for edge in edges]


    def get_assortativity(self,set1):
        """
        Returns the assortativity coefficient of the graph.
        This coefficient is the Pearson correlation coefficient of degree between pairs of linked nodes.
        """
        edges_degree = []
        for vertex, neighbors in set1.items():
            vertex_degree = self.get_degree(vertex,set1)
            for neighbor in neighbors:
                edges_degree.append([vertex_degree, self.get_degree(neighbor,set1)])

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


    def get_density(self):
        nb_edges    = self.get_nb_edges()
        nb_vertices = self.get_nb_vertices()
        return (2*nb_edges)/(nb_vertices*(nb_vertices-1))


    def get_nb_vertices(self):
        return len(self.all.keys())


    def get_nb_edges(self):
        return len(list(chain.from_iterable(self.all.values())))/2 #divided by 2 because undirected graph


    def get_connected_components(self):
        """
        Returns a list of list containing the connected components.
        Each element is a connected component X. Each sub-element is a vertex that belongs to the connected component X.
        [ [CC1_vertex1, CC1_vertex2, CC2_vertex3], [CC2_vertex1, CC2_vertex2]]
        """
        visited = []
        cc_all = []

        for vertex in self.all:
            if vertex not in visited:
                cc_vertex = []
                cc_all.append(self.depth_first_search(vertex, cc_vertex, visited))
        return cc_all


    def analyze(self,set):
        start_time = time.time()
        # self.get_degree_distribution(set)

        df = pd.DataFrame(columns = {"value":""})

        # print("get_bfs_all")
        # bfs_all_set = self.get_bfs_all(set)

        df.loc["nb_vertices"] = self.get_nb_vertices()
        df.loc["nb_edges"] = self.get_nb_edges()
        df.loc["density"] = self.get_density()
        print("before CC")
        df.loc["clustering_coeff"] = self.get_clustering_coeff(set)
        print("after CC")
        df.loc["nb_connected_components"] = len(self.get_connected_components())
        print("before diameter")

        df.loc["diameter"] = self.get_diameter(set)
        # df.loc["diameter"] = self.get_diameter(set, bfs_all = bfs_all_set)
        print("after diameter")

        df.loc["Assortativity"] = self.get_assortativity(set)
        print("before path length")

        df.loc["path_length"] = self.get_path_length(set)
        # df.loc["path_length"] = self.get_path_length(set,bfs_all = bfs_all_set)
        print("after path length")

        df.loc["degree_avg"] = self.get_degree_mean(set)
        df.loc["degree_sum"] = self.get_degrees_sum(set)
        df.loc["degree_min"] = self.get_degree_min(set)
        df.loc["degree_max"] = self.get_degree_max(set)

        print("--- %s seconds ---" % (time.time() - start_time))

        return df,None #bfs_all_set



    #
    # def get_connected_components(self):
    #     # self.super().get_connected_components(self.all, self.all)
    #     super(SimpleGraph, self).get_connected_components(self.all, self.all)


        # @classmethod
        # def watts_strogatz(self, n, k, p):
        #     """
        #     Constructs an instance of the class using Watts-Strogatz model.
        #     Args:
        #         n (int)   : number of vertices.
        #         k (int)   : each vertex is connected to its k nearest neighbors.
        #         p (float) : probability to rewire any edge. Must be between 0 and 1.
        #
        #     Returns:
        #         Graph : A instance of the class.
        #     """
        #     assert p>=0 and p<=1, "The probability p must be between 0 and 1."
        #
        #     G = self.ring_lattice(n,k)
        #     nb_edges = len(G.edges)
        #     halfk = k//2
        #
        #     for lap in range(halfk):
        #         for i in range (lap, nb_edges, halfk):
        #             current_vertex = G.edges[i][0]
        #             if np.random.binomial(1, p):
        #                 choice_set = set(G.vertices) - set(G.get_neighbors(current_vertex)) - {current_vertex}
        #                 if choice_set:
        #                     new_extremity = random.choice(list(choice_set))
        #                     G.edges[i] = [current_vertex,new_extremity]
        #     return G
        #
        #
        #
        # def plot_nx(self, circular=False, node_size = 400, figsize=(5,4)):
        #     """Plots the graph using NetworkX function."""
        #     G=nx.Graph()
        #     G.add_nodes_from(self.vertices)
        #     G.add_edges_from(self.edges)
        #     if circular==True:
        #         pos=nx.circular_layout(G)
        #         pylab.figure(3,figsize=figsize)
        #         nx.draw(G, pos, with_labels=True, node_color='y', edge_color='#909090', node_size=node_size)
        #         pylab.show()
        #         #nx.draw_circular(G, node_color='y', edge_color='#909090', node_size=500, with_labels=True)
        #     else:
        #         nx.draw(G, node_color='y', edge_color='#909090',with_labels=True)
        #     plt.show()

        # def try(self):
        #     print(graph)
