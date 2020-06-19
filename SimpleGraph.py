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

class SimpleGraph(Graph):

    def __init__(self, all):

        Graph.__init__(self, 1)
        self.all = all
        # if self.default_set == None:
        #     self.default_set = self.all



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


#==============================================================================
# CLASS METHODS
#==============================================================================
    @classmethod
    def from_lists(cls,vertices, edges):
        all = {vertice:set() for vertice in vertices}
        all = defaultdict(set,all)

        for extremity1,extremity2 in edges:
            all[extremity1].add(extremity2)
            all[extremity2].add(extremity1)

        return cls(all)



    @classmethod
    def from_file(cls,filename,separator = ",", start = 0):
        with open(filename, 'r') as filehandle: # OK
            lines = filehandle.readlines()
            d = defaultdict(set)

            for i in range (start, len(lines)):
                elts = list(map(int, lines[i].split(separator)))
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
            return self(vertices,edges)



        def analyze(self,set):
            start_time = time.time()
            self.get_degree_distribution(set)
            features = ["nb_vertices",
                        "nb_edges",
                        "clustering_coeff",
                        "connected_components",
                        "diameter",
                        "path_length",
                        # "assortativity",
                        "avg_degree",
                        "degrees_sum",
                        "degree_mean",
                        "degree_min",
                        "degree_max"]


            # df.loc["connected_components"] = len(self.get_connected_components())
            # df.loc["diameter"] = self.get_diameter()
            # df.loc["Assortativity"] = self.get_assortativity()
            # df.loc["path_length"] = self.get_path_length()
            # df.loc["degree_avg"] = self.get_degree_mean()
            # df.loc["degree_sum"] = self.get_degrees_sum()
            # df.loc["degree_min"] = self.get_degree_min()
            # df.loc["degree_max"] = self.get_degree_max()
            # print("--- %s secconds ---" % (time.time() - start_time))
            #
            # return df

            df = pd.DataFrame(columns = {"value":""})

            df.loc["nb_vertices"] = len(set.keys())
            df.loc["nb_edges"] = len(list(chain.from_iterable(set.values())))
            df.loc["clustering_coeff"] = self.get_clustering_coeff(set)
            df.loc["connected_components"] = len(self.get_connected_components(set))
            df.loc["diameter"] = self.get_diameter(set)
            # df.loc["Assortativity"] = self.get_assortativity()
            # print("HERE")
            df.loc["path_length"] = self.get_path_length(set)
            df.loc["degree_avg"] = self.get_degree_mean(set)
            df.loc["degree_sum"] = self.get_degrees_sum(set)
            df.loc["degree_min"] = self.get_degree_min(set)
            df.loc["degree_max"] = self.get_degree_max(set)
            print("--- %s seconds ---" % (time.time() - start_time))

            return df



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
