from Graph import Graph
from UnipartiteGraph import UnipartiteGraph
from copy import deepcopy
from collections import defaultdict
from random import shuffle
import time
import pandas as pd
from itertools import chain
import queue
import random
import itertools
from collections import Counter
from copy import deepcopy
from collections import defaultdict
from collections import namedtuple


# Authorship information
__author__  = "Pov Cécile"
__credits__ = ["Tabourier Lionel","Tarissan Fabien"]
__version__ = "2.0"
__date__    = "12/10/2020"
__email__   = "cecile.pov@gmail.com"



class BipartiteGraph(Graph):

    """
    This class is a implementation for undirected and unweighted bipartite graphs.
    Edges are only allowed between nodes from 2 distincts sets.
    A bipartite graph is a k-partite graph such that k = 2.
    A graph is a data structure composed of vertices and edges.

    Vertices and edges are represented as adjacency lists (dictionary). Each element is
    a pair vertex_ID:set of neighbors of vertex_ID.

    Attributes:
        top (dict): adjacency list for the top-vertices set
        bottom (dict): adjacency list for the bottom-vertices set

    Example:

    The following bipartite graph G:
    A    B   C     (A,B,C are top vertices)
    |   / \ /
    1  2   3  4    (1,2,3 are top vertices)

    Is represented as:
    G.top    = {"A":{1},
                "B":{2,3},
                "C":{3}}

    G.bottom = {1:{"A"},
                2:{"B"},
                3:{"B","C"},
                4:{}}
    """

    def __init__(self, top, bottom):
        Graph.__init__(self, 2) #Graph.__init__(self, vertices,edges)
        self.top = top
        self.bottom = bottom


#==============================================================================
# CLASS METHODS
#==============================================================================

    @classmethod
    def from_file(cls,filename,separator = ",", start = 0):
        """
        Constructs an instance of the class using a formatted file.
        The file format must be a list of edges that may be preceded by a header/metadata (non formatted).
        Except for the header, each line is formatted as "V1 V2", describing an edge between
        vertex V1 (top set) and V2 (bottom set).

        Args:
            filename  (str)
            separator (str)
            start (int) : first formatted line

        Returns:
            Graph : A instance of the class.
        """
        with open(filename, 'r') as filehandle: # OK
            lines = filehandle.readlines()
            bottom = defaultdict(set)
            top = defaultdict(set)

            for i in range (start, len(lines)):
                # print(lines[i])
                elts = list(map(int, lines[i].strip().split(separator)))
                bottom_vertex = elts[0]
                top_vertex = elts[1]

                bottom[bottom_vertex].add(top_vertex)
                top[top_vertex].add(bottom_vertex)

            return cls(bottom,top)


    @classmethod
    def configuration_model(cls, degree_dist_top=None, degree_dist_bottom=None, graph=None): # Improvement
        """
        Constructs an instance of the class using the bipartite version of Configuration Model.
        Args:
            degree_dist_top (dict<_,int>, optional): degree distribution of top vertices.
                                           Each element of the dictionary is a pair degree:count.
            degree_dist_bottom (dict<_,int>, optional): degree distribution of bottom vertices
            graph (BipartiteGraph, optional): Default to None. The function will take the degree distributions
                                              of the specified graph.

        Returns:
            Graph: A instance of the class.
        """
        if degree_dist_top == None and degree_dist_bottom==None :
            degree_dist_top = graph.get_all_degrees(graph.top)
            degree_dist_bottom = graph.get_all_degrees(graph.bottom)

        top_stubs = [k for k,v in degree_dist_top.items() for i in range(v)]
        bottom_stubs = [k for k,v in degree_dist_bottom.items() for i in range(v)]

        shuffle(bottom_stubs)

        new_top = defaultdict(set)
        new_bottom = defaultdict(set)

        for i in range (len(top_stubs)):
            new_top[top_stubs[i]].add(bottom_stubs[i])
            new_bottom[bottom_stubs[i]].add(top_stubs[i])

        cm_graph = cls(new_top,new_bottom)
        print(cm_graph.get_nb_edges(),"/",sum(degree_dist_top.values()))
        return cm_graph



    def save_graph(self,filename):
        """
        Save the graph in the following formatted file:
        A1 A2
        B1 B2
        With
        Returns:
            Graph: A instance of the class.
        """
        with open('generated/'+filename+'_top.graph', 'w') as writer:
            for k,v in self.top.items():
                writer.write(str(k))
                for elt in v:
                    writer.write(" ")
                    writer.write(str(elt))
                writer.write("\n")

        with open('generated/'+filename+'_bottom.graph', 'w') as writer:
            for k,v in self.bottom.items():
                writer.write(str(k))
                for elt in v:
                    writer.write(" ")
                    writer.write(str(elt))
                writer.write("\n")



    @classmethod
    def read_graph(cls,filetop,filebottom):
        def read_set(filename):
            with open('generated/'+filename+'.graph', 'r') as filehandle:
                lines = filehandle.readlines()

            set1 = defaultdict(set)
            for line in lines:
                words = line.split()

                k = int(words[0])
                v = set(map(lambda i:int(i),words[1:]))
    #             print(v)
                set1[k] = v
            return set1

        return cls(read_set(filetop),read_set(filebottom))


#==============================================================================
# TOOLS
#==============================================================================

    # @staticmethod
    def switcher(self,set_selected):
        if set_selected == self.top:
            return self.top, self.bottom
        elif set_selected == self.bottom:
            return self.bottom, self.top


    # @staticmethod
    def get_neighbors2(self,vertex, set1):
        """
        Returns the neighbors of neighbors of the specified vertex.
        """
        set1, set_other = self.switcher(set1)

        neighbors1 = set1.get(vertex) #self.get_neighbors(vertex)

        neighbors2 = set()
        for neighbor in neighbors1:
            neighbors2.update(set_other.get(neighbor))
            neighbors2.remove(vertex)
        return set(neighbors2)


    # def add_edge()
    def get_projection(self,set1):
        """
        Return the unipartite projection (top or bottom) of the graph
        Arguments:
            set1 (dict): the set of vertices (top or bottom) that will be preserved after the projection
        """
        neighbors_dist2 = {}
        set1, set_other = self.switcher(set1)

        for vertex in set1:
            neighbors2 = self.get_neighbors2(vertex, set1)
            neighbors_dist2[vertex]= neighbors2

        return UnipartiteGraph(neighbors_dist2)


    def depth_first_search(self, vertex, cc_vertex, visited,dico, value):#toggle):
        """
        Computes depth-first search (DFS) algorithm starting from a vertex.
        """
        visited[value].append(vertex)
        cc_vertex[value].append(vertex) #cc_vertex.append(vertex)
        current_set = dico.get(value)

        neighbors = self.get_neighbors(vertex,current_set)
        for neighbor in neighbors:
            new_value = not value
            if neighbor not in visited[new_value]:
                cc_vertex = self.depth_first_search(neighbor, cc_vertex, visited,dico, new_value)

        return cc_vertex


    def remove_vertex(self, vertex, vertex_set):
        # , set_other = self.switcher(vertex_set)
        # if vertex_set == self.top:
        vertex_set, set_other = self.switcher(vertex_set)
        vertex_neighbors = self.get_neighbors(vertex,vertex_set)
        vertex_set = vertex_set.pop(vertex, None)

        for neighbor in vertex_neighbors:
            set_other[neighbor].remove(vertex)


    def add_vertex(self, vertex, vertex_neighbors, vertex_set):
        vertex_set, set_other = self.switcher(vertex_set)
        vertex_set[vertex] = vertex_neighbors

        for neighbor in vertex_neighbors:
            set_other[neighbor].add(vertex)


    def remove_edge(self,top_extremity, bottom_extremity):
        self.top[top_extremity].remove(bottom_extremity)
        self.bottom[bottom_extremity].remove(top_extremity)


    def add_edge(self,top_extremity, bottom_extremity):
        self.top[top_extremity].update(bottom_extremity)
        self.bottom[bottom_extremity].update(top_extremity)

#==============================================================================
# ANALYSIS
#==============================================================================

    #Deprecated - Slower: several projection computation.
    def get_redundancy2(self,vertex,set1): #(self, mode, vertex):

        def count_edges(projection):
            total_count = 0
            for k,v in {k: v for k, v in projection.all.items() if k in vertex_neighbors}.items():
                total_count += sum(map(lambda x : x in vertex_neighbors, v))
            return total_count/2

        set1, set_other = self.switcher(set1)
        vertex_neighbors = self.get_neighbors(vertex, set1)

        G_projection =  self.get_projection(set_other)
        all_nb_edges_between_neighbors = count_edges(G_projection)

        self.remove_vertex(vertex,set1)
        G_projection_removed =  self.get_projection(set_other)
        removed_nb_edges_between_neighbors = count_edges(G_projection_removed)

        self.add_vertex(vertex,vertex_neighbors, set1)

        if not all_nb_edges_between_neighbors:
            return 1,removed_nb_edges_between_neighbors,all_nb_edges_between_neighbors

        return removed_nb_edges_between_neighbors/all_nb_edges_between_neighbors, removed_nb_edges_between_neighbors,all_nb_edges_between_neighbors



    def get_redundancy(self,vertex,set1): #(self, mode, vertex):
        """
        Return the redundancy coefficient of the specified vertex.
        """
        set1, set_other = self.switcher(set1)

        nb_persistant_edges = 0
        nb_combinations = 0
        for (e1,e2) in itertools.combinations(set1.get(vertex),2):
            e1_neighbors = set_other.get(e1)
            e2_neighbors = set_other.get(e2)
            inter = e1_neighbors & e2_neighbors
            inter.remove(vertex)
            if len(inter):
                nb_persistant_edges+=1
            nb_combinations+=1

        if not nb_combinations:
            return 1, nb_persistant_edges,nb_combinations

        return nb_persistant_edges/nb_combinations, nb_persistant_edges,nb_combinations



    def cc_bullet_pair(self, vertex1, vertex2,set1): # cc(u,v)
        # assert vertex1 and vertex2 in self.bottom or vertex1 and vertex2 in self.top
        assert (vertex1 and vertex2) in set1
        neighbors_vertex1 = self.get_neighbors(vertex1,set1)
        neighbors_vertex2 = self.get_neighbors(vertex2,set1)

        union = list(set(neighbors_vertex1).union(neighbors_vertex2))
        intersection = list(set(neighbors_vertex1) & set(neighbors_vertex2))

        return len(intersection)/len(union)


    def cc_bullet(self, vertex,set1): #cc(u)
        """
        Computes the bipartite clustering coefficient of a vertex.
        """
        neighbors2 = self.get_neighbors2(vertex,set1)
        if len(neighbors2) == 0:
            return None # clustering coefficient is not defined for nodes v such that N(N(v)) = Ø

        cc_bullet_pair_sum = 0
        for neighbor2 in neighbors2:
            cc_bullet_pair_sum += self.cc_bullet_pair(vertex, neighbor2,set1)
        return cc_bullet_pair_sum/len(neighbors2)


    def cc_bullet_set(self, set1):
        # cc_bullet_set_sum = 0
        # for vertex in set1:
        #     cc_bullet_set_sum += self.cc_bullet(vertex,set1)
        # return cc_bullet_pair_sum/len(set1)

        cc_bullet_set_sum = 0
        nb_vertices = 0
        for vertex in set1:
            cc_bullet_vertex = self.cc_bullet(vertex,set1)
            if cc_bullet_vertex:
                cc_bullet_set_sum += cc_bullet_vertex
                nb_vertices += 1
        return cc_bullet_set_sum/nb_vertices

    def cc_bullet_graph(self):
        nb_vertices_top = self.get_nb_vertices(self.top)
        nb_vertices_bottom = self.get_nb_vertices(self.bottom)

        cc_bullet_top    = self.cc_bullet_set(self.top)
        cc_bullet_bottom = self.cc_bullet_set(self.bottom)

        return ((nb_vertices_top*cc_bullet_top)+(nb_vertices_bottom*cc_bullet_bottom))/(nb_vertices_top+cc_bullet_bottom)



    def get_density(self):
        nb_edges    = self.get_nb_edges()
        nb_vertices_top = self.get_nb_vertices(self.top)
        nb_vertices_bottom = self.get_nb_vertices(self.bottom)

        return (nb_edges)/(nb_vertices_top*nb_vertices_bottom)


    def get_nb_vertices(self,set):
        return len(set.keys())


    def get_nb_edges(self):
        return len(list(chain.from_iterable(self.top.values()))) #divided by 2 because undirected graph




    def get_connected_components(self,set1,set2):
        """
        Returns a list of list containing the connected components.
        [ [CC1_vertex1, CC1_vertex2, CC2_vertex3], [CC2_vertex1, CC2_vertex2]]
        """
        dico  = {True:set1, False:set2}
        visited = defaultdict(list)
        cc_all = []
        connectedComponent = namedtuple('connectedComponent', ['top', 'bottom'])

        for vertex in set1:
            if vertex not in visited[True]:
                cc_vertex = defaultdict(list)
                self.depth_first_search(vertex, cc_vertex, visited, dico,True)
                #cc_vertex["top"] = cc_vertex.pop(True)
                #cc_vertex["bottom"] = cc_vertex.pop(False)
                cc_vertex = connectedComponent(frozenset(cc_vertex[True]), frozenset(cc_vertex[False]))#cc_vertex = connectedComponent(frozenset(elt["top"]), frozenset(elt["bottom"]))
                cc_all.append(cc_vertex)


        for vertex in set2: # Loop to detect singleton in set2
            if vertex not in visited[False]:
                #print(vertex)
                cc_vertex = defaultdict(list)
                self.depth_first_search(vertex, cc_vertex, visited, dico,False)
                #cc_vertex["top"] = cc_vertex.pop(True)
                #cc_vertex["bottom"] = cc_vertex.pop(False)
                cc_vertex = connectedComponent(frozenset(cc_vertex[True]), frozenset(cc_vertex[False]))#cc_vertex = connectedComponent(frozenset(elt["top"]), frozenset(elt["bottom"]))
                cc_all.append(cc_vertex)

        return cc_all



#==============================================================================
# COUNT BICLIQUES
#==============================================================================


    def get_consensus_set(self,Sx,set1): # perf can be improved
        # print(Sx)
        neighbors_all = list()
        for elt in Sx:
            neighbors_vertex = self.get_neighbors(elt,set1)
            neighbors_all.append(neighbors_vertex)

        # print("NEIGHBORS",neighbors_all)
        consensus_set = set(set.intersection(*neighbors_all))
        return consensus_set


    def find_all_maximal(self):#, strategy = "random", noverlap=True):
        """
        Find all the maximal bicliques of the graph based on E.Kayaaslan article.
        https://www.researchgate.net/publication/221394877_On_Enumerating_All_Maximal_Bicliques_of_Bipartite_Graphs
        """
        Y = set([k for k,v in self.bottom.items() if len(v) > 0]) #set(self.bottom.keys())
        S = set()
        Q = queue.Queue()

        for yj in Y: # initialize S and Q with list of neighbors of every bottom node
            if len(self.get_neighbors(yj,self.bottom)) > 1: #exclude 1-top bicliques and empty neighbourhood
                N_yj = frozenset(self.get_neighbors(yj,self.bottom))
                S.add(N_yj)
                Q.put(N_yj)

        while not Q.empty():
            Sx = Q.get()
            consensus_Sx = self.get_consensus_set(Sx,self.top)
            yj_not_in_consensus_Sx = Y - consensus_Sx # all bottom nodes not in the consensus set
            for yj in yj_not_in_consensus_Sx:
                N_yj = self.get_neighbors(yj,self.bottom)
                S_new = Sx.intersection(N_yj)
                if S_new not in S and len(S_new) > 1: #exclude 1-top bicliques
                    S.add(S_new)
                    Q.put(S_new)

        bicliques = set() #C_max
        for Sx in S:
            Sx_consensus = self.get_consensus_set(Sx,self.top)
            if len(Sx_consensus) >1:  #exclude 1-bottom bicliques and bottom singletons
                bicliques.add(tuple( [frozenset(Sx), frozenset(Sx_consensus)] ))
        # print("bicliques found",bicliques)
        # if noverlap:
        #     bicliques = self.remove_overlap(bicliques,strategy) # Pattern can be improved
        return bicliques

    #@staticmethod
    def remove_overlap(self,bicliques_all, strategy = "random"):
        """
        Returns a set of non-overlapping maximal bicliques, according to a particular heuristic.
        Two overlapping maximal bicliques share at least one edge.
        Arguments:
            bicliques_all (set): a set of maximal bicliques
            strategy (str, optional): heuristic to remove
        """
        bicliques_found = list(bicliques_all)
        selected = dict()

        if strategy == "random":
            random.shuffle(bicliques_found)
        elif strategy == "maxtop":
            bicliques_found = sorted(bicliques_found, key=lambda biclique: len(biclique[0]),reverse = True)
        elif strategy == "maxbottom":
            bicliques_found = sorted(bicliques_found, key=lambda biclique: len(biclique[1]),reverse = True)
        elif strategy == "maxnodes":
            bicliques_found = sorted(bicliques_found, key=lambda biclique: len(list(chain.from_iterable(biclique))),reverse = True)

        bicliques_found_q = queue.Queue()
        [bicliques_found_q.put(i) for i in bicliques_found]

        while not bicliques_found_q.empty():
            current_top, current_bottom = bicliques_found_q.get()
            current_biclique = (current_top,current_bottom)
            current_edges = set(itertools.product(current_top, current_bottom))
            selected_edges = set(selected.keys())
            overlap_edges = selected_edges.intersection(current_edges)

            if not overlap_edges:
                for edge in current_edges:
                    selected[edge] = current_biclique

        return set(selected.values())



    def get_degree_to_bicliques(self,bicliques):
        # Détecte les noeuds qui font "pont" entre deux bicliques.
        count_all = Counter({})
        for biclique in bicliques:
            count_current = Counter(chain.from_iterable(biclique))
            count_all.update(count_current)

        return count_all


    def get_degree_to_bicliques_dist(self,bicliques, plot = 1): # Improvement : shared code with classic degree distribution function
        count_all = self.get_degree_to_bicliques(bicliques)
        distribution = Counter(count_all.values())

        if plot == 1:
            index_name = "degré vers des bicliques maximales"
            df_distribution = pd.DataFrame.from_dict(distribution, orient='index').reset_index()
            df_distribution.rename(columns={'index':index_name,0:'count'}, inplace=True)
            df_distribution.sort_values(by=index_name,ascending=1, inplace=True) # sort the degree by ascending order
            ax = df_distribution.plot.bar(x=index_name, y="count", title="Distribution des degrés vers des bicliques maximales", rot=0)

        return distribution



#==============================================================================
# TRIPARTITE ENCODING
#==============================================================================

    @staticmethod
    def get_bottom(top):
        bottom = defaultdict(set)

        for k,v in top.items():
            for elt in v:
                bottom[elt].add(k)
        return bottom


    def tripartite_model(self,strategy = "random", noverlap=True, max_bicliques = None):
        """
        Naive tripartite model.
        """
        if not max_bicliques:
            max_bicliques = self.find_all_maximal()

        if noverlap:
            max_bicliques = self.remove_overlap(max_bicliques,strategy=strategy)

        # # if not max_bicliques:
        # max_bicliques = self.find_all_maximal()
        # print(type(max_bicliques))
        # print(max_bicliques)
        # if noverlap:
        #     max_bicliques = self.remove_overlap(max_bicliques, strategy)

        # CONSTRUCT SUB12 AND SUB13
        top12 = defaultdict(set)
        top13 = defaultdict(set)

        v1_idx = 0
        for top,bottom in max_bicliques:
            top12[v1_idx] = set(top)
            top13[v1_idx] = set(bottom)
            v1_idx+=1

        sub12 = BipartiteGraph(top12, self.get_bottom(top12))
        sub13 = BipartiteGraph(top13, self.get_bottom(top13))

        # CONSTRUCT SUB23
        sub23 = deepcopy(self) # improvement: memory usage

        for k,v in max_bicliques:
            for top_node in k:
                for bottom_node in v:
                    sub23.remove_edge(top_node,bottom_node)

        # print("max_bicliques: ", max_bicliques)
        # print("SUB12")
        # print(sub12.top)
        # print(sub12.bottom )
        #
        # print("SUB13")
        # print(sub13.top)
        # print(sub13.bottom )
        #
        # print("SUB23")
        # print(sub23.top)
        # print(sub23.bottom )

        # CONFIGURATION MODEL ON ALL SUBGRAPHS
        sub12_cm = BipartiteGraph.configuration_model(sub12.get_all_degrees(sub12.top),
                                                      sub12.get_all_degrees(sub12.bottom))

        sub13_cm = BipartiteGraph.configuration_model(sub13.get_all_degrees(sub13.top),
                                                      sub13.get_all_degrees(sub13.bottom))

        sub23_cm = BipartiteGraph.configuration_model(sub23.get_all_degrees(sub23.top),
                                                      sub23.get_all_degrees(sub23.bottom))


        # FINAL GRAPH WITH SUB13 AND SUB12
        graph_cm_top = defaultdict(set)

        for idx_set1,nodes_set2 in sub12_cm.top.items():
            nodes_set3 = sub13_cm.top.get(idx_set1)
            for elt in nodes_set2:
                graph_cm_top[elt].update(nodes_set3)


        # ADD ALL SUB23 EDGES TO FINAL GRAPH
        for k,v in sub23_cm.top.items():
            graph_cm_top[k].update(v)

        bg_cm = BipartiteGraph(graph_cm_top, self.get_bottom(graph_cm_top))

        return bg_cm, max_bicliques


#==============================================================================
# TRIPARTITE ENCODING 2ND VERSION
#==============================================================================


    def shuffle_bicliques(self,bicliques): # self not used
    """
    Shuffle bicliques according to their size. (Tripartite model 2)
    """
        def groupby_degree_to_other_set(upper_level):
            stubs_X1 = defaultdict(list)
            stubs_1X = defaultdict(list)

            for k,v in upper_level.items():
                idx = v.degree
                stubs_X1[idx].extend(set(v.vertex))
                stubs_1X[idx].extend([k]*len(v.vertex))

            return stubs_1X, stubs_X1

        def shuffler(stubsA, stubsB):
            assert stubsA.keys() == stubsB.keys()
            for key in stubsA.keys():
                shuffle(stubsA.get(key))
                shuffle(stubsB.get(key))

        def connect_stubs(tab12, tab21):
            rd12 = defaultdict(list)
            for k,v in tab12.items():
                for i in range (len(v)):
                    rd12[v[i]].append(tab21.get(k)[i])

            return rd12

        BicliqueSet = namedtuple('BicliqueSet', ['vertex', 'degree'])

        degree_to_bottom = defaultdict(BicliqueSet) #top12
        degree_to_top = defaultdict(BicliqueSet) #top13
        v1_idx = 0

        for top,bottom in bicliques: # for all bicliques
            degree_to_bottom[v1_idx] = BicliqueSet(top, len(bottom))#set(top)
            degree_to_top[v1_idx] = BicliqueSet(bottom, len(top)) #set(bottom)
            v1_idx+=1

        #Group stubs by degree to other set
        stubs12, stubs21 = groupby_degree_to_other_set(degree_to_bottom)
        stubs13, stubs31 = groupby_degree_to_other_set(degree_to_top)

        #Shuffle stubs order
        shuffler(stubs12, stubs21)
        shuffler(stubs13, stubs31)

        #New edges
        new_edges_12 = connect_stubs(stubs12, stubs21)
        new_edges_13 = connect_stubs(stubs13, stubs31)

        # -------
        new_edges = [new_edges_12, new_edges_13]
        new_bicliques = {}
        for k in new_edges_12.keys():
            new_bicliques[k] = tuple(d[k] for d in new_edges)

        # ------
        # Bipartite version
        bipartite_proj_top = defaultdict(set)
        for k,v in new_edges_12.items():
            for elt in v:
                bipartite_proj_top[elt].update(new_edges_13.get(k))


        print("degree_to_bottom",degree_to_bottom)
        print("degree_to_top",degree_to_top)

        print("stubs12",stubs12)
        print("stubs21",stubs21)
        print("stubs13",stubs13)
        print("stubs31",stubs31)

        print("new_edges_12",new_edges_12)
        print("new_edges_13",new_edges_13)
        print("new_edges",new_edges)
        print("new_graph",bipartite_proj_top)

        new_graph = BipartiteGraph(bipartite_proj_top,BipartiteGraph.get_bottom(bipartite_proj_top) )
        return new_graph


    def shuffle_not_bicliques(self, max_bicliques):
        subgraph23 = deepcopy(self) # improvement: memory usage

        for k,v in max_bicliques:
            for top_node in k:
                for bottom_node in v:
                    subgraph23.remove_edge(top_node,bottom_node)

        subgraph23_cm = BipartiteGraph.configuration_model(subgraph23.get_all_degrees(subgraph23.top),
                                                      subgraph23.get_all_degrees(subgraph23.bottom))

        return subgraph23_cm


    def tripartite_model_2(self,strategy = "random", noverlap=True, max_bicliques = None):
        """
        Tripartite model as proposed by F.Tarissan and L.Tabourier.
        https://hal.archives-ouvertes.fr/hal-01211186/document
        """
        if not max_bicliques:
            max_bicliques = self.find_all_maximal()

        if noverlap:
            max_bicliques = self.remove_overlap(max_bicliques,strategy=strategy)

        #Shuffle non Biclique
        sub23_cm = self.shuffle_not_bicliques(max_bicliques)

        #Shuffle bicliques
        bicliques_tri = self.shuffle_bicliques(max_bicliques)

        # ADD ALL SUB23 EDGES TO FINAL GRAPH
        for k,v in sub23_cm.top.items():
            bicliques_tri.top[k].update(v)

        bg_cm = BipartiteGraph(bicliques_tri.top, self.get_bottom(bicliques_tri.top))

        return bg_cm, max_bicliques



#==============================================================================
# DAMASHCKE
#==============================================================================

    def get_phi(self, G, set1):

        set1, set_other = self.switcher(set1)

        sigma_G = self.get_consensus_set(G, set1)
        phi_G   = self.get_consensus_set(sigma_G, set_other)

        return phi_G



    def extension(self, G, set1):
        phi_G = self.get_phi(G, set1)

        max_id = 0
        for elt in G:
            id_elt = list(set1).index(elt)
            if  id_elt > max_id: max_id = id_elt


        g_candidates = list(set1)[max_id:]
        for g in g_candidates:
            phi_G_union_g = self.get_phi(G.union({g}), set1)
            g_index = list(set1).index(g)

            substract = phi_G_union_g - phi_G
            substract_indices = [list(set1).index(s) for s in substract]

            if all(i>g_index for i in substract_indices):
                G.union({g})


    def foo(self, str): # test
        print(self.__dict__[str])


    def analyze(self,set,withcc = True):
        start_time = time.time()
        # self.get_degree_distribution(set)

        df = pd.DataFrame(columns = {"value":""})

        df.loc["nb_vertices"] = self.get_nb_vertices(set)
        df.loc["nb_edges"] = self.get_nb_edges()
        df.loc["density"] = self.get_density()
        df.loc["cc_bullet_set"] = self.cc_bullet_set(set)

        if withcc == True:
            connected_components = self.get_connected_components(self.top,self.bottom)
            df.loc["nb_connected_components"] = len(connected_components)
        else:
            connected_components = None
        # df.loc["diameter"] = self.get_diameter(set)
        # df.loc["Assortativity"] = self.get_assortativity(set)
        # df.loc["path_length"] = self.get_path_length(set)
        df.loc["degree_avg"] = self.get_degree_mean(set)
        df.loc["degree_sum"] = self.get_degrees_sum(set)
        df.loc["degree_min"] = self.get_degree_min(set)
        df.loc["degree_max"] = self.get_degree_max(set)

        print("--- %s seconds ---" % (time.time() - start_time))
        return df, connected_components #connected_components




    # def get_redundancy_old(self,vertex,set1): #(self, mode, vertex):
    #
    #     def count_edges(projection):
    #         total_count = 0
    #         for k,v in {k: v for k, v in projection.all.items() if k in vertex_neighbors}.items():
    #             total_count += sum(map(lambda x : x in vertex_neighbors, v))
    #         return total_count/2
    #
    #     set1, set_other = self.switcher(set1)
    #     vertex_neighbors = self.get_neighbors(vertex, set1)
    #
    #     G_projection =  self.get_projection(set_other)
    #     all_nb_edges_between_neighbors = count_edges(G_projection)
    #     print("all_nb_edges_between_neighbors",all_nb_edges_between_neighbors)
    #
    #     print("BEFORE REMOVING")
    #     print(self.top)
    #     print(self.bottom)
    #     self.remove_vertex(vertex,set1)
    #     G_projection_removed =  self.get_projection(set_other)
    #     removed_nb_edges_between_neighbors = count_edges(G_projection_removed)
    #     print("removed_nb_edges_between_neighbors",removed_nb_edges_between_neighbors)
    #
    #     print("BEFORE")
    #     print(self.top)
    #     print(self.bottom)
    #     self.add_vertex(vertex,vertex_neighbors, set1)
    #     print("AFTER")
    #     print(self.top)
    #     print(self.bottom)
    #
    #
    #     if not all_nb_edges_between_neighbors:
    #         return 1,removed_nb_edges_between_neighbors,all_nb_edges_between_neighbors
    #
    #     return removed_nb_edges_between_neighbors/all_nb_edges_between_neighbors, removed_nb_edges_between_neighbors,all_nb_edges_between_neighbors



    # def count_edges_between_neighbors(self,vertex, set):#= default_set):
    #     set, set_other = self.switcher(set)
    #
    #     count = 0
    #     neighbors_vertex = set.get(vertex)
    #     print("neighbors_vertex", neighbors_vertex)
    #     for neighbor in neighbors_vertex:
    #         count += len(neighbors_vertex & set_other.get(neighbor))
    #     return count/2 # because undirected graph
    #
    #
    #
    # #NO CHANGES
    # def depth_first_search(self, vertex, cc_vertex, visited,set):
    #     """
    #     Computes depth-first search (DFS) algorithm starting from a vertex.
    #     """
    #     visited.append(vertex)
    #     cc_vertex.append(vertex)
    #
    #     #print(cc_vertex)
    #     print(self.g)
    #     for neighbor in self.get_neighbors(vertex,set):
    #         if neighbor not in visited:
    #             cc_vertex = self.depth_first_search(neighbor, cc_vertex, visited,set)
    #     return cc_vertex



    # def get_redundancy2(self, mode, vertex):
    #     if mode == "bottom":
    #         top_or_bottom = self.bottom
    #         other = self.top
    #         other.remove(vertex)
    #
    #     elif "top":
    #         top_or_bottom = self.top
    #         other = self.bottom
    #         other.remove(vertex)
    #
    #     edges_removed = self.edges
    #     edges_removed = [elt for elt in edges_removed if elt[0] != vertex and elt[1] != vertex]
    #
    #     if  mode == "bottom":
    #         G_removed = BipartiteGraph(top_or_bottom+other, edges_removed,other ,top_or_bottom)
    #     else :
    #         G_removed = BipartiteGraph(top_or_bottom+other, edges_removed,top_or_bottom,other)
    #
    #     G_projection_all =  self.get_projection(mode)
    #     G_projection_removed = G_removed.get_projection(mode)#.remove(vertex))
    #
    #     vertex_neighbors = self.get_neighbors(vertex)
    #     nb_neighbors_edges_all      = len([elt for elt in G_projection_all.edges if elt[0] in vertex_neighbors and elt[1] in vertex_neighbors])
    #     nb_neighbors_edges_removed  = len([elt for elt in G_projection_removed.edges if elt[0] in vertex_neighbors and elt[1] in vertex_neighbors])
    #
    #
    #     print(nb_neighbors_edges_removed, "/",nb_neighbors_edges_all )
    #     return nb_neighbors_edges_removed/nb_neighbors_edges_all
    #
    #



    def get_redundancy_old(self,vertex,set1): #(self, mode, vertex):
        # version avec nb_combinations en len(list(generator))
        set1, set_other = self.switcher(set1)

        start_time = time.time()
        nb_persistant_edges = 0
        # nb_combinations = 0
        # combinations = list(itertools.combinations(set1.get(vertex),2))
        for (e1,e2) in itertools.combinations(set1.get(vertex),2):
        # for (e1,e2) in combinations:

            e1_neighbors = set_other.get(e1)
            e2_neighbors = set_other.get(e2)
            inter = e1_neighbors & e2_neighbors
            inter.remove(vertex)
            if len(inter):
                nb_persistant_edges+=1
            # nb_combinations+=1

        # vertex_degree = len(set1.get(vertex))
        nb_combinations = len(list(itertools.combinations(set1.get(vertex),2)))
        print(time.time()- start_time)

        return nb_persistant_edges/nb_combinations, nb_persistant_edges,nb_combinations
