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


# Authorship information
__author__  = "Pov Cécile"
__credits__ = ["Tabourier Lionel","Tarissan Fabien"]
__version__ = "1.0"
__date__    = "10/07/2020"
__email__   = "cecile.pov@gmail.com"



class BipartiteGraph(Graph):

    def __init__(self, top, bottom):
        Graph.__init__(self, 2) #Graph.__init__(self, vertices,edges)
        self.top = top
        self.bottom = bottom


#==============================================================================
# CLASS METHODS
#==============================================================================

    @classmethod
    def from_file(cls,filename,separator = ",", start = 0):
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
    def configuration_model(cls, degree_dist_top=None, degree_dist_bottom=None, graph=None):

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
        set1, set_other = self.switcher(set1)

        neighbors1 = set1.get(vertex) #self.get_neighbors(vertex)

        neighbors2 = set()
        for neighbor in neighbors1:
            neighbors2.update(set_other.get(neighbor))
            neighbors2.remove(vertex)
        return set(neighbors2)


    # def add_edge()
    def get_projection(self,setu):
        neighbors_dist2 = {}
        setu, set_other = self.switcher(setu)

        for vertex in setu:
            neighbors2 = self.get_neighbors2(vertex, setu)
            neighbors_dist2[vertex]= neighbors2

        return UnipartiteGraph(neighbors_dist2)


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
        # print("all_nb_edges_between_neighbors",all_nb_edges_between_neighbors)

        # print("BEFORE REMOVING")
        # print(self.top)
        # print(self.bottom)
        self.remove_vertex(vertex,set1)
        G_projection_removed =  self.get_projection(set_other)
        removed_nb_edges_between_neighbors = count_edges(G_projection_removed)
        # print("removed_nb_edges_between_neighbors",removed_nb_edges_between_neighbors)

        # print("BEFORE")
        # print(self.top)
        # print(self.bottom)
        self.add_vertex(vertex,vertex_neighbors, set1)
        # print("AFTER")
        # print(self.top)
        # print(self.bottom)


        if not all_nb_edges_between_neighbors:
            return 1,removed_nb_edges_between_neighbors,all_nb_edges_between_neighbors

        return removed_nb_edges_between_neighbors/all_nb_edges_between_neighbors, removed_nb_edges_between_neighbors,all_nb_edges_between_neighbors




    def get_redundancyB(self,vertex,set1): #(self, mode, vertex):

        set1, set_other = self.switcher(set1)

        start_time = time.time()
        nb_persistant_edges = 0
        nb_combinations = 0
        # combinations = list(itertools.combinations(set1.get(vertex),2))
        for (e1,e2) in itertools.combinations(set1.get(vertex),2):
        # for (e1,e2) in combinations:

            e1_neighbors = set_other.get(e1)
            e2_neighbors = set_other.get(e2)
            inter = e1_neighbors & e2_neighbors
            inter.remove(vertex)
            if len(inter):
                nb_persistant_edges+=1
            nb_combinations+=1

        # vertex_degree = len(set1.get(vertex))
        # nb_combinations = len(list(itertools.combinations(set1.get(vertex),2)))
        print(time.time()- start_time)

        return nb_persistant_edges/nb_combinations, nb_persistant_edges,nb_combinations



    def cc_bullet_pair(self, vertex1, vertex2,set1): # cc(u,v)
        # assert vertex1 and vertex2 in self.bottom or vertex1 and vertex2 in self.top
        assert (vertex1 and vertex2) in set1
        neighbors_vertex1 = self.get_neighbors(vertex1,set1)
        neighbors_vertex2 = self.get_neighbors(vertex2,set1)
        #
        # print(neighbors_vertex1)
        # print(neighbors_vertex2)
        union = list(set(neighbors_vertex1).union(neighbors_vertex2))
        intersection = list(set(neighbors_vertex1) & set(neighbors_vertex2))
        # print("union",len(union))
        # print("inter",len(intersection))
        # print("inter",len(union)/len(intersection))

        return len(intersection)/len(union)


    def cc_bullet(self, vertex,set1): #cc(u)
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
        #

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

        Y = set(self.bottom.keys())
        S = set()
        Q = queue.Queue()

        for yj in Y:
            if len(self.get_neighbors(yj,self.bottom)) >1: #exclude 1-top bicliques
                N_yj = frozenset(self.get_neighbors(yj,self.bottom))
                S.add(N_yj)
                Q.put(N_yj)

        while not Q.empty():
            Sx = Q.get()
            yj_not_in_Sx = Y - self.get_consensus_set(Sx,self.top)
            for yj in yj_not_in_Sx:
                N_yj = self.get_neighbors(yj,self.bottom)
                S_new = Sx.intersection(N_yj)
                if S_new not in S and len(S_new) > 1:
                    S.add(S_new)
                    Q.put(S_new)

        bicliques = set() #C_max
        for Sx in S:
            Sx_consensus = self.get_consensus_set(Sx,self.top)
            if len(Sx_consensus) >1:
                bicliques.add(tuple( [frozenset(Sx), frozenset(Sx_consensus)] ))
        # print("bicliques found",bicliques)
        # if noverlap:
        #     bicliques = self.remove_overlap(bicliques,strategy) # Pattern can be improved
        return bicliques

    @staticmethod
    def remove_overlap(self,bicliques_all, strategy = "random"):
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
            # a = bicliques_found_q.get()
            # print(a)

            current_biclique = (current_top,current_bottom)

            current_edges = set(itertools.product(current_top, current_bottom))

            selected_edges = set(selected.keys())

            overlap_edges = selected_edges.intersection(current_edges)

            if not overlap_edges:
                for edge in current_edges:
                    selected[edge] = current_biclique

        return set(selected.values())



    def get_degree_to_bicliques(self,bicliques):
        # DETECTE LES NOEUDS QUI FONT "PONTS" ENTRE DEUX BICLIQUES
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

        if not max_bicliques:
            max_bicliques = self.find_all_maximal()

        if noverlap:
            max_bicliques = self.remove_overlap(strategy,max_bicliques)

        print("OK")
        #
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



    def analyze(self,set):
        start_time = time.time()
        self.get_degree_distribution(set)

        df = pd.DataFrame(columns = {"value":""})

        df.loc["nb_vertices"] = self.get_nb_vertices(set)
        df.loc["nb_edges"] = self.get_nb_edges()
        df.loc["density"] = self.get_density()

        # df.loc["clustering_coeff"] = self.get_clustering_coeff(set)
        df.loc["nb_connected_components"] = len(self.get_connected_components(self.top,self.bottom))
        # df.loc["diameter"] = self.get_diameter(set)
        # df.loc["Assortativity"] = self.get_assortativity(set)
        # df.loc["path_length"] = self.get_path_length(set)
        df.loc["degree_avg"] = self.get_degree_mean(set)
        df.loc["degree_sum"] = self.get_degrees_sum(set)
        df.loc["degree_min"] = self.get_degree_min(set)
        df.loc["degree_max"] = self.get_degree_max(set)
        print("--- %s seconds ---" % (time.time() - start_time))
        return df




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

#
# from Graph import Graph
# from UnipartiteGraph import UnipartiteGraph
# from copy import deepcopy
# from collections import defaultdict
# from random import shuffle
# import time
# import pandas as pd
# from itertools import chain
# import queue
# import random
# import itertools
# from collections import Counter
#
# from collections import defaultdict
#
#
# # Authorship information
# __author__  = "Pov Cécile"
# __credits__ = ["Tabourier Lionel","Tarissan Fabien"]
# __version__ = "1.0"
# __date__    = "10/07/2020"
# __email__   = "cecile.pov@gmail.com"
#
#
#
# class BipartiteGraph(Graph):
#
#     def __init__(self, top, bottom):
#         Graph.__init__(self, 2) #Graph.__init__(self, vertices,edges)
#         self.top = top
#         self.bottom = bottom
#
#
# #==============================================================================
# # CLASS METHODS
# #==============================================================================
#
#     @classmethod
#     def from_file(cls,filename,separator = ",", start = 0):
#         with open(filename, 'r') as filehandle: # OK
#             lines = filehandle.readlines()
#             bottom = defaultdict(set)
#             top = defaultdict(set)
#
#             for i in range (start, len(lines)):
#                 # print(lines[i])
#                 elts = list(map(int, lines[i].strip().split(separator)))
#                 bottom_vertex = elts[0]
#                 top_vertex = elts[1]
#
#                 bottom[bottom_vertex].add(top_vertex)
#                 top[top_vertex].add(bottom_vertex)
#
#             return cls(bottom,top)
#
#
#     @classmethod
#     def configuration_model(cls, degree_dist_top=None, degree_dist_bottom=None, graph=None):
#
#         if degree_dist_top == None and degree_dist_bottom==None :
#             degree_dist_top = graph.get_all_degrees(graph.top)
#             degree_dist_bottom = graph.get_all_degrees(graph.bottom)
#
#         top_stubs = [k for k,v in degree_dist_top.items() for i in range(v)]
#         bottom_stubs = [k for k,v in degree_dist_bottom.items() for i in range(v)]
#
#         shuffle(bottom_stubs)
#
#         new_top = defaultdict(set)
#         new_bottom = defaultdict(set)
#
#         for i in range (len(top_stubs)):
#             new_top[top_stubs[i]].add(bottom_stubs[i])
#             new_bottom[bottom_stubs[i]].add(top_stubs[i])
#
#         cm_graph = cls(new_top,new_bottom)
#         print(cm_graph.get_nb_edges(),"/",sum(degree_dist_top.values()))
#         return cm_graph
#
#
#
#     def save_graph(self,filename):
#         with open('generated/'+filename+'_top.graph', 'w') as writer:
#             for k,v in self.top.items():
#                 writer.write(str(k))
#                 for elt in v:
#                     writer.write(" ")
#                     writer.write(str(elt))
#                 writer.write("\n")
#
#         with open('generated/'+filename+'_bottom.graph', 'w') as writer:
#             for k,v in self.bottom.items():
#                 writer.write(str(k))
#                 for elt in v:
#                     writer.write(" ")
#                     writer.write(str(elt))
#                 writer.write("\n")
#
#
#
#     @classmethod
#     def read_graph(cls,filetop,filebottom):
#
#         def read_set(filename):
#             with open('generated/'+filename+'.graph', 'r') as filehandle:
#                 lines = filehandle.readlines()
#
#             set1 = defaultdict(set)
#             for line in lines:
#                 words = line.split()
#
#                 k = int(words[0])
#                 v = set(map(lambda i:int(i),words[1:]))
#     #             print(v)
#                 set1[k] = v
#             return set1
#
#         return cls(read_set(filetop),read_set(filebottom))
#
#
# #==============================================================================
# # TOOLS
# #==============================================================================
#
#     # @staticmethod
#     def switcher(self,set_selected):
#         if set_selected == self.top:
#             return self.top, self.bottom
#         elif set_selected == self.bottom:
#             return self.bottom, self.top
#
#
#     # @staticmethod
#     def get_neighbors2(self,vertex, setu):
#         setu, set_other = self.switcher(setu)
#
#         neighbors1 = setu.get(vertex) #self.get_neighbors(vertex)
#
#         neighbors2 = set()
#         for neighbor in neighbors1:
#             neighbors2.update(set_other.get(neighbor))
#             neighbors2.remove(vertex)
#         return set(neighbors2)
#
#
#     # def add_edge()
#     def get_projection(self,setu):
#         neighbors_dist2 = {}
#         setu, set_other = self.switcher(setu)
#
#         for vertex in setu:
#             neighbors2 = self.get_neighbors2(vertex, setu)
#             neighbors_dist2[vertex]= neighbors2
#
#         return SimpleGraph(neighbors_dist2)
#
#
#     def remove_vertex(self, vertex, vertex_set):
#         # , set_other = self.switcher(vertex_set)
#         # if vertex_set == self.top:
#         vertex_set, set_other = self.switcher(vertex_set)
#         vertex_neighbors = self.get_neighbors(vertex,vertex_set)
#         vertex_set = vertex_set.pop(vertex, None)
#
#         for neighbor in vertex_neighbors:
#             set_other[neighbor].remove(vertex)
#
#
#     def add_vertex(self, vertex, vertex_neighbors, vertex_set):
#         vertex_set, set_other = self.switcher(vertex_set)
#         vertex_set[vertex] = vertex_neighbors
#
#         for neighbor in vertex_neighbors:
#             set_other[neighbor].add(vertex)
#
#
#
#     def remove_edge(self,top_extremity, bottom_extremity):
#         self.top[top_extremity].remove(bottom_extremity)
#         self.bottom[bottom_extremity].remove(top_extremity)
#
#
#     def add_edge(self,top_extremity, bottom_extremity):
#         self.top[top_extremity].update(bottom_extremity)
#         self.bottom[bottom_extremity].update(top_extremity)
#
# #==============================================================================
# # ANALYSIS
# #==============================================================================
#
#     def get_redundancy(self,vertex,set): #(self, mode, vertex):
#
#         def count_edges(projection):
#             total_count = 0
#             for k,v in {k: v for k, v in projection.all.items() if k in vertex_neighbors}.items():
#                 total_count += sum(map(lambda x : x in vertex_neighbors, v))
#             return total_count/2
#
#         set, set_other = self.switcher(set)
#         vertex_neighbors = self.get_neighbors(vertex, set)
#
#         G_projection =  self.get_projection(set_other)
#
#         all_nb_edges_between_neighbors = count_edges(G_projection)
#
#         self.remove_vertex(vertex,set)
#         G_projection_removed =  self.get_projection(set_other)
#
#         removed_nb_edges_between_neighbors = count_edges(G_projection_removed)
#
#         self.add_vertex(vertex,vertex_neighbors, set)
#
#         return all_nb_edges_between_neighbors, removed_nb_edges_between_neighbors
#
#
#
#     def cc_bullet_pair(self, vertex1, vertex2): # cc(u,v)
#         assert vertex1 and vertex2 in self.bottom or vertex1 and vertex2 in self.top
#         neighbors_vertex1 = self.get_neighbors(vertex1)
#         neighbors_vertex2 = self.get_neighbors(vertex2)
#         #
#         # print(neighbors_vertex1)
#         # print(neighbors_vertex2)
#         union = list(set(neighbors_vertex1).union(neighbors_vertex2))
#         intersection = list(set(neighbors_vertex1) & set(neighbors_vertex2))
#         # print("union",len(union))
#         # print("inter",len(intersection))
#         # print("inter",len(union)/len(intersection))
#         return len(intersection)/len(union)
#
#
#     def cc_bullet(self, vertex): #cc(u)
#         neighbors2 = self.get_neighbors2(vertex)
#         cc_bullet_pair_sum = 0
#         for neighbor2 in neighbors2:
#             cc_bullet_pair_sum += self.cc_bullet_pair(vertex, neighbor2)
#         return cc_bullet_pair_sum/len(neighbors2)
#
#
#     def cc_bullet_set(self, set):
#         cc_bullet_set_sum = 0
#         for vertex in set:
#             cc_bullet_set_sum += cc_bullet(vertex)
#         return cc_bullet_pair_sum/len(set)
#
#
#     def cc_bullet_graph(self):
#         nb_vertices_top = self.get_nb_vertices(self.top)
#         nb_vertices_bottom = self.get_nb_vertices(self.bottom)
#
#         cc_bullet_top    = self.cc_bullet_set(self.top)
#         cc_bullet_bottom = self.cc_bullet_set(self.bottom)
#
#         return ((nb_vertices_top*cc_bullet_top)+(nb_vertices_bottom*cc_bullet_bottom))/(nb_vertices_top+cc_bullet_bottom)
#
#
#
#     def get_density(self):
#         nb_edges    = self.get_nb_edges()
#         nb_vertices_top = self.get_nb_vertices(self.top)
#         nb_vertices_bottom = self.get_nb_vertices(self.bottom)
#
#         return (nb_edges)/(nb_vertices_top*nb_vertices_bottom)
#
#
#     def get_nb_vertices(self,set):
#         return len(set.keys())
#
#
#     def get_nb_edges(self):
#         return len(list(chain.from_iterable(self.top.values()))) #divided by 2 because undirected graph
#
#
# #==============================================================================
# # COUNT BICLIQUES
# #==============================================================================
#
#
#     def get_consensus_set(self,Sx,set1): # perf can be improved
#         # print(Sx)
#         neighbors_all = list()
#         for elt in Sx:
#             neighbors_vertex = self.get_neighbors(elt,set1)
#             neighbors_all.append(neighbors_vertex)
#
#         # print("NEIGHBORS",neighbors_all)
#         consensus_set = set(set.intersection(*neighbors_all))
#         return consensus_set
#
#
#
#     def find_all_maximal(self, strategy = "random", noverlap=True):
#
#         Y = set(self.bottom.keys())
#         S = set()
#         Q = queue.Queue()
#
#         for yj in Y:
#             if len(self.get_neighbors(yj,self.bottom)) >1: #exclude 1-top bicliques
#                 N_yj = frozenset(self.get_neighbors(yj,self.bottom))
#                 S.add(N_yj)
#                 Q.put(N_yj)
#
#         while not Q.empty():
#             Sx = Q.get()
#             yj_not_in_Sx = Y - self.get_consensus_set(Sx,self.top)
#             for yj in yj_not_in_Sx:
#                 N_yj = self.get_neighbors(yj,self.bottom)
#                 S_new = Sx.intersection(N_yj)
#                 if S_new not in S and len(S_new) > 1:
#                     S.add(S_new)
#                     Q.put(S_new)
#
#         bicliques = set() #C_max
#         for Sx in S:
#             Sx_consensus = self.get_consensus_set(Sx,self.top)
#             if len(Sx_consensus) >1:
#                 bicliques.add(tuple( [frozenset(Sx), frozenset(Sx_consensus)] ))
#         # print("bicliques found",bicliques)
#         if noverlap:
#             bicliques = self.remove_overlap(bicliques,strategy) # Pattern can be improved
#         return bicliques
#
#
#
#     def remove_overlap(self,bicliques_all, strategy = "random"):
#         bicliques_found = list(bicliques_all)
#         selected = dict()
#
#         if strategy == "random":
#             random.shuffle(bicliques_found)
#         elif strategy == "maxtop":
#             bicliques_found = sorted(bicliques_found, key=lambda biclique: len(biclique[0]),reverse = True)
#         elif strategy == "maxbottom":
#             bicliques_found = sorted(bicliques_found, key=lambda biclique: len(biclique[1]),reverse = True)
#         elif strategy == "maxnodes":
#             bicliques_found = sorted(bicliques_found, key=lambda biclique: len(list(chain.from_iterable(biclique))),reverse = True)
#
#         bicliques_found_q = queue.Queue()
#         [bicliques_found_q.put(i) for i in bicliques_found]
#
#         while not bicliques_found_q.empty():
#             current_top, current_bottom = bicliques_found_q.get()
#             current_biclique = (current_top,current_bottom)
#
#             current_edges = set(itertools.product(current_top, current_bottom))
#
#             selected_edges = set(selected.keys())
#
#             overlap_edges = selected_edges.intersection(current_edges)
#
#             if not overlap_edges:
#                 for edge in current_edges:
#                     selected[edge] = current_biclique
#
#         return set(selected.values())
#
#
#
#     def get_degree_to_bicliques(self,bicliques):
#         # DETECTE LES NOEUDS QUI FONT "PONTS" ENTRE DEUX BICLIQUES
#         count_all = Counter({})
#         for biclique in bicliques:
#             count_current = Counter(chain.from_iterable(biclique))
#             count_all.update(count_current)
#
#         return count_all
#
#
#     def get_degree_to_bicliques_dist(self,bicliques, plot = 1): # Improvement : shared code with classic degree distribution function
#         count_all = self.get_degree_to_bicliques(bicliques)
#         distribution = Counter(count_all.values())
#
#         if plot == 1:
#             index_name = "degré vers des bicliques maximales"
#             df_distribution = pd.DataFrame.from_dict(distribution, orient='index').reset_index()
#             df_distribution.rename(columns={'index':index_name,0:'count'}, inplace=True)
#             df_distribution.sort_values(by=index_name,ascending=1, inplace=True) # sort the degree by ascending order
#             ax = df_distribution.plot.bar(x=index_name, y="count", title="Distribution des degrés vers des bicliques maximales", rot=0)
#
#         return distribution
#
#
#
# #==============================================================================
# # TRIPARTITE ENCODING
# #==============================================================================
#
#     @staticmethod
#     def get_bottom(top):
#         bottom = defaultdict(set)
#
#         for k,v in top.items():
#             for elt in v:
#                 bottom[elt].add(k)
#         return bottom
#
#
#     def tripartite_model(self,strategy = "random", noverlap=True):
#
#         max_bicliques = self.find_all_maximal(strategy,noverlap)
#
#         # CONSTRUCT SUB12 AND SUB13
#         top12 = defaultdict(set)
#         top13 = defaultdict(set)
#
#         v1_idx = 0
#         for top,bottom in max_bicliques:
#             top12[v1_idx] = set(top)
#             top13[v1_idx] = set(bottom)
#             v1_idx+=1
#
#         sub12 = BipartiteGraph(top12, self.get_bottom(top12))
#         sub13 = BipartiteGraph(top13, self.get_bottom(top13))
#
#
#         # CONSTRUCT SUB23
#         sub23 = deepcopy(self) # improvement: memory usage
#
#         for k,v in max_bicliques:
#             for top_node in k:
#                 for bottom_node in v:
#                     sub23.remove_edge(top_node,bottom_node)
#
#
#         # CONFIGURATION MODEL ON ALL SUBGRAPHS
#         sub12_cm = BipartiteGraph.configuration_model(sub12.get_all_degrees(sub12.top),
#                                                       sub12.get_all_degrees(sub12.bottom))
#
#         sub13_cm = BipartiteGraph.configuration_model(sub13.get_all_degrees(sub13.top),
#                                                       sub13.get_all_degrees(sub13.bottom))
#
#         sub23_cm = BipartiteGraph.configuration_model(sub23.get_all_degrees(sub23.top),
#                                                       sub23.get_all_degrees(sub23.bottom))
#
#
#         # FINAL GRAPH WITH SUB13 AND SUB12
#         graph_cm_top = defaultdict(set)
#
#         for idx_set1,nodes_set2 in sub12_cm.top.items():
#             nodes_set3 = sub13_cm.top.get(idx_set1)
#             for elt in nodes_set2:
#                 graph_cm_top[elt].update(nodes_set3)
#
#
#         # ADD ALL SUB23 EDGES TO FINAL GRAPH
#         for k,v in sub23_cm.top.items():
#             graph_cm_top[k].update(v)
#
#         bg_cm = BipartiteGraph(graph_cm_top, self.get_bottom(graph_cm_top))
#
#         return bg_cm, max_bicliques
#
#
#
#     def analyze(self,set):
#         start_time = time.time()
#         self.get_degree_distribution(set)
#
#         df = pd.DataFrame(columns = {"value":""})
#
#         df.loc["nb_vertices"] = self.get_nb_vertices(set)
#         df.loc["nb_edges"] = self.get_nb_edges()
#         df.loc["density"] = self.get_density()
#
#         # df.loc["clustering_coeff"] = self.get_clustering_coeff(set)
#         df.loc["nb_connected_components"] = len(self.get_connected_components(self.top,self.bottom))
#         # df.loc["diameter"] = self.get_diameter(set)
#         # df.loc["Assortativity"] = self.get_assortativity(set)
#         # df.loc["path_length"] = self.get_path_length(set)
#         df.loc["degree_avg"] = self.get_degree_mean(set)
#         df.loc["degree_sum"] = self.get_degrees_sum(set)
#         df.loc["degree_min"] = self.get_degree_min(set)
#         df.loc["degree_max"] = self.get_degree_max(set)
#         print("--- %s seconds ---" % (time.time() - start_time))
#         return df
#
#
#
#     # def count_edges_between_neighbors(self,vertex, set):#= default_set):
#     #     set, set_other = self.switcher(set)
#     #
#     #     count = 0
#     #     neighbors_vertex = set.get(vertex)
#     #     print("neighbors_vertex", neighbors_vertex)
#     #     for neighbor in neighbors_vertex:
#     #         count += len(neighbors_vertex & set_other.get(neighbor))
#     #     return count/2 # because undirected graph
#     #
#     #
#     #
#     # #NO CHANGES
#     # def depth_first_search(self, vertex, cc_vertex, visited,set):
#     #     """
#     #     Computes depth-first search (DFS) algorithm starting from a vertex.
#     #     """
#     #     visited.append(vertex)
#     #     cc_vertex.append(vertex)
#     #
#     #     #print(cc_vertex)
#     #     print(self.g)
#     #     for neighbor in self.get_neighbors(vertex,set):
#     #         if neighbor not in visited:
#     #             cc_vertex = self.depth_first_search(neighbor, cc_vertex, visited,set)
#     #     return cc_vertex
#
#
#
#     # def get_redundancy2(self, mode, vertex):
#     #     if mode == "bottom":
#     #         top_or_bottom = self.bottom
#     #         other = self.top
#     #         other.remove(vertex)
#     #
#     #     elif "top":
#     #         top_or_bottom = self.top
#     #         other = self.bottom
#     #         other.remove(vertex)
#     #
#     #     edges_removed = self.edges
#     #     edges_removed = [elt for elt in edges_removed if elt[0] != vertex and elt[1] != vertex]
#     #
#     #     if  mode == "bottom":
#     #         G_removed = BipartiteGraph(top_or_bottom+other, edges_removed,other ,top_or_bottom)
#     #     else :
#     #         G_removed = BipartiteGraph(top_or_bottom+other, edges_removed,top_or_bottom,other)
#     #
#     #     G_projection_all =  self.get_projection(mode)
#     #     G_projection_removed = G_removed.get_projection(mode)#.remove(vertex))
#     #
#     #     vertex_neighbors = self.get_neighbors(vertex)
#     #     nb_neighbors_edges_all      = len([elt for elt in G_projection_all.edges if elt[0] in vertex_neighbors and elt[1] in vertex_neighbors])
#     #     nb_neighbors_edges_removed  = len([elt for elt in G_projection_removed.edges if elt[0] in vertex_neighbors and elt[1] in vertex_neighbors])
#     #
#     #
#     #     print(nb_neighbors_edges_removed, "/",nb_neighbors_edges_all )
#     #     return nb_neighbors_edges_removed/nb_neighbors_edges_all
#     #
#     #



    def get_redundancy(self,vertex,set1): #(self, mode, vertex):
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
