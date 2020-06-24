from Graph import Graph
from SimpleGraph import SimpleGraph
from copy import deepcopy
from collections import defaultdict
from random import shuffle
import time
import pandas as pd
from itertools import chain


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
                elts = list(map(int, lines[i].split(separator)))
                bottom_vertex = elts[0]
                top_vertex = elts[1]

                bottom[bottom_vertex].add(top_vertex)
                top[top_vertex].add(bottom_vertex)

            return cls(bottom,top)



    @classmethod
    def configuration_model(cls, degree_dist_top, degree_dist_bottom):

        top_stubs = [k for k,v in degree_dist_top.items() for i in range(v)]
        bottom_stubs = [k for k,v in degree_dist_bottom.items() for i in range(v)]

        shuffle(bottom_stubs)

        new_top = defaultdict(set)
        new_bottom = defaultdict(set)

        for i in range (len(top_stubs)):
            new_top[top_stubs[i]].add(bottom_stubs[i])
            new_bottom[bottom_stubs[i]].add(top_stubs[i])

        return cls(new_top,new_bottom)


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
    def get_neighbors2(self,vertex, setu):
        setu, set_other = self.switcher(setu)

        neighbors1 = setu.get(vertex) #self.get_neighbors(vertex)

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

        return SimpleGraph(neighbors_dist2)


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

#==============================================================================
# ANALYSIS
#==============================================================================

    def get_redundancy(self,vertex,set): #(self, mode, vertex):

        def count_edges(projection):
            total_count = 0
            for k,v in {k: v for k, v in projection.all.items() if k in vertex_neighbors}.items():
                total_count += sum(map(lambda x : x in vertex_neighbors, v))
            return total_count/2

        set, set_other = self.switcher(set)
        vertex_neighbors = self.get_neighbors(vertex, set)

        G_projection =  self.get_projection(set_other)

        all_nb_edges_between_neighbors = count_edges(G_projection)

        self.remove_vertex(vertex,set)
        G_projection_removed =  self.get_projection(set_other)

        removed_nb_edges_between_neighbors = count_edges(G_projection_removed)

        self.add_vertex(vertex,vertex_neighbors, set)

        return all_nb_edges_between_neighbors, removed_nb_edges_between_neighbors



    def cc_bullet_pair(self, vertex1, vertex2):
        assert vertex1 and vertex2 in self.bottom or vertex1 and vertex2 in self.top
        neighbors_vertex1 = self.get_neighbors(vertex1)
        neighbors_vertex2 = self.get_neighbors(vertex2)
        #
        # print(neighbors_vertex1)
        # print(neighbors_vertex2)
        union = list(set(neighbors_vertex1).union(neighbors_vertex2))
        intersection = list(set(neighbors_vertex1) & set(neighbors_vertex2))
        # print("union",len(union))
        # print("inter",len(intersection))
        # print("inter",len(union)/len(intersection))
        return len(intersection)/len(union)



    def cc_bullet(self, vertex):
        neighbors2 = self.get_neighbors2(vertex)
        cc_bullet_pair_sum = 0
        for neighbor2 in neighbors2:
            cc_bullet_pair_sum += self.cc_bullet_pair(vertex, neighbor2)
        return cc_bullet_pair_sum/len(neighbors2)


    def analyze(self,set):
        pass








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




    def analyze(self,set):
        start_time = time.time()
        self.get_degree_distribution(set)


        df = pd.DataFrame(columns = {"value":""})

        df.loc["nb_vertices_top"] = len(set.keys())
        # df.loc["nb_vertices_top"] = len(set.keys())
        df.loc["nb_edges"] = len(list(chain.from_iterable(set.values())))
        df.loc["clustering_coeff"] = self.get_clustering_coeff(set)
        df.loc["connected_components"] = len(self.get_connected_components(set))
        df.loc["diameter"] = self.get_diameter(set)
        # df.loc["Assortativity"] = self.get_assortativity()
        df.loc["path_length"] = self.get_path_length(set)
        df.loc["degree_avg"] = self.get_degree_mean(set)
        df.loc["degree_sum"] = self.get_degrees_sum(set)
        df.loc["degree_min"] = self.get_degree_min(set)
        df.loc["degree_max"] = self.get_degree_max(set)
        print("--- %s seconds ---" % (time.time() - start_time))

        return df




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
#         # 750170420C16BQZNV
#         0768768187
# huynhalice0320@gmail.com
