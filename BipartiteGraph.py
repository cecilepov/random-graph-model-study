from Graph import Graph

class BipartiteGraph(Graph):

    def __init__(self, vertices, edges, top, bottom):
        Graph.__init__(self, vertices,edges)
        self.top = top
        self.bottom = bottom

    def get_projection(self,mode):
        neighbors_dist2 = {}
        if mode == "bottom":
            top_or_bottom = self.bottom
        elif "top":
            top_or_bottom = self.top

        for vertex in top_or_bottom:
            edges_projection = set()
            for neighbor in self.get_neighbors(vertex): #N(v)
                edges_projection.update(self.get_neighbors(neighbor))#N(N(v))
                edges_projection.remove(vertex)
#               print(neighbor , " ", self.get_neighbors(neighbor))
            neighbors_dist2[vertex]= edges_projection

        edges = set()
        for (k,v) in neighbors_dist2.items():
            for vertex in v:
                edges.add((k,vertex))
        edges = list(set(tuple(sorted(p)) for p in edges))
        edges = [list(elt) for elt in edges]

        return Graph(self.bottom, edges)



    def get_neighbors2(self,vertex):
        neighbors1 = self.get_neighbors(vertex)
        neighbors2 = set()
        for neighbor in neighbors1:
            neighbors2.update(self.get_neighbors(neighbor))
            neighbors2.remove(vertex)

        return list(neighbors2)


    def get_redundancy(self, mode, vertex):
        if mode == "bottom":
            top_or_bottom = self.bottom
            other = self.top
            other.remove(vertex)

        elif "top":
            top_or_bottom = self.top
            other = self.bottom
            other.remove(vertex)

        edges_removed = self.edges
        edges_removed = [elt for elt in edges_removed if elt[0] != vertex and elt[1] != vertex]

        if  mode == "bottom":
            G_removed = BipartiteGraph(top_or_bottom+other, edges_removed,other ,top_or_bottom)
        else :
            G_removed = BipartiteGraph(top_or_bottom+other, edges_removed,top_or_bottom,other)

        G_projection_all =  self.get_projection(mode)
        G_projection_removed = G_removed.get_projection(mode)#.remove(vertex))

        vertex_neighbors = self.get_neighbors(vertex)
        nb_neighbors_edges_all      = len([elt for elt in G_projection_all.edges if elt[0] in vertex_neighbors and elt[1] in vertex_neighbors])
        nb_neighbors_edges_removed  = len([elt for elt in G_projection_removed.edges if elt[0] in vertex_neighbors and elt[1] in vertex_neighbors])


        print(nb_neighbors_edges_removed, "/",nb_neighbors_edges_all )
        return nb_neighbors_edges_removed/nb_neighbors_edges_all



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
