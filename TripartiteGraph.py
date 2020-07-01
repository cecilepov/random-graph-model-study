from Graph import Graph
from UnipartiteGraph import UnipartiteGraph
from copy import deepcopy
from collections import defaultdict
from random import shuffle
import time
import pandas as pd
from itertools import chain



class TripartiteGraph(Graph):

    def __init__(self, v1, v2, v3):
        Graph.__init__(self, 3) #Graph.__init__(self, vertices,edges)
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3





    def get_projection_23():
        new_v2 = default_dict(set)
        new_v3 = defaultdict(set)

        for elt in self.v2
