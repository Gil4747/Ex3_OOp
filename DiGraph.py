from GraphInterface import  GraphInterface
from nodeData import  nodeData
class DiGraph (GraphInterface):
    edges=0
    nodes = {}
    go = { }
    back = { }
    mc = 0
    max_x = -2147483648.0
    max_y = -2147483648.0
    min_x = 2147483648.0
    min_y = 2147483648.0

    """This abstract class represents an interface of a graph."""
    def __init__ (self,nodes={},go={},back={},mc=0,edges=0,max_x=-2147483648.0,max_y=-2147483648.0,min_x=2147483648.0,min_y=2147483648.0):
        self.edges = edges
        self.nodes = nodes
        self.go = go
        self.back = back
        self.mc=mc
        self.max_x = max_x
        self.max_y = max_y
        self.min_x = min_x
        self.min_y = min_y

    def v_size(self) -> int:
        return self.nodes.__len__()

    def e_size(self) -> int:
        return self.edges

    def get_all_v(self) -> dict:
        return self.nodes

    def all_in_edges_of_node(self, id1: int) -> dict:
        return self.back[id1]

    def all_out_edges_of_node(self, id1: int) -> dict:
        return self.go[id1]

    def get_mc(self) -> int:
        return self.mc

    def add_edge(self, id1: int, id2: int, weight: float) -> bool:
        if self.nodes.__contains__(id1) & self.nodes.__contains__(id2):
            if self.go[id1].__contains__(id2):
                return False
            self.go[id1].update({id2: weight})
            self.back[id2].update({id1: weight})
            self.edges=self.edges+1
            self.mc=self.mc+1
            return True
        return False

    def add_node(self, node_id: int, pos: tuple = None) -> bool:
        if node_id in self.nodes:
            return False
        if(pos is not None):
            if (pos.x < self.min_x):
                self.min_x = pos.x
            if (pos.x > self.max_x):
                self.max_x = pos.x
            if (pos.y < self.min_y):
                self.min_y = pos.y
            if (pos.y > self.max_y):
                self.max_y = pos.y
        n = nodeData(node_id, pos)
        self.nodes.update({node_id:n })
        self.go[node_id] = {}
        self.back[node_id] = {}
        self.mc = self.mc+1
        return True

    def remove_node(self, node_id: int) -> bool:
        if node_id in self.nodes:
            del self.nodes[node_id]
            for n in self.back[node_id]:
                del self.go[n][node_id]
            del self.back[node_id]
            for n in self.go[node_id]:
                del self.back[n][node_id]
            del self.go[node_id]
            self.mc = self.mc + 1
            return True
        return False

    def remove_edge(self, node_id1: int, node_id2: int) -> bool:
        if self.nodes.__contains__(node_id1) & self.nodes.__contains__(node_id2) & self.go[node_id1].__contains__(node_id2):
            del self.go[node_id1][node_id2]
            del self.back[node_id2][node_id1]
            self.mc = self.mc + 1
            return True
        return False

