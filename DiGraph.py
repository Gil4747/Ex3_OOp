from GraphInterface import GraphInterface
from nodeData import nodeData


class DiGraph(GraphInterface):
    edges = 0
    nodes = {}
    go = {}
    back = {}
    mc = 0
    max_x = -2147483648.0  # minimum value in python
    max_y = -2147483648.0
    min_x = 2147483648.0  # maximum value in python
    min_y = 2147483648.0

    """This abstract class represents an interface of a graph."""

    def __init__(self, max_x=-2147483648.0, max_y=-2147483648.0, min_x=2147483648.0, min_y=2147483648.0):
        self.edges = 0
        self.nodes = {} #A dictionary whose key is the id of the vertex and the value is the vertex
        self.go = {} # all the neighbors that goes out from the node
        self.back = {} # all the neighbors that goes in from the node
        self.mc = 0
        self.max_x = max_x #The largest x of all the positions of the vertices in the graph, is for the plot_graph function
        self.max_y = max_y #The largest y of all the positions of the vertices in the graph, is for the plot_graph function
        self.min_x = min_x #The smallest x of all the positions of the vertices in the graph is for the plot_graph function
        self.min_y = min_y #The smallest y of all the positions of the vertices in the graph is for the plot_graph function

    def v_size(self) -> int:
        """
        Returns the number of vertices in this graph
        @return: The number of vertices in this graph
        """
        return self.nodes.__len__()

    def e_size(self) -> int:
        """
        Returns the number of edges in this graph
        @return: The number of edges in this graph
        """
        return self.edges

    def get_all_v(self) -> dict:
        """return a dictionary of all the nodes in the Graph"""
        return self.nodes

    def all_in_edges_of_node(self, id1: int) -> dict:
        """return a dictionary of all the nodes connected to (into) node_id """
        return self.back[id1]

    def all_out_edges_of_node(self, id1: int) -> dict:
        """return a dictionary of all the nodes connected from node_id , each node is represented using a pair"""
        return self.go[id1]

    def get_mc(self) -> int:
        """
          Returns the current version of this graph,
          on every change in the graph state - the MC should be increased
          @return: The current version of this graph.
          """
        return self.mc

    def add_edge(self, id1: int, id2: int, weight: float) -> bool:
        """
        Adds an edge to the graph.
        @param id1: The start node of the edge
        @param id2: The end node of the edge
        @param weight: The weight of the edge
        @return: True if the edge was added successfully, False o.w.
        Note: If the edge already exists or one of the nodes dose not exists the functions will do nothing
        """
        if (self.nodes.get(id1)!=None) & (self.nodes.get(id2)!=None):
            if self.go[id1].get(id2)!=None:
                return False
            self.go[id1].update({id2: weight})
            self.back[id2].update({id1: weight})
            self.edges = self.edges + 1
            self.mc = self.mc + 1
            return True
        return False

    def add_node(self, node_id: int, pos: tuple = None) -> bool:
        """
         Adds a node to the graph.
        @param node_id: The node ID
        @param pos: The position of the node
        @return: True if the node was added successfully, False o.w.
        Note: if the node id already exists the node will not be added
         """
        if self.nodes.get(node_id)!=None:
            return False
        if (pos is not None):
            if (pos[0] < self.min_x):
                self.min_x = pos[0]
            if (pos[0] > self.max_x):
                self.max_x = pos[0]
            if (pos[1] < self.min_y):
                self.min_y = pos[1]
            if (pos[1] > self.max_y):
                self.max_y = pos[1]
        n = nodeData(node_id, pos)
        self.nodes.update({node_id: n})
        self.go[node_id] = {}
        self.back[node_id] = {}
        self.mc = self.mc + 1
        return True

    def remove_node(self, node_id: int) -> bool:
        """
        Removes a node from the graph.
        @param node_id: The node ID
        @return: True if the node was removed successfully, False o.w.
        Note: if the node id does not exists the function will do nothing
        """
        if self.nodes.get(node_id)!=None:
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
        """
        Removes an edge from the graph.
        @param node_id1: The start node of the edge
        @param node_id2: The end node of the edge
        @return: True if the edge was removed successfully, False o.w.
        Note: If such an edge does not exists the function will do nothing
        """
        if (self.nodes.get(node_id1)!=None) & (self.nodes.get(node_id2)!=None) & (self.go[node_id1].get(node_id2)!=None):
            del self.go[node_id1][node_id2]
            del self.back[node_id2][node_id1]
            self.mc = self.mc + 1
            self.edges-=1
            return True
        return False
