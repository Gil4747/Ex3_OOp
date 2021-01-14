#from geoLoction import  geoLoction
class nodeData:
    id = 0 #The id of the vertex
    tag = -1 #The id of his "dad" vertex on the way to a particular vertex, we will use it in shortest_path.
    dist = 2147483648.0 #The distance of the vertex from a particular vertex, we will use this in the function shortest_path.
    pos = () #The position of the vertex

    def __init__(self, id=id+1, pos=None, tag=-1, dist=2147483648.0):
        self.id=id
        self.pos = pos
        self.dist=dist
        self.tag = tag

    def get_id(self):
        """
        Returns the id of this vertex
        @return: the id of this vertex
        """
        return self.id

    def get_pos(self):
        """
         Returns the position of this vertex
        @return: the position of this vertex
        """
        return self.pos

    def get_tag(self):
        """
         Returns the tag of this vertex
        @return: the tag of this vertex
        """
        return self.tag

    def get_dist(self):
        """
        Returns The distance of the vertex from a particular vertex
        @return: The distance of the vertex from a particular vertex
        """
        return self.dist

    def set_tag(self, t):
        """
        :param t: the new value of the tag
        """
        self.tag = t

    def set_dist(self, d):
        """
        :param d: the new value of the dist
        """
        self.dist = d

    def set_pos(self, p):
        """
        :param p: the new value of the position
        """
        self.pos = p

    def __str__(self):
        ans = {}
        for i in self.get_all_v():
            s = str(i) + ": |edges out| " + str(self.all_out_edges_of_node(i).__len__())
            s += " |edges in| " + str(self.all_in_edges_of_node(i).__len__())
            ans.update({i: s})
        return str(ans)