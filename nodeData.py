from geoLoction import  geoLoction
class nodeData:
    id = 0
    tag = -1
    dist = 2147483648.0
    pos = None

    def __init__(self, id=id+1, pos=geoLoction(), tag=-1, dist=2147483648.0):
        self.id=id
        self.pos = pos
        self.dist=dist
        self.tag = tag

    def get_id(self):
        return self.id

    def get_pos(self):
        return self.pos

    def get_tag(self):
        return self.tag

    def get_dist(self):
        return self.dist

    def set_tag(self, t):
        self.tag = t

    def set_dist(self, d):
        self.dist = d