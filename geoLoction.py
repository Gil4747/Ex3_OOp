import math
class geoLoction:
    x = 0
    y = 0
    z = 0

    def __init__(self,x=0,y=0,z=0):
        self.x=x
        self.y = y
        self.z = z
    def distance(self, g):
        return math.sqrt(math.pow(self.x-g.x,2)+math.pow(self.y-g.y)+math.pow(self.z-g.z,2))

