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
        print("++++++++++++")
        print ((self.x-g.x) ** 2)
        print(((self.x-g.x) ** 2+(self.y-g.y) ** 2+(self.z-g.z))**(2))
        ans=((self.x-g.x) ** 2+(self.y-g.y) ** 2+(self.z-g.z))**(2)
        return ans

