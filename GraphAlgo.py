from abc import ABC
from typing import List
from DiGraph import DiGraph
from GraphAlgoInterface import GraphAlgoInterface
from GraphInterface import GraphInterface
from geoLoction import geoLoction
from nodeData import nodeData
import json
import math
import matplotlib.pyplot as plt
import random



class PriorityQueue(object):
    def __init__(self):
        self.queue = []

    # def __str__(self):
    #     return ' '.join([str(i) for i in self.queue])

    # for checking if the queue is empty

    def is_empty(self):
        return len(self.queue) == 0

    # for inserting an element in the queue
    def insert(self, data):
        self.queue.append(data)

        self.queue.sort(reverse=True, key=lambda nodeData: -nodeData.dist)

        # for popping an element based on Priority

    def peek(self):
        return self.queue[0]

    def delete(self):
        ans = self.queue[0]
        del self.queue[0]
        return ans


class GraphAlgo(GraphAlgoInterface):

    my_graph = DiGraph()

    def __init__(self, my_graph=DiGraph()):
        self.my_graph = my_graph

    def get_graph(self) -> GraphInterface:
        return self.my_graph

    def save_to_json(self, file_name: str) -> bool:
        nodes = self.my_graph.get_all_v()
        jsonObje = {}
        jsonObjn = {}
        Edges = []
        Nodes = []
        jsonObj_ans = {}
        for i in nodes:
            jsonObjn.update({"id": i})
            Nodes.append(jsonObjn)
            jsonObjn = {}
            edges = self.my_graph.all_out_edges_of_node(i)
            for j in edges:
                jsonObje.update({"src": i})
                jsonObje.update({"w": edges[j]})
                jsonObje.update({"dest": j})
        jsonObj_ans.update({"Edges": Edges})
        jsonObj_ans.update({"Nodes": Nodes})

        with open(file_name, 'w') as f:
            json.dump(jsonObj_ans, f)

    def load_from_json(self, file_name: str) -> bool:
        with open(file_name, 'r') as f:
            jsonObj = json.load(f)
        g = DiGraph()
        nodes = jsonObj.get("Nodes")
        for i in nodes:
            if("pos" in i):
                s = i.get("pos").split(",");
                location1 = geoLoction(s[0],s[1],s[2])
                g.add_node(i.get("id"),location1)
            g.add_node((i.get("id")))
        edges = jsonObj.get("Edges")
        for i in edges:
            src = i.get("src")
            dest = i.get("dest")
            w = i.get("w")
            # g.go[src].update({dest: w})
            # g.back[dest].update({src: w})
            g.add_edge(src,dest,w)

        self.my_graph = g

    def shortest_path(self, id1: int, id2: int) -> (float, list):
            if (self.my_graph.v_size()>0  & id1 != id2 & self.my_graph.get_all_v().__contains__(id1) & self.my_graph.get_all_v().__contains__(id2)):
                visited = {}
                ans = []
                queue = PriorityQueue()
                start = self.my_graph.get_all_v().get(id1)
                s = nodeData(start.get_id(), start.get_pos(), start.get_tag(), start.get_dist())
                end = self.my_graph.get_all_v().get(id2)
                visited.update({id1: start})
                queue.insert(start)
                start.set_dist(0.0)
                start.set_tag(0)
                while (not queue.is_empty()):
                    start = queue.delete()
                    visited.update({start.get_id(): start})
                    edges_from_start = self.my_graph.all_out_edges_of_node(start.get_id())
                    for n in edges_from_start:
                        if (n not in visited) & (self.my_graph.get_all_v().get(id2).get_dist() == 2147483648) \
                                | (self.my_graph.get_all_v().get(n).get_dist() > (start.get_dist() + edges_from_start[n])):
                                self.my_graph.get_all_v().get(n).set_dist(start.get_dist() + edges_from_start[n])
                                bla= self.my_graph.get_all_v().get(n)
                                bla2=start.get_id()
                                bla.set_tag(bla2)
                                queue.insert(self.my_graph.get_all_v().get(n))
                if (end.get_tag() != -1.0) | (self.my_graph.get_all_v().get(id2).get_dist() != 2147483648):
                    while (end.get_tag() in visited) & (end.get_id() != s.get_id()):
                        ans.append(end.get_id())
                        t=end.get_tag()
                        f=visited.get(t)
                        end=nodeData(f.get_id(),f.get_pos(),f.get_tag(),f.get_dist())
                    ans.append(s.get_id())
                    ans.reverse()

                    for i in self.my_graph.get_all_v():
                        self.my_graph.get_all_v().get(i).set_tag(0)
                        self.my_graph.get_all_v().get(i).set_dist(2147483648.0)

                    key = -1
                    final_all = (math.inf, [])
                    ans_dist = 0.0
                    if (ans != None):
                        for i in ans:
                            if key == -1:
                                key = i
                                continue;
                            id=i
                            b=self.my_graph.all_out_edges_of_node(key).get(id)
                            ans_dist += b
                            key = i
                            final_all = (ans_dist, ans)
                        return final_all
                return (math.inf, [])

    def connected_components(self) -> List[list]:
        vis = self.dfs(self.my_graph.get_all_v().get(0).get_id(),self.my_graph)
        tg= self.transpose()
        lis = []
        while (vis.__len__() is not 0) :
            v=vis[(vis.__len__()-1)]
            vi=self.dfs(v.get_id(),self.my_graph)
            lis += vi
            for x in vi:
                if(x in vis.keys()):
                    del vis[x]

        return lis

    def dfs(self, id1: int, graph: DiGraph) -> (dict):
            if (self.my_graph.v_size() > 0) & (id1 in self.my_graph.get_all_v().keys()):
                visited = {}
                queue = PriorityQueue()
                start = self.my_graph.get_all_v().get(id1)
                visited.update({id1: start})
                queue.insert(start)
                while (not queue.is_empty()):
                    start = queue.delete()
                    visited.update({start.get_id(): start})
                    edges_from_start = self.my_graph.all_out_edges_of_node(start.get_id())
                    for n in edges_from_start:
                        if (n not in visited):
                            queue.insert(self.my_graph.get_all_v().get(n))
                return visited
            return {}

    def transpose(self) -> (DiGraph):
     graph=self.my_graph
     tg=DiGraph()
     for n in graph.get_all_v():
         tg.add_node(self.my_graph.get_all_v().get(n).get_id(), self.my_graph.get_all_v().get(n).get_pos())
     for node in graph.get_all_v():
         for e in graph.all_out_edges_of_node(node):
            tg.add_edge(graph.get_all_v().get(e),self.my_graph.get_all_v().get(node).get_id(),self.my_graph.all_out_edges_of_node(node).get(e))
     return tg

    def plot_graph(self) -> None:
        point1 = []
        point2 = []
        x=[]
        y=[]
        z = []
        count=0
        for i in self.my_graph.get_all_v():
            if(self.my_graph.get_all_v().get(i).get_pos() is None):
                a=random.randint(1,3)
                b=random.randint(1,3)
                point1.append(a)
                point2.append(b)
                x_value = [point1[count]]
                y_value = [point2[count]]
                count+=1
                plt.plot(x_value[0],y_value[0], 'r-o')
                for j in self.my_graph.all_out_edges_of_node(i):
                    a = random.randint(1, 3)
                    b = random.randint(1, 3)
                    point1.append(a)
                    point2.append(b)
                    x_values = [point1[count]]
                    y_values=[point2[count]]
                    count += 1
                    plt.plot(x_values[0],y_values[0], 'r-o')
                    plt.plot(point1,point2, 'r-o')

            else:
                x.append(self.my_graph.get_all_v().get(i).get_pos().x)
                y.append(self.my_graph.get_all_v().get(i).get_pos().y)
                z.append(self.my_graph.get_all_v().get(i).get_pos().z)

                # plotting the points
                plt.plot(x, y, z)

        # naming the x axis
        plt.xlabel('x - axis')
        # naming the y axis
        plt.ylabel('y - axis')


        # giving a title to my graph
        plt.title('My first graph!')

        # function to show the plot
        plt.show()