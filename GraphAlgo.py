from abc import ABC
from typing import List

import matplotlib
import numpy as np

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
    max_x=0.0
    max_y=0.0
    min_x=2147483648.0
    min_y=2147483648.0
    my_graph = DiGraph()

    def __init__(self, my_graph=DiGraph(),max_x=0.0,max_y=0.0,min_x=2147483648.0,min_y=2147483648.0):
        self.my_graph = my_graph
        self.max_x=max_x
        self.max_y=max_y
        self.min_x=min_x
        self.min_y=min_y

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
                Edges.append(jsonObje)
                jsonObje = {}
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
                if(float(s[0])<self.min_x):
                    self.min_x=float(s[0])
                if (float(s[0]) > self.max_x):
                    self.max_x = float(s[0])
                if (float(s[1]) < self.min_y):
                    self.min_y = float(s[1])
                if(float(s[1]) > self.max_y):
                    self.max_y = float(s[1])
                location1 = geoLoction(float(s[0]),float(s[1]),float(s[2]))
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
        i=0
        while (vis._len_() != 0) :
            v=vis[i]
            vi=self.dfs(v.get_id(),tg)
            i+=vi._len_()
            li=list(vi.keys())
            lis += [li]
            for x in vi:
               del vis[x]
               tg.remove_node(x)

        return lis

    def connected_component(self, id1: int) -> list:
        lis=self.connected_components()
        for x in lis:
            for y in x:
                if y==id1:
                    return x

    def dfs(self, id1: int, graph: DiGraph) -> (dict):
            if (graph.v_size() > 0) &\
                    ((id1 in graph.get_all_v()) & (len(graph.all_out_edges_of_node(id1)) !=0) ):
                visited = {}
                queue = PriorityQueue()
                start = graph.get_all_v().get(id1)
                visited.update({id1: start})
                queue.insert(start)
                while (not queue.is_empty()):
                    start = queue.delete()
                    visited.update({start.get_id(): start})
                    edges_from_start =graph.all_out_edges_of_node(start.get_id())
                    for n in edges_from_start:
                        if (n not in visited)&(len(graph.all_in_edges_of_node(n))!=0):
                            print(n)
                            queue.insert(graph.get_all_v().get(n))
                return visited
            return {id1:graph.get_all_v().get(id1)}

    def transpose(self) -> (DiGraph):
     tg=DiGraph()
     for n in self.my_graph.get_all_v():
         tg.add_node(self.my_graph.get_all_v().get(n).get_id(), self.my_graph.get_all_v().get(n).get_pos())
     for node in self.my_graph.get_all_v():
         for e in self.my_graph.all_out_edges_of_node(node):
            tg.add_edge(self.my_graph.get_all_v().get(e).get_id(),self.my_graph.get_all_v().get(node).get_id(),self.my_graph.all_out_edges_of_node(node).get(e))
     return tg

    def plot_graph(self) -> None:
        points = {}
        x = []
        y = []
        z = []
        visited_x={}
        visited_y={}

        count = 0
        a=0.0
        b=0.0
        # for i in self.my_graph.get_all_v():
        if (self.my_graph.get_all_v().get(0).get_pos() is None):
            fig = plt.figure(figsize=(5, 5))
            ax2 = plt.axes(xlim=(0, 4), ylim=(0, 4))
            ax = fig.gca()

            for i in self.my_graph.get_all_v():
                if (i not in visited_x.keys()):
                    a = random.randint(1, 3)+0.5
                    b = random.randint(1, 3)+0.5
                    if(points.get(0)!=None):
                        if a==b:
                            while ((a == b)):
                                a = random.randint(1, 3) + 0.5
                                b = random.randint(1, 3) + 0.5
                        if (self.point_equal(i,a,b,points)==True):
                            while ((self.point_equal(i,a,b,points)==True)):
                                a = random.randint(1, 3)+0.5
                                b = random.randint(1, 3)+0.5
                                if a == b:
                                    while ((a == b)):
                                        a = random.randint(1, 3) + 0.5
                                        b = random.randint(1, 3) + 0.5
                    else:
                        if a == b:
                            while a == b:
                                a = random.randint(1, 3)+0.5
                                b = random.randint(1, 3)+0.5
                    x.append(a)
                    y.append(b)
                    dic = {0: a, 1: b}
                    print(dic.get(0), dic.get(1))
                    visited_x.update({i: a})
                    visited_y.update({i: b})
                    points.update({i: dic})
                    ax.scatter(x,y, c="red", s=30)
                    plt.annotate(i, (visited_x[i], visited_y[i]))

                for j in self.my_graph.all_out_edges_of_node(i):
                        if ((j in visited_x.keys()) & (j in visited_y.keys())):
                            # plt.scatter(x, y)
                            # ax.scatter(x, y, c="red", s=30)
                           # ax = plt.axes(xlim=(0, 4), ylim=(0, 4))
                            #plt.annotate(i, (visited_x[i], visited_y[i]))
                            ax.arrow(visited_x[i], visited_y[i], visited_x[j] - visited_x[i], visited_y[j] - visited_y[i],
                                     head_width=0.05, head_length=0.1, fc='k', ec='k')

                        else:
                            a = random.randint(1, 3)+0.5
                            b = random.randint(1, 3)+0.5
                            if (points.get(0) != None):
                                if a == b:
                                    while ((a == b)):
                                        a = random.randint(1, 3) + 0.5
                                        b = random.randint(1, 3) + 0.5
                                if (self.point_equal(j,a,b, points) == True):
                                    while ((self.point_equal(j, a,b, points) == True)):
                                        a = random.randint(1, 3) + 0.5
                                        b = random.randint(1, 3) + 0.5
                                        if a == b:
                                            while ((a == b)):
                                                a = random.randint(1, 3) + 0.5
                                                b = random.randint(1, 3) + 0.5
                            else:
                                if a == b:
                                    while a == b:
                                        a = random.randint(1, 3)
                                        b = random.randint(1, 3)
                            x.append(a)
                            y.append(b)
                            dic={0:a,1:b}
                            print(dic.get(0),dic.get(1))
                            visited_x.update({j: a})
                            visited_y.update({j: b})
                            points.update({j: dic})
                            ax.scatter(x, y, c="red", s=30)
                           # ax = plt.axes(xlim=(0, 4), ylim=(0, 4))
                            plt.annotate(j, (visited_x[j], visited_y[j]))
                            ax.arrow(visited_x[i],visited_y[i],visited_x[j]-visited_x[i],visited_y[j]-visited_y[i], head_width=0.05, head_length=0.1, fc='k', ec='k')
    ##################################################

        else:
            # x_ = np.arange(self.min_x,  (self.max_x), 0.25)
            # y_ = np.arange(self.min_y, self.max_y, 0.25)
            # x_, y_ = np.meshgrid(x_, y_)
            fig = plt.figure()
            ax2 = plt.axes(xlim=(self.min_x-0.001, (self.max_x+0.003)), ylim=(self.min_y-0.001, self.max_y+0.001))
            ax = fig.gca()
            for i in self.my_graph.get_all_v():
                x.append(self.my_graph.get_all_v().get(i).get_pos().x)
                y.append(self.my_graph.get_all_v().get(i).get_pos().y)
                visited_x.update({i: self.my_graph.get_all_v().get(i).get_pos().x})
                visited_y.update({i: (self.my_graph.get_all_v().get(i).get_pos().y)})
                #points.update({i: dic})
                ax.scatter(x, y, c="red", s=30)
                plt.annotate(i, (visited_x[i], visited_y[i]))

                for j in self.my_graph.all_out_edges_of_node(i):
                    if ((j in visited_x.keys()) & (j in visited_y.keys())):
                        plt.plot(x, y, '-', 'k')
                        # ax.arrow(round(self.my_graph.get_all_v().get(i).get_pos().x,5),
                        #          round(self.my_graph.get_all_v().get(i).get_pos().y,5),
                        #          (round(self.my_graph.get_all_v().get(j).get_pos().x-self.my_graph.get_all_v().get(i).get_pos().x),5),
                        #          (round(self.my_graph.get_all_v().get(j).get_pos().y-self.my_graph.get_all_v().get(i).get_pos().y),5),
                        #          head_width=0.0, head_length=0.0, fc='k', ec='k')

                    else:
                        x.append((self.my_graph.get_all_v().get(j).get_pos().x))
                        y.append(self.my_graph.get_all_v().get(j).get_pos().y)
                        visited_x.update({j: (self.my_graph.get_all_v().get(j).get_pos().x)})
                        visited_y.update({j: self.my_graph.get_all_v().get(j).get_pos().y})
                        ax.scatter(x, y, c="red", s=30)
                        # ax = plt.axes(xlim=(0, 4), ylim=(0, 4))
                        print(round(self.my_graph.get_all_v().get(i).get_pos().x,5),round(self.my_graph.get_all_v().get(i).get_pos().y,5))
                        print(round(self.my_graph.get_all_v().get(j).get_pos().x,5), round((self.my_graph.get_all_v().get(j).get_pos().y ),5))
                        plt.annotate(j, (visited_x[j], visited_y[j]))
                        #plt.plot(round(self.my_graph.get_all_v().get(i).get_pos().x,5), round(self.my_graph.get_all_v().get(i).get_pos().y,5),round((self.my_graph.get_all_v().get(j).get_pos().x-self.my_graph.get_all_v().get(i).get_pos().x ),5),round((self.my_graph.get_all_v().get(j).get_pos().y-self.my_graph.get_all_v().get(i).get_pos().y),5), 'g')
                        plt.plot(x,y,'-','k')
                        #ax.arrow(round(self.my_graph.get_all_v().get(i).get_pos().x,5),round(self.my_graph.get_all_v().get(i).get_pos().y,5), round((self.my_graph.get_all_v().get(j).get_pos().x-self.my_graph.get_all_v().get(i).get_pos().x ),5), round((self.my_graph.get_all_v().get(j).get_pos().y-self.my_graph.get_all_v().get(i).get_pos().y),5),
                            #head_width=0.0, head_length=0.0, fc='k', ec='k')


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

    def point_equal(self, id, x , y, points) -> bool:
        count=0
        for i in points.values():
            if (i.get(0)==x) & (i.get(1)==y):
                count2=0
                for j in points.keys():
                    if(count2==count):
                        if(j!=id):
                            return True
                        else:
                            break
                    count2+=1
            count+=1
            # points.update({ii_id:ii})
        return False
