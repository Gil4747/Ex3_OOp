from abc import ABC
from typing import List

import matplotlib
import numpy as np

from DiGraph import DiGraph
from GraphAlgoInterface import GraphAlgoInterface
from GraphInterface import GraphInterface
from nodeData import nodeData
import json
import math
import matplotlib.pyplot as plt
import random


class PriorityQueue(object):
    def __init__(self):
        self.queue = []

    # for checking if the queue is empty

    def is_empty(self):
        return len(self.queue) == 0

    # for inserting an element in the queue
    def insert(self, data):
        """
        Will place the vertex in the queue according to its distance. The smaller its distance, the closer it will get to the top of the column
        :param data: ths new value
        """
        self.queue.append(data)
        self.queue.sort(reverse=True, key=lambda nodeData: -nodeData.dist)

        # for popping an element based on Priority
    def peek(self):
        return self.queue[0]

    def delete(self):
        """
        # Deletes the first value in the queue
        :return:the first value in the queue
        """
        ans = self.queue[0]
        del self.queue[0]
        return ans


class GraphAlgo(GraphAlgoInterface):
    my_graph = DiGraph()
    def __init__(self, my_graph = DiGraph()):
        self.my_graph = my_graph

    def get_graph(self) -> GraphInterface:
        """
        Returns the graph
        :return: returns the graph
        """
        return self.my_graph

    def save_to_json(self, file_name: str) -> bool:
        """
        Saves the graph in JSON format to a file
        @param file_name: The path to the out file
        @return: True if the save was successful, False o.w.
        """
        nodes = self.my_graph.get_all_v()
        jsonObje = {} #A dictionary that will store all the information of the edge there
        jsonObjn = {} #A dictionary that will store all the information of the vertex there
        Edges = [] #An array that will contain all the information about the arcs
        Nodes = [] #An array that will contain all the information about the vertices
        jsonObj_ans = {} #A dictionary that will contain all the information about the vertices and arcs of the graph
        for i in nodes:
            jsonObjn.update({"id": i})
            Nodes.append(jsonObjn)  # add Nodes to jsonObjn
            jsonObjn = {}
            edges = self.my_graph.all_out_edges_of_node(i)
            for j in edges:
                jsonObje.update({"src": i})
                jsonObje.update({"w": edges[j]})
                jsonObje.update({"dest": j})
                Edges.append(jsonObje)  # add Edges to jsonObjn
                jsonObje = {}
        jsonObj_ans.update({"Edges": Edges})
        jsonObj_ans.update({"Nodes": Nodes})

        with open(file_name, 'w') as f: #Open the file and write to it
            json.dump(jsonObj_ans, f)

    def load_from_json(self, file_name: str) -> bool:
        """
        Loads a graph from a json file.
        @param file_name: The path to the json file
        @returns True if the loading was successful, False o.w.
        """
        with open(file_name, 'r') as f: #Opens the file and reads from it
            jsonObj = json.load(f)
        self.my_graph = DiGraph() #Set the graph as a new graph to receive the graph from the file
        nodes = jsonObj.get("Nodes") #An array that contains all the dictionaries that each dictionary contains vertex information
        for i in nodes:
            if ("pos" in i):
                s = i.get("pos").split(",")
                x=float(s[0]) #The x of the position of the vertex
                y=float(s[1]) #The y of the position of the vertex
                z=float(s[2]) #The z of the position of the vertex
                location=(x,y,z)
                # for the plot:
                if (x < self.my_graph.min_x): #Checks whether it is smaller than x smaller than the locations of the vertices
                    self.my_graph.min_x = x
                if (x > self.my_graph.max_x):
                    self.my_graph.max_x = x
                if (y < self.my_graph.min_y):
                    self.my_graph.min_y = y
                if (y > self.my_graph.max_y):
                    self.my_graph.max_y = y
                self.my_graph.add_node(i.get("id"), location)
            else:
                self.my_graph.add_node((i.get("id")))
        edges = jsonObj.get("Edges") #An array that contains all the dictionaries that each dictionary contains edge information
        for i in edges:
            src = i.get("src")
            dest = i.get("dest")
            w = i.get("w")
            # self.my_graph .go[src].update({dest: w})
            # self.my_graph .back[dest].update({src: w})
            self.my_graph.add_edge(src, dest, w)

        # self.my_graph = g

    def shortest_path(self, id1: int, id2: int) -> (float, list):
        """
        Returns the shortest path from node id1 to node id2 using Dijkstra's Algorithm
        @param id1: The start node id
        @param id2: The end node id
        @return: The distance of the path, a list of the nodes ids that the path goes through
        If there is no path between id1 and id2, or one of them dose not exist the function returns (float('inf'),[])
        """
        if ((self.my_graph.v_size() > 0)  & (id1 in self.my_graph.get_all_v())) & (id2 in self.my_graph.get_all_v()):
            visited = {} #To know which vertex I passed through and also all the neighbors coming out of it
            ans = []
            queue = PriorityQueue()
            start = self.my_graph.get_all_v().get(id1)  # start=id1
            s = nodeData(start.get_id(), start.get_pos(), start.get_tag(), start.get_dist())
            end = self.my_graph.get_all_v().get(id2)
            visited.update({id1: start})
            queue.insert(start)
            start.set_dist(0.0) #The distance of the vertex from the starting point
            start.set_tag(0)
            if(id1==id2):
                ans=(0.0,[id1])
                return ans

            while not queue.is_empty():
                start = queue.delete()
                visited.update({start.get_id(): start})
                edges_from_start = self.my_graph.all_out_edges_of_node(start.get_id())
                for n in edges_from_start:
                    if (n not in visited) & (self.my_graph.get_all_v().get(id2).get_dist() == 2147483648) \
                            | (self.my_graph.get_all_v().get(n).get_dist() > (start.get_dist() + edges_from_start[n])):
                        if self.my_graph.get_all_v().get(n).get_dist() > (start.get_dist() + edges_from_start[n]):
                            self.my_graph.get_all_v().get(n).set_dist(start.get_dist() + edges_from_start[n])#The distance of the vertex from the starting point
                            bla = self.my_graph.get_all_v().get(n)
                            bla2 = start.get_id()
                            bla.set_tag(bla2)#Name the id of the vertex from which it came out in the tag
                            queue.insert(self.my_graph.get_all_v().get(n))#Puts the neighbors into the priority queue
            if (end.get_tag() != -1.0) | (self.my_graph.get_all_v().get(id2).get_dist() != 2147483648):
                while (end.get_tag() in visited) & (end.get_id() != s.get_id()):
                    ans.append(end.get_id())#List of all vertices from id2 to id1
                    t = end.get_tag()
                    f = visited.get(t)
                    end = nodeData(f.get_id(), f.get_pos(), f.get_tag(), f.get_dist())
                ans.append(s.get_id())
                ans.reverse()#Makes the list the way from id1 to id2

                for i in self.my_graph.get_all_v():
                    self.my_graph.get_all_v().get(i).set_tag(-1) #Initializes the vertex tag
                    self.my_graph.get_all_v().get(i).set_dist(2147483648.0)#Initializes the dist of a vertex

                key = -1
                final_all = (math.inf, [])
                ans_dist = 0.0
                if (ans != None):
                    for i in ans:
                        #Calculate the path from id1 to id2 we found
                        if key == -1:
                            key = i
                            continue;
                        id = i
                        b = self.my_graph.all_out_edges_of_node(key).get(id)
                        ans_dist += b
                        key = i
                        final_all = (ans_dist, ans)#The distance and path from vertex id1 to id2
                    return final_all
            return (math.inf, [])

    def connected_components(self) -> List[list]:
        vis = self.dfs(self.my_graph.get_all_v().get(0).get_id(), self.my_graph)
        tg = self.transpose()
        lis = []
        i = 0

        while (vis.__len__() != 0):
            v = vis[i]
            vi = self.dfs(v.get_id(), tg)
            i += vi.__len__()
            li = list(vi.keys())
            lis += [li]
            for x in vi:
                del vis[x]
                tg.remove_node(x)

        return lis

    def connected_component(self, id1: int) -> list:
        lis = self.connected_components()
        for x in lis:
            for y in x:
                if y == id1:
                    return x

    def dfs(self, id1: int, graph: DiGraph) -> (dict):
        if (graph.v_size() > 0) & \
                ((id1 in graph.get_all_v()) & (len(graph.all_out_edges_of_node(id1)) != 0)):
            visited = {}
            queue = PriorityQueue()
            start = graph.get_all_v().get(id1)
            visited.update({id1: start})
            queue.insert(start)
            while (not queue.is_empty()):
                start = queue.delete()
                visited.update({start.get_id(): start})
                edges_from_start = graph.all_out_edges_of_node(start.get_id())
                for n in edges_from_start:
                    if (n not in visited) & (len(graph.all_in_edges_of_node(n)) != 0):
                        # print(n)
                        queue.insert(graph.get_all_v().get(n))
            return visited
        return {id1: graph.get_all_v().get(id1)}

    def transpose(self) -> (DiGraph):
        tg = DiGraph()
        for n in self.my_graph.get_all_v():
            tg.add_node(self.my_graph.get_all_v().get(n).get_id(), self.my_graph.get_all_v().get(n).get_pos())
        for node in self.my_graph.get_all_v():
            for e in self.my_graph.all_out_edges_of_node(node):
                tg.add_edge(self.my_graph.get_all_v().get(e).get_id(), self.my_graph.get_all_v().get(node).get_id(),
                            self.my_graph.all_out_edges_of_node(node).get(e))
        return tg

    def plot_graph(self) -> None:
        """
        Plots the graph.
        If the nodes have a position, the nodes will be placed there.
        Otherwise, they will be placed in a random but elegant manner.
        @return: None
        """
        x = []#For all the x of the vertices
        y = []#For all the y of the vertices
        z = []
        visited_x = {}#For all the x I've already been through
        visited_y = {}#For all the y I've already been through

        a = 0.0
        b = 0.0
        if (self.my_graph.get_all_v().get(0).get_pos() is None):
            fig = plt.figure(figsize=(5, 5)) #Opening a window for my graph
            ax2 =plt.axes()
            ax = fig.gca()
            random_number=2
            start_random=1

            for i in self.my_graph.get_all_v():
                if visited_x.get(i)==None:
                    a = random.uniform(1, 100000)
                    b = random.uniform(start_random, random_number)
                    start_random+=1
                    random_number+=1
                    visited_x.update({i: a})
                    visited_y.update({i: b})
                    x.append(a)
                    y.append(b)
                else:
                    x.append(visited_x.get(i))
                    y.append(visited_y.get(i))

                for j in self.my_graph.all_out_edges_of_node(i):
                    if visited_x.get(j)!=None: #If we have already drawn this vertex
                        plt.annotate(text='', xy=(visited_x[i],  visited_y[i]), xytext=(visited_x[j],  visited_y[j]), arrowprops=dict(arrowstyle='<-'))
                        plt.annotate(i, (visited_x[i], visited_y[i]), color='blue')#Writes the id of the vertex in the desired location
                        plt.annotate(j, (visited_x[j], visited_y[j]), color='blue')#Writes the id of the vertex in the desired location
                    else:#If we have not yet drawn this vertex
                        a = random.uniform(1, 100000)
                        b = random.uniform(start_random, random_number)
                        start_random += 1
                        random_number += 1
                        x.append(a)
                        y.append(b)
                        # dic = {0: a, 1: b}
                        visited_x.update({j: a})
                        visited_y.update({j: b})
                        plt.annotate(text='', xy=(visited_x[i], visited_y[i]), xytext=(visited_x[j], visited_y[j]),
                                     arrowprops=dict(arrowstyle='<-'))#Draws an arrow from point to point
                        plt.annotate(i, (visited_x[i], visited_y[i]),color='blue')#Writes the id of the vertex in the desired location
                        plt.annotate(j, (visited_x[j], visited_y[j]),color='blue')#Writes the id of the vertex in the desired location

        else:#If the vertices have positions on the graph
            fig = plt.figure()
            ax = fig.gca()
            ax.axis([self.my_graph.min_x - 0.001, self.my_graph.max_x + 0.001, self.my_graph.min_y - 0.001,
                     self.my_graph.max_y + 0.001])
            # fig, ax = plt.subplots()
            # ax.quiver([0,0], [0,0], [1, 0],  [1, -1],scale=5)
            # ax.quiver(35.22508, 32.10537, 35.20037, 32.09950)

            # ax.quiver(35.21217, 32.10537, 35.19698, 0.002)
            # ax2 = plt.axes(xlim=(self.min_x-0.001, (self.max_x+0.003)), ylim=(self.min_y, self.max_y))
            i_did_it={ }
            visited={}
            for i in self.my_graph.get_all_v():
                if visited.get(i) is None:
                    i_did_it[i]={}
                    x_i = self.my_graph.get_all_v().get(i).get_pos()[0]
                    y_i = self.my_graph.get_all_v().get(i).get_pos()[1]
                    visited.update({i:0})
                    x.append(x_i)
                    y.append(y_i)
                    plt.annotate(i, (x_i, y_i), color='blue')
                    # ax.scatter(x, y, c="red", s=30)
                else:
                    x_i = self.my_graph.get_all_v().get(i).get_pos()[0]
                    y_i = self.my_graph.get_all_v().get(i).get_pos()[1]

                # plt.annotate(i, (x_i, y_i), color='blue')
                for j in self.my_graph.all_out_edges_of_node(i):
                    if visited.get(j) is None:
                        i_did_it[j] = {}
                        x_j = self.my_graph.get_all_v().get(j).get_pos()[0]
                        y_j = self.my_graph.get_all_v().get(j).get_pos()[1]
                        visited.update({j: 0})
                        x.append(x_j)
                        y.append(y_j)
                        plt.annotate(j, (x_j, y_j), color='blue')
                        # ax.scatter(x, y, c="red", s=30)
                    else:
                        x_j = self.my_graph.get_all_v().get(j).get_pos()[0]
                        y_j = self.my_graph.get_all_v().get(j).get_pos()[1]
                    if(self.my_graph.all_out_edges_of_node(j).get(i) is not None) & (i_did_it.get(j).get(i) is None) & (i_did_it.get(i).get(j) is None):
                        i_did_it[j].update({i:0})
                        i_did_it[i].update({j: 0})
                        plt.annotate('', xy=(x_i, y_i), xytext=(x_j, y_j), arrowprops=dict(arrowstyle='<->'))
                    else:
                        if i_did_it.get(i).get(j) is None:
                            i_did_it[i].update({j: 0})
                            plt.annotate('', xy=(x_i, y_i), xytext=(x_j, y_j), arrowprops=dict(arrowstyle='<-'))




                    # plt.annotate(j, (x_j, y_j), color='blue')
                # plt.annotate(text='', xy=(x__, y__), xytext=(x__2, y__2), arrowprops=dict(arrowstyle='<-'))
        ax.scatter(x, y, c="red", s=30)
        # naming the x axis
        plt.xlabel('x - axis')
        # naming the y axis
        plt.ylabel('y - axis')

        # giving a title to my graph
        plt.title('My first graph!')

        # function to show the plot
        plt.show()
        return None

    # def point_equal(self, id, x, y, points) -> bool:
    #     count = 0
    #     for i in points.values():
    #         if (i.get(0) == x) & (i.get(1) == y):
    #             count2 = 0
    #             for j in points.keys():
    #                 if (count2 == count):
    #                     if (j != id):
    #                         return True
    #                     else:
    #                         break
    #                 count2 += 1
    #         count += 1
    #         # points.update({ii_id:ii})
    #     return False
