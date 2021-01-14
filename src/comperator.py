import datetime
import math
import random

from numpy import source

from GraphAlgo import GraphAlgo as ga
import networkx as nx


class comperator:
    def _init_(self):
        g_al = ga()

    def tester(self, file_name: str) -> list:
        g_al = ga()
        g_al.load_from_json(file_name)
        gnx = self.ga2nx(g_al)
        rands = self.rand_in_range(g_al.my_graph.v_size())
        t1 = datetime.datetime.now().second + (60 * datetime.datetime.now().minute) + (datetime.datetime.now().microsecond/1000000)
        for i in rands:
            g_al.shortest_path(i[0], i[1]) != math.inf
        t2 = datetime.datetime.now().second + (60 * datetime.datetime.now().minute) + (datetime.datetime.now().microsecond/1000000)
        for i in rands:
            try:
                nx.shortest_path(gnx, i[0], i[1], weight='weight')
                print(i[0], i[1])
            except:
                print("fail")
                continue

        t3 = datetime.datetime.now().second + (60 * datetime.datetime.now().minute) + (datetime.datetime.now().microsecond/1000000)
        return [" my graph:" , t2 - t1," networkx graph:" , t3 - t2]

    def tester2(self, file_name: str):
        g_al = ga()
        g_al.load_from_json(file_name)
        gnx = self.ga2nx(g_al)
        t1 = datetime.datetime.now().second + (60 * datetime.datetime.now().minute) + (datetime.datetime.now().microsecond/1000000)
        g_al.connected_components()
        t2 = datetime.datetime.now().second + (60 * datetime.datetime.now().minute) + (datetime.datetime.now().microsecond/1000000)
        nx.strongly_connected_components(gnx)
        t3 = datetime.datetime.now().second + (60 * datetime.datetime.now().minute) + (datetime.datetime.now().microsecond/1000000)
        return [" my graph:" , t2 - t1," networkx graph:" , t3 - t2]


    def ga2nx(self, g_al: ga) -> nx.DiGraph:
        g = g_al.my_graph
        gnx = nx.DiGraph()
        if (g is None) | (g.get_all_v() is None) | (len(g.get_all_v()) == 0):
            return gnx
        for n in list(g.get_all_v().keys()):
            for e in list(g.all_out_edges_of_node(n).keys()):
                gnx.add_edge(n, e)
                gnx[n][e]["weight"] = g.all_out_edges_of_node(n).get(e)
        return gnx

    def rand_in_range(self, ran: int) -> list:
        ans = []
        n=10
        if ran >= 10000:
            n = 2
            if ran >= 10000:
                n = 1
        for i in range(1):
            sub = []
            a = random.randint(0, ran)
            b = random.randint(0, ran)
            sub.append(a)
            sub.append(b)
            ans += [sub]
        return ans