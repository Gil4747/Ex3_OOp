import math
import unittest
import networkx
from DiGraph import DiGraph
from GraphAlgo import GraphAlgo


class MyTestCase(unittest.TestCase):
     def test_save_and_load(self):
        g_algo = GraphAlgo()
        i = 0
        while i < 4:
            g_algo.my_graph.add_node(i)
            i += 1
        g_algo.my_graph.add_edge(0, 1, 4.8)
        g_algo.my_graph.add_edge(0, 2, 5.0)
        g_algo.my_graph.add_edge(1, 3, 4.1)
        g_algo.my_graph.add_edge(1, 0, 4.0)
        f = "C:\\json_file\\Ex3_save.json"
        g_algo.save_to_json(f)
        g_algo2 = GraphAlgo()
        g_algo2.load_from_json(f)
        self.assertEqual(g_algo2.my_graph.v_size(),g_algo2.my_graph.v_size())
        self.assertEqual(g_algo2.my_graph.edges, g_algo2.my_graph.edges)
        for i in range(0,4):
            self.assertEqual(g_algo2.my_graph.get_all_v().get(i).get_id(),g_algo.my_graph.get_all_v().get(i).get_id())
            self.assertEqual(g_algo2.my_graph.get_all_v().get(i).get_dist(), g_algo.my_graph.get_all_v().get(i).get_dist())
            self.assertEqual(g_algo2.my_graph.get_all_v().get(i).get_tag(),g_algo.my_graph.get_all_v().get(i).get_tag())
        g_algo2.plot_graph()
        file = "C:\\json_file\\G_10_80_2.json"
        g_algo.load_from_json(file)
        g_algo.plot_graph()

     def test_shortest_path(self):
         g_algo = GraphAlgo()
         g_algo.my_graph = DiGraph()
         i = 0
         while i < 4:
             l =(i, 0, 0);
             g_algo.my_graph.add_node(i, l)
             i += 1
         g_algo.my_graph.add_edge(0, 1, 4.2)
         g_algo.my_graph.add_edge(1, 2, 0.1)
         g_algo.my_graph.add_edge(2, 3, 0.2)
         g_algo.my_graph.add_edge(0, 2, 5.0)
         g_algo.my_graph.add_edge(3, 0, 4.5)
         g_algo.my_graph.add_edge(1, 3, 10.0)
         dist, path = g_algo.shortest_path(0, 0)
         self.assertEqual(0, dist)
         ans = [0]
         for i in path:
             self.assertEqual(i, ans[i])
         dist, path = g_algo.shortest_path(0, 3)
         self.assertEqual(4.5, dist)
         ans = [0, 1, 2, 3]
         for i in path:
             self.assertEqual(i, ans[i])
         g_algo.plot_graph()
         i=4
         while i < 15:
             l =(i, 0, 0);
             g_algo.my_graph.add_node(i, l)
             i += 1
         g_algo.my_graph.add_edge(3, 4, 4.2)
         g_algo.my_graph.add_edge(1, 8, 0.1)
         g_algo.my_graph.add_edge(2, 7, 0.2)
         g_algo.my_graph.add_edge(6, 2, 5.0)
         g_algo.my_graph.add_edge(7, 0, 4.5)
         g_algo.my_graph.add_edge(4, 8, 10.0)
         g_algo.my_graph.add_edge(3, 9, 4.2)
         g_algo.my_graph.add_edge(1, 10, 0.1)
         g_algo.my_graph.add_edge(2, 11, 0.2)
         g_algo.my_graph.add_edge(6, 14, 5.0)
         g_algo.my_graph.add_edge(14, 0, 4.5)
         g_algo.my_graph.add_edge(4, 11, 10.0)
         g_algo.my_graph.add_edge(11, 1, 0.1)
         g_algo.my_graph.add_edge(2, 12, 0.2)
         g_algo.my_graph.add_edge(12, 14, 5.0)
         g_algo.my_graph.add_edge(13, 0, 4.5)
         g_algo.my_graph.add_edge(9, 7, 10.0)
         g_algo.my_graph.add_edge(12, 13, 10.0)
         g_algo.my_graph.add_edge(14, 13, 4.0)
         dist, path = g_algo.shortest_path(12, 13)
         self.assertEqual(9.0, dist)
         ans = [12,14,13]
         for i in range(0,path.__len__()):
             self.assertEqual(path[i], ans[i])

         dist, path = g_algo.shortest_path(3, 5)
         self.assertEqual(math.inf, dist)
         ans = []
         self.assertEqual(path, ans)
         g_algo.plot_graph()
         file = "C:\\json_file\\G_10_80_2.json"
         g_algo.load_from_json(file)
         g_algo.my_graph.remove_edge(13, 14)
         dist, path = g_algo.shortest_path(1, 7)
         self.assertEqual(1,path[0])
         self.assertEqual(7, path[1])

         print(dist, path)
         dist, path = g_algo.shortest_path(47, 19)
         print(dist, path)

     def test_connected_component(self):
         g_algo = GraphAlgo()
         i = 0
         while i < 4:
             g_algo.my_graph.add_node(i)
             i += 1
         g_algo.my_graph.add_edge(0, 1, 4.8)
         g_algo.my_graph.add_edge(0, 2, 5.0)
         g_algo.my_graph.add_edge(1, 3, 4.1)
         g_algo.my_graph.add_edge(1, 0, 4.0)
         lis=g_algo.connected_component(0)
         print(lis)
         assert_lis=[0,1]
         for i in lis:
            self.assertEqual(i,assert_lis[i])


     def test_connected_components(self):
         g_algo = GraphAlgo()
         i = 0
         while i < 4:
             g_algo.my_graph.add_node(i)
             i += 1
         g_algo.my_graph.add_edge(0, 1, 4.8)
         g_algo.my_graph.add_edge(0, 2, 5.0)
         g_algo.my_graph.add_edge(1, 3, 4.1)
         g_algo.my_graph.add_edge(1, 0, 4.0)


     def test_plot(self):
         g_algo = GraphAlgo()
         g_algo.my_graph = DiGraph()
         g_algo.plot_graph()




if __name__ == '__main__':
    unittest.main()
