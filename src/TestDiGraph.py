import unittest

from DiGraph import DiGraph

class TestDigraph(unittest.TestCase):
    def test_get_all_v(self):
        g = DiGraph()
        self.assertTrue(g.get_all_v().__len__()==0)

    def test_all_in_edges_of_node(self):
        g = DiGraph()
        g.add_node(0)
        self.assertTrue(len(g.all_in_edges_of_node(0)) == 0)

    def add_edge(self):
        g = DiGraph()
        i = 0
        while i < 4:
            g.add_node(i)
            i += 1
        g.add_edge(0, 1, 4.8)
        g.add_edge(0, 2, 5.0)
        g.add_edge(1, 3, 4.1)
        g.add_edge(1, 0, 4.0)
        g.add_edge(2, 3, 3.8)
        g.add_edge(3, 2, 5.0)
        g.add_edge(2, 1, 8.0)
        g.add_edge(0, 1, 1.8)
        self.assertTrue(g.edges == 7)

    def test_add_node(self):
        g = DiGraph()
        i = 0
        while i < 4:
            g.add_node(i)
            i += 1
        self.assertEqual(g.v_size(), 4)
        g.add_node(1)
        self.assertEqual(g.v_size(), 4)

    def test_remove_node(self):
        g = DiGraph()
        i = 0
        while i < 4:
            g.add_node(i)
            i += 1
        self.assertEqual(g.v_size(), 4)
        g.remove_node(0)
        g.remove_node(0)
        self.assertEqual(g.v_size(), 3)

    def test_remove_edge(self):
        g = DiGraph()
        i = 0
        while i < 4:
            g.add_node(i)
            i += 1
        g.add_edge(0, 1, 4.8)
        g.add_edge(0, 2, 5.0)
        g.add_edge(1, 3, 4.1)
        g.add_edge(1, 0, 4.0)
        g.add_edge(2, 3, 3.8)
        g.add_edge(3, 2, 5.0)
        g.add_edge(0, 1, 1.8)
        g.remove_edge(2,1)
        self.assertTrue(g.edges == 6)
        g.remove_edge(2, 3)
        self.assertEqual(g.all_out_edges_of_node(2).get(3) , None)
        self.assertTrue(g.all_in_edges_of_node(3).get(2) is None)
        self.assertTrue(g.edges == 5)
        g.remove_edge(1, 2)
        self.assertTrue(g.edges == 5)

    def test_mc(self):
        g = DiGraph()
        i = 0
        while i < 4:
            g.add_node(i)
            i += 1
        g.add_edge(0, 1,0.8)
        g.add_edge(0, 2,2.3)
        g.add_edge(1, 3, 1.0)
        g.add_edge(3, 1, 9.0)
        self.assertEqual(g.get_mc(), 8)
        g.remove_edge(1, 0)
        self.assertEqual(g.get_mc(), 8)
        g.add_edge(0, 1, 0.1)
        self.assertEqual(g.get_mc(), 8)
        g.remove_edge(1, 3)
        self.assertEqual(g.get_mc(), 9)

if __name__ == '__main__':
    unittest.main()


